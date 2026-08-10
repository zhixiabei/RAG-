from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import unicodedata
from urllib.parse import quote

import httpx

DEFAULT_DATASET = "heishanliang_rag_eval_v1.0.0.jsonl"


class EvaluationError(RuntimeError):
    pass


def precision_recall_f1(expected: set[str], actual: set[str]) -> dict[str, float | None]:
    if not expected:
        return {"precision": None, "recall": None, "f1": None}
    true_positives = len(expected & actual)
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def answer_char_f1(expected_answer: str | None, actual_answer: str | None) -> float | None:
    expected_tokens = _answer_tokens(expected_answer or "")
    actual_tokens = _answer_tokens(actual_answer or "")
    if not expected_tokens:
        return None
    if not actual_tokens:
        return 0.0
    overlap = sum((Counter(expected_tokens) & Counter(actual_tokens)).values())
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    if not precision + recall:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def measure_retrieval_hits(
    sample: dict,
    response: dict,
) -> dict:
    citations = response.get("citations") or []
    retrieved_document_ids = set(response.get("retrieved_document_ids") or []) or {
        citation.get("document_id") for citation in citations if citation.get("document_id")
    }
    retrieved_chunk_ids = set(response.get("retrieved_chunk_ids") or []) or {
        citation.get("chunk_id") for citation in citations if citation.get("chunk_id")
    }
    expected_document_ids = set(sample.get("source_document_ids") or [])
    expected_chunk_ids = set(sample.get("source_chunk_ids") or [])

    document_hit = (
        bool(expected_document_ids & retrieved_document_ids)
        if expected_document_ids
        else None
    )
    chunk_hit = (
        bool(expected_chunk_ids & retrieved_chunk_ids)
        if expected_chunk_ids
        else None
    )
    document_metrics = precision_recall_f1(expected_document_ids, retrieved_document_ids)
    chunk_metrics = precision_recall_f1(expected_chunk_ids, retrieved_chunk_ids)
    generated_answer = str(response.get("answer") or "")

    return {
        "question_id": sample.get("question_id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "difficulty": sample.get("difficulty"),
        "document_hit": document_hit,
        "chunk_hit": chunk_hit,
        "document_precision": document_metrics["precision"],
        "document_recall": document_metrics["recall"],
        "document_f1": document_metrics["f1"],
        "chunk_precision": chunk_metrics["precision"],
        "chunk_recall": chunk_metrics["recall"],
        "chunk_f1": chunk_metrics["f1"],
        "answer_f1": answer_char_f1(sample.get("expected_answer"), generated_answer),
        "answer": generated_answer,
        "retrieval_used": bool(response.get("retrieval_used")),
        "retrieved_count": int(response.get("retrieved_count") or 0),
        "retrieved_document_ids": sorted(retrieved_document_ids),
        "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
        "expected_document_ids": sorted(expected_document_ids),
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "response_time_ms": _optional_number(
            response.get("client_response_time_ms", response.get("response_time_ms"))
        ),
        "server_response_time_ms": _optional_number(response.get("response_time_ms")),
        "token_usage": response.get("token_usage") or {"available": False},
    }


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(values: list[bool]) -> float | None:
    return _average([1.0 if value else 0.0 for value in values])


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def summarize(results: list[dict]) -> dict:
    completed = [result for result in results if "error" not in result]
    latencies = [
        float(result["response_time_ms"])
        for result in completed
        if _optional_number(result.get("response_time_ms")) is not None
    ]
    reported_usage = [
        result.get("token_usage") or {}
        for result in completed
        if (result.get("token_usage") or {}).get("available")
    ]
    return {
        "sample_count": len(results),
        "completed_count": len(completed),
        "error_count": len(results) - len(completed),
        "document_hit_rate": _rate([
            result["document_hit"]
            for result in completed
            if result.get("document_hit") is not None
        ]),
        "chunk_hit_rate": _rate([
            result["chunk_hit"]
            for result in completed
            if result.get("chunk_hit") is not None
        ]),
        "document_precision": _average_metric(completed, "document_precision"),
        "document_recall": _average_metric(completed, "document_recall"),
        "document_f1": _average_metric(completed, "document_f1"),
        "chunk_precision": _average_metric(completed, "chunk_precision"),
        "chunk_recall": _average_metric(completed, "chunk_recall"),
        "chunk_f1": _average_metric(completed, "chunk_f1"),
        "answer_f1": _average_metric(completed, "answer_f1"),
        "average_response_time_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_response_time_ms": _percentile(latencies, 0.50),
        "p95_response_time_ms": _percentile(latencies, 0.95),
        "p99_response_time_ms": _percentile(latencies, 0.99),
        "token_usage_sample_count": len(reported_usage),
        "input_tokens": sum(int(usage.get("input_tokens") or 0) for usage in reported_usage),
        "output_tokens": sum(int(usage.get("output_tokens") or 0) for usage in reported_usage),
        "total_tokens": sum(int(usage.get("total_tokens") or 0) for usage in reported_usage),
        "average_tokens": (
            round(
                sum(int(usage.get("total_tokens") or 0) for usage in reported_usage)
                / len(reported_usage),
                2,
            )
            if reported_usage
            else None
        ),
    }


def _answer_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", normalized)


def _optional_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _average_metric(results: list[dict], key: str) -> float | None:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    return _average(values)


def load_dataset(path: Path, include_non_approved: bool = False) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"数据集第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not sample.get("question_id") or not sample.get("question"):
                raise EvaluationError(
                    f"数据集第 {line_number} 行缺少 question_id 或 question"
                )
            if include_non_approved or sample.get("status") in {None, "approved"}:
                samples.append(sample)
    if not samples:
        raise EvaluationError("数据集中没有可评测的样本")
    return samples


def select_dataset_samples(
    samples: list[dict],
    question_ids: list[str] | None = None,
) -> list[dict]:
    """Restrict a loaded dataset to explicitly requested question IDs."""
    requested = {str(question_id).strip() for question_id in question_ids or []}
    requested.discard("")
    if not requested:
        return samples

    selected = [
        sample for sample in samples
        if str(sample.get("question_id") or "") in requested
    ]
    found = {str(sample.get("question_id")) for sample in selected}
    missing = sorted(requested - found)
    if missing:
        raise EvaluationError(
            "指定的 question_id 不存在于测试集: " + ", ".join(missing)
        )
    return selected


def load_dataset_from_testset_tool(
    base_url: str,
    timeout: float,
    include_non_approved: bool = False,
    *,
    question_ids: list[str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[dict], dict]:
    base_url = base_url.rstrip("/")
    selected_ids = [
        str(question_id).strip()
        for question_id in question_ids or []
        if str(question_id).strip()
    ]
    selected_ids = list(dict.fromkeys(selected_ids))
    export_scope = (
        "selected"
        if selected_ids
        else "all" if include_non_approved else "approved"
    )
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            payload = _request_json(
                client,
                "POST",
                f"{base_url}/api/datasets/export",
                json={
                    "format": "jsonl",
                    "scope": export_scope,
                    **({"questionIds": selected_ids} if selected_ids else {}),
                },
            )
    except httpx.HTTPError as exc:
        raise EvaluationError(f"Cannot connect to test-set tool {base_url}: {exc}") from exc

    data = payload.get("data")
    if not isinstance(data, dict):
        raise EvaluationError("Test-set tool export response is missing data")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise EvaluationError("Test-set tool export contains no evaluable samples")
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict) or not sample.get("question_id") or not sample.get("question"):
            raise EvaluationError(
                f"Test-set tool export sample {index} is missing question_id or question"
            )
    samples = select_dataset_samples(samples, selected_ids)
    metadata = data.get("metadata")
    return samples, metadata if isinstance(metadata, dict) else {}


def find_samples_outside_knowledge_base(
    samples: list[dict],
    document_ids: set[str],
) -> list[dict]:
    mismatched = []
    for sample in samples:
        expected_ids = {
            str(document_id)
            for document_id in sample.get("source_document_ids") or []
            if document_id
        }
        if expected_ids and not expected_ids.intersection(document_ids):
            mismatched.append(sample)
    return mismatched


def _request_json(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    response = client.request(method, url, **kwargs)
    if response.is_error:
        try:
            error_payload = response.json()
            detail = error_payload.get("detail")
            if not detail and isinstance(error_payload.get("error"), dict):
                detail = error_payload["error"].get("message")
        except (ValueError, AttributeError):
            detail = response.text.strip()
        raise EvaluationError(
            f"{method} {url} 返回 {response.status_code}: {detail or '无错误详情'}"
        )
    if not response.content:
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        raise EvaluationError(f"{method} {url} 未返回 JSON") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{method} {url} 返回的数据不是 JSON 对象")
    return payload


def evaluate_sample(
    client: httpx.Client,
    base_url: str,
    knowledge_base_id: str,
    sample: dict,
    model: str | None,
    cleanup: bool,
) -> dict:
    kb_path = quote(knowledge_base_id, safe="")
    conversation_id = None
    try:
        conversation = _request_json(
            client,
            "POST",
            f"{base_url}/api/v1/knowledge-bases/{kb_path}/conversations",
            json={"title": f"评测 {sample['question_id']}"},
        )
        conversation_id = conversation.get("id")
        if not conversation_id:
            raise EvaluationError("创建评测对话后未获得 conversation id")
        started_at = perf_counter()
        response = _request_json(
            client,
            "POST",
            f"{base_url}/api/v1/knowledge-bases/{kb_path}/chat",
            json={
                "conversation_id": conversation_id,
                "question": sample["question"],
                "model": model,
            },
        )
        response["client_response_time_ms"] = round((perf_counter() - started_at) * 1_000, 2)
        return measure_retrieval_hits(sample, response)
    finally:
        if cleanup and conversation_id:
            try:
                client.delete(
                    f"{base_url}/api/v1/conversations/{quote(conversation_id, safe='')}"
                )
            except httpx.HTTPError:
                pass


def run_evaluation(
    dataset_path: Path | None,
    knowledge_base_id: str,
    base_url: str,
    model: str | None,
    timeout: float,
    cleanup: bool,
    include_non_approved: bool,
    testset_tool_url: str | None = None,
    question_ids: list[str] | None = None,
) -> dict:
    dataset_metadata = {}
    if testset_tool_url:
        samples, dataset_metadata = load_dataset_from_testset_tool(
            testset_tool_url,
            timeout,
            include_non_approved,
            question_ids=question_ids,
        )
        dataset_source = f"{testset_tool_url.rstrip('/')}/api/datasets/export"
    else:
        if dataset_path is None:
            raise EvaluationError("A local dataset path is required")
        samples = load_dataset(
            dataset_path,
            include_non_approved or bool(question_ids),
        )
        samples = select_dataset_samples(samples, question_ids)
        dataset_source = str(dataset_path.resolve())
    base_url = base_url.rstrip("/")
    results = []
    stop_reason = None
    with httpx.Client(timeout=timeout) as client:
        try:
            health = client.get(f"{base_url}/health")
        except httpx.HTTPError as exc:
            raise EvaluationError(f"无法连接后端 {base_url}: {exc}") from exc
        if health.status_code != 200:
            raise EvaluationError(
                f"后端未就绪，/health 返回 {health.status_code}: {health.text.strip()}"
            )
        if testset_tool_url:
            kb_path = quote(knowledge_base_id, safe="")
            document_response = client.get(
                f"{base_url}/api/v1/knowledge-bases/{kb_path}/documents"
            )
            if document_response.is_error:
                raise EvaluationError(
                    f"读取知识库文档失败（{document_response.status_code}）: {document_response.text.strip()}"
                )
            try:
                documents = document_response.json()
            except ValueError as exc:
                raise EvaluationError("知识库文档接口未返回 JSON") from exc
            if not isinstance(documents, list):
                raise EvaluationError("知识库文档接口返回的数据不是列表")
            mismatched = find_samples_outside_knowledge_base(
                samples,
                {
                    str(document.get("id"))
                    for document in documents
                    if isinstance(document, dict) and document.get("id")
                },
            )
            if mismatched:
                first = mismatched[0]
                raise EvaluationError(
                    "测试集工具中有 "
                    f"{len(mismatched)} 条样本仍引用当前知识库不存在的文档 ID；"
                    f"首条为 {first.get('question_id')}: {first.get('source_document_ids')}. "
                    "请将旧问题改为非 Approved，并基于同步后的 UUID Chunk 重新生成问题。"
                )

        for index, sample in enumerate(samples, start=1):
            question_id = str(sample["question_id"])
            print(f"[{index}/{len(samples)}] {question_id} {sample['question']}", flush=True)
            try:
                result = evaluate_sample(
                    client,
                    base_url,
                    knowledge_base_id,
                    sample,
                    model,
                    cleanup,
                )
                print(
                    f"  doc_hit={result['document_hit']} "
                    f"chunk_hit={result['chunk_hit']} "
                    f"chunk_f1={result.get('chunk_f1')} "
                    f"answer_f1={result.get('answer_f1')} "
                    f"latency_ms={result.get('response_time_ms')}",
                    flush=True,
                )
            except httpx.TimeoutException:
                remaining_count = len(samples) - index
                error = f"单题请求超时（{timeout:g} 秒）"
                result = {
                    "question_id": question_id,
                    "question": sample.get("question"),
                    "error": error,
                }
                stop_reason = (
                    f"{question_id} {error}，已停止剩余 {remaining_count} 道题"
                )
                print(f"  ERROR {stop_reason}", file=sys.stderr, flush=True)
            except (EvaluationError, httpx.HTTPError) as exc:
                result = {
                    "question_id": question_id,
                    "question": sample.get("question"),
                    "error": str(exc),
                }
                print(f"  ERROR {exc}", file=sys.stderr, flush=True)
            results.append(result)
            if stop_reason:
                break

    summary = summarize(results)
    summary.update({
        "requested_count": len(samples),
        "remaining_count": len(samples) - len(results),
        "stopped_early": stop_reason is not None,
    })

    return {
        "dataset": dataset_source,
        "dataset_metadata": dataset_metadata,
        "question_ids": question_ids or [],
        "knowledge_base_id": knowledge_base_id,
        "base_url": base_url,
        "model": model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "stop_reason": stop_reason,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 RAG JSONL 端到端评测")
    parser.add_argument("--knowledge-base-id", required=True, help="已导入评测文档的知识库 ID")
    dataset_source = parser.add_mutually_exclusive_group()
    dataset_source.add_argument("--dataset", type=Path, default=None, help="评测集 JSONL 路径")
    dataset_source.add_argument(
        "--testset-tool-url",
        default=None,
        help="直接从测试集工具导出当前数据集，例如 http://localhost:3000",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="RAG 后端地址")
    parser.add_argument("--model", default=None, help="可选的聊天模型名称")
    parser.add_argument("--timeout", type=float, default=180.0, help="单次 HTTP 请求超时秒数")
    parser.add_argument("--output", type=Path, default=Path("rag_eval_report.json"), help="评测报告路径")
    parser.add_argument("--keep-conversations", action="store_true", help="保留每题创建的评测对话")
    parser.add_argument("--include-non-approved", action="store_true", help="同时评测未 approved 的样本")
    parser.add_argument(
        "--question-id",
        dest="question_ids",
        action="append",
        metavar="ID",
        help="只评测指定题目 ID；需要多个题目时重复使用此参数",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = args.dataset or (None if args.testset_tool_url else Path(DEFAULT_DATASET))
    if dataset_path is not None and not dataset_path.is_file():
        raise SystemExit(f"评测集不存在: {dataset_path}")

    try:
        report = run_evaluation(
            dataset_path,
            args.knowledge_base_id,
            args.base_url,
            args.model,
            args.timeout,
            not args.keep_conversations,
            args.include_non_approved,
            args.testset_tool_url,
            args.question_ids,
        )
    except (EvaluationError, ValueError) as exc:
        print(f"评测无法启动: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print("\n评测汇总")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {args.output.resolve()}")

    if summary["error_count"]:
        return 2
    return 0
