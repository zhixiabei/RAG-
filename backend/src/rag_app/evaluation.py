from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import quote

import httpx

from .config import Settings
from .model_gateway_factory import ModelGateway, build_model_gateway


DEFAULT_DATASET = "heishanliang_rag_eval_v1.0.0.jsonl"
REFUSAL_PATTERNS = (
    "知识库中无相关内容",
    "知识库中没有相关内容",
    "未找到相关内容",
    "没有找到相关内容",
    "无法在知识库中找到",
    "知识库不包含相关",
    "无法根据知识库回答",
    "无法依据知识库回答",
    "没有任何关于",
    "没有找到任何关于",
    "无法基于这些资料",
)


class EvaluationError(RuntimeError):
    pass


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def character_f1(expected: str, actual: str) -> float:
    expected_chars = Counter(normalize_text(expected))
    actual_chars = Counter(normalize_text(actual))
    if not expected_chars or not actual_chars:
        return 0.0
    overlap = sum((expected_chars & actual_chars).values())
    precision = overlap / sum(actual_chars.values())
    recall = overlap / sum(expected_chars.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def keyword_recall(keywords: list[str], answer: str) -> float | None:
    normalized_keywords = [normalize_text(keyword) for keyword in keywords]
    normalized_keywords = [keyword for keyword in normalized_keywords if keyword]
    if not normalized_keywords:
        return None
    normalized_answer = normalize_text(answer)
    matched = sum(keyword in normalized_answer for keyword in normalized_keywords)
    return matched / len(normalized_keywords)


def is_refusal(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(pattern) in normalized for pattern in REFUSAL_PATTERNS)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise EvaluationError("答案向量维度不一致或为空")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise EvaluationError("答案向量范数为 0")
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def semantic_answer_score(
    models: ModelGateway,
    expected_answer: str,
    answer: str,
) -> float:
    if not expected_answer.strip():
        raise EvaluationError("非拒答样本缺少标准答案")
    if not answer.strip():
        return 0.0
    try:
        embeddings = models.embed([expected_answer, answer])
    except (RuntimeError, httpx.HTTPError) as exc:
        raise EvaluationError(f"答案向量评分失败: {exc}") from exc
    if len(embeddings) != 2:
        raise EvaluationError("Embedding 服务未返回两条答案向量")
    return round(max(0.0, cosine_similarity(embeddings[0], embeddings[1])) * 100, 2)


def score_response(
    sample: dict,
    response: dict,
    answer_score: float | None,
    min_answer_score: float,
    check_source_ids: bool = True,
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
    answer = str(response.get("answer") or "")
    expected_refusal = bool(sample.get("should_refuse"))
    refusal_detected = is_refusal(answer)

    document_hit = (
        bool(expected_document_ids & retrieved_document_ids)
        if check_source_ids and expected_document_ids
        else None
    )
    chunk_hit = (
        bool(expected_chunk_ids & retrieved_chunk_ids)
        if check_source_ids and expected_chunk_ids
        else None
    )
    answer_f1 = character_f1(str(sample.get("expected_answer") or ""), answer)
    keywords = [str(keyword) for keyword in sample.get("keywords") or []]
    answer_keyword_recall = keyword_recall(keywords, answer)
    refusal_correct = expected_refusal == refusal_detected
    answer_score_passed = None

    if not expected_refusal:
        if answer_score is None:
            raise EvaluationError("非拒答样本缺少答案向量分数")
        answer_score_passed = answer_score >= min_answer_score

    if expected_refusal:
        passed = refusal_correct
    else:
        passed = (
            refusal_correct
            and (not check_source_ids or document_hit is not False)
            and (not check_source_ids or chunk_hit is not False)
            and bool(answer_score_passed)
        )

    return {
        "question_id": sample.get("question_id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "difficulty": sample.get("difficulty"),
        "passed": passed,
        "expected_refusal": expected_refusal,
        "refusal_detected": refusal_detected,
        "refusal_correct": refusal_correct,
        "document_hit": document_hit,
        "chunk_hit": chunk_hit,
        "source_id_check_skipped": not check_source_ids,
        "answer_score": answer_score,
        "answer_score_passed": answer_score_passed,
        "answer_char_f1": round(answer_f1, 4),
        "keyword_recall": (
            round(answer_keyword_recall, 4)
            if answer_keyword_recall is not None
            else None
        ),
        "retrieval_used": bool(response.get("retrieval_used")),
        "retrieved_count": int(response.get("retrieved_count") or 0),
        "retrieved_document_ids": sorted(retrieved_document_ids),
        "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
        "expected_document_ids": sorted(expected_document_ids),
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "answer": answer,
        "expected_answer": sample.get("expected_answer") or "",
    }


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(values: list[bool]) -> float | None:
    return _average([1.0 if value else 0.0 for value in values])


def summarize(results: list[dict]) -> dict:
    completed = [result for result in results if "error" not in result]
    return {
        "sample_count": len(results),
        "completed_count": len(completed),
        "error_count": len(results) - len(completed),
        "passed_count": sum(bool(result.get("passed")) for result in completed),
        "pass_rate": _rate([bool(result.get("passed")) for result in completed]),
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
        "refusal_accuracy": _rate([
            bool(result.get("refusal_correct")) for result in completed
        ]),
        "average_answer_score": _average([
            float(result["answer_score"])
            for result in completed
            if result.get("answer_score") is not None
        ]),
        "average_answer_char_f1": _average([
            float(result["answer_char_f1"])
            for result in completed
            if not result.get("expected_refusal")
        ]),
        "average_keyword_recall": _average([
            float(result["keyword_recall"])
            for result in completed
            if result.get("keyword_recall") is not None
        ]),
    }


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


def load_dataset_from_testset_tool(
    base_url: str,
    timeout: float,
    include_non_approved: bool = False,
    *,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[dict], dict]:
    base_url = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            payload = _request_json(
                client,
                "POST",
                f"{base_url}/api/datasets/export",
                json={
                    "format": "jsonl",
                    "scope": "all" if include_non_approved else "approved",
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
    scoring_models: ModelGateway,
    base_url: str,
    knowledge_base_id: str,
    sample: dict,
    model: str | None,
    min_answer_score: float,
    check_source_ids: bool,
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
        answer = str(response.get("answer") or "")
        answer_score = (
            None
            if sample.get("should_refuse")
            else semantic_answer_score(
                scoring_models,
                str(sample.get("expected_answer") or ""),
                answer,
            )
        )
        return score_response(
            sample,
            response,
            answer_score,
            min_answer_score,
            check_source_ids,
        )
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
    scoring_models: ModelGateway,
    timeout: float,
    min_answer_score: float,
    check_source_ids: bool,
    cleanup: bool,
    include_non_approved: bool,
    testset_tool_url: str | None = None,
) -> dict:
    dataset_metadata = {}
    if testset_tool_url:
        samples, dataset_metadata = load_dataset_from_testset_tool(
            testset_tool_url,
            timeout,
            include_non_approved,
        )
        dataset_source = f"{testset_tool_url.rstrip('/')}/api/datasets/export"
    else:
        if dataset_path is None:
            raise EvaluationError("A local dataset path is required")
        samples = load_dataset(dataset_path, include_non_approved)
        dataset_source = str(dataset_path.resolve())
    base_url = base_url.rstrip("/")
    results = []
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
                    scoring_models,
                    base_url,
                    knowledge_base_id,
                    sample,
                    model,
                    min_answer_score,
                    check_source_ids,
                    cleanup,
                )
                state = "PASS" if result["passed"] else "FAIL"
                answer_score = result.get("answer_score")
                answer_score_text = "N/A" if answer_score is None else f"{answer_score:.1f}/100"
                print(
                    f"  {state} doc_hit={result['document_hit']} "
                    f"chunk_hit={result['chunk_hit']} "
                    f"answer_score={answer_score_text}",
                    flush=True,
                )
            except (EvaluationError, httpx.HTTPError) as exc:
                result = {
                    "question_id": question_id,
                    "question": sample.get("question"),
                    "passed": False,
                    "error": str(exc),
                }
                print(f"  ERROR {exc}", file=sys.stderr, flush=True)
            results.append(result)

    return {
        "dataset": dataset_source,
        "dataset_metadata": dataset_metadata,
        "knowledge_base_id": knowledge_base_id,
        "base_url": base_url,
        "model": model,
        "answer_embedding_model": getattr(scoring_models, "embedding_model", None),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "min_answer_score": min_answer_score,
        "source_id_check_skipped": not check_source_ids,
        "summary": summarize(results),
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
    parser.add_argument(
        "--min-answer-score",
        type=float,
        default=75.0,
        help="标准答案与实际答案的最低向量相似度百分数，范围 0 到 100（默认 75）",
    )
    parser.add_argument(
        "--skip-source-id-check",
        action="store_true",
        help="忽略测试集中的 source_document_ids/source_chunk_ids，仅评估答案和拒答（快速测试）",
    )
    parser.add_argument("--output", type=Path, default=Path("rag_eval_report.json"), help="评测报告路径")
    parser.add_argument("--keep-conversations", action="store_true", help="保留每题创建的评测对话")
    parser.add_argument("--include-non-approved", action="store_true", help="同时评测未 approved 的样本")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="低于该通过率时返回退出码 1，适合 CI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.min_answer_score <= 100:
        raise SystemExit("--min-answer-score 必须在 0 到 100 之间")
    if args.min_pass_rate is not None and not 0 <= args.min_pass_rate <= 1:
        raise SystemExit("--min-pass-rate 必须在 0 到 1 之间")
    dataset_path = args.dataset or (None if args.testset_tool_url else Path(DEFAULT_DATASET))
    if dataset_path is not None and not dataset_path.is_file():
        raise SystemExit(f"评测集不存在: {dataset_path}")

    try:
        scoring_models = build_model_gateway(Settings())
        report = run_evaluation(
            dataset_path,
            args.knowledge_base_id,
            args.base_url,
            args.model,
            scoring_models,
            args.timeout,
            args.min_answer_score,
            not args.skip_source_id_check,
            not args.keep_conversations,
            args.include_non_approved,
            args.testset_tool_url,
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
    if args.min_pass_rate is not None and (summary["pass_rate"] or 0.0) < args.min_pass_rate:
        return 1
    return 0
