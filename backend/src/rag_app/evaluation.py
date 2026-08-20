from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import unicodedata
from urllib.parse import quote

import httpx

from agent import AnswerJudgeAgent
from agent.telemetry import collect_model_usage

DEFAULT_DATASET = "heishanliang_rag_eval_v1.0.0.jsonl"
EVIDENCE_COVERAGE_THRESHOLD = 0.8
EVIDENCE_SHINGLE_SIZE = 5


class EvaluationError(RuntimeError):
    pass


def precision_recall(expected: set[str], actual: set[str]) -> dict[str, float | None]:
    if not expected:
        return {"precision": None, "recall": None}
    true_positives = len(expected & actual)
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def reciprocal_rank(expected: set[str], ranked_actual: list[str]) -> float | None:
    if not expected:
        return None
    for rank, item_id in enumerate(ranked_actual, start=1):
        if item_id in expected:
            return round(1 / rank, 4)
    return 0.0


def _ordered_unique(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return list(dict.fromkeys(str(value) for value in values if value))


def _retrieval_k(response: dict, ranked_ids: list[str]) -> int:
    for value in (response.get("retrieval_k"), response.get("retrieved_count")):
        if value is None or isinstance(value, bool):
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return len(ranked_ids)


def _normalize_match_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def _text_shingles(value: object) -> set[str]:
    text = _normalize_match_text(value)
    if not text:
        return set()
    size = min(EVIDENCE_SHINGLE_SIZE, len(text))
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def _document_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].casefold()


def _gold_answer_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def normalize_dataset_sample(sample: dict) -> dict:
    """Map the evidence-first JSONL schema to the internal judge/evaluation contract."""
    normalized = dict(sample)
    if not normalized.get("question") and normalized.get("query"):
        normalized["question"] = str(normalized["query"]).strip()
    if "expected_answer" not in normalized and "gold_answer" in normalized:
        normalized["expected_answer"] = _gold_answer_text(normalized.get("gold_answer"))

    evidence = normalized.get("evidence")
    if isinstance(evidence, list):
        evidence_items = []
        evidence_texts = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            text_span = str(item.get("text_span") or "").strip()
            if not text_span:
                continue
            evidence_items.append({
                "document": str(item.get("document") or "").strip() or None,
                "page": item.get("page"),
                "section": str(item.get("section") or "").strip() or None,
                "text_span": text_span,
            })
            evidence_texts.append(text_span)
        normalized["evidence"] = evidence_items
        normalized["evidence_texts"] = evidence_texts
    return normalized


def _ranked_retrieved_chunks(response: dict, citations: list[dict]) -> list[dict]:
    chunks = response.get("retrieved_chunks")
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return [
        {
            "chunk_id": citation.get("chunk_id"),
            "document_id": citation.get("document_id"),
            "title": citation.get("title"),
            "text": citation.get("excerpt") or "",
        }
        for citation in citations
        if isinstance(citation, dict)
    ]


def _measure_evidence_retrieval(evidence: list[dict], ranked_chunks: list[dict], retrieval_k: int) -> dict:
    ranked_at_k = ranked_chunks[:retrieval_k]
    chunk_shingles = [_text_shingles(chunk.get("text")) for chunk in ranked_at_k]
    coverages = []
    reciprocal_ranks = []
    details = []

    for index, item in enumerate(evidence, start=1):
        expected_shingles = _text_shingles(item.get("text_span"))
        cumulative_shingles: set[str] = set()
        first_cover_rank = None
        coverage = 0.0
        for rank, retrieved_shingles in enumerate(chunk_shingles, start=1):
            cumulative_shingles.update(retrieved_shingles)
            coverage = (
                len(expected_shingles & cumulative_shingles) / len(expected_shingles)
                if expected_shingles
                else 0.0
            )
            if first_cover_rank is None and coverage >= EVIDENCE_COVERAGE_THRESHOLD:
                first_cover_rank = rank
        coverages.append(coverage)
        reciprocal_ranks.append(1 / first_cover_rank if first_cover_rank else 0.0)
        details.append({
            "index": index,
            "document": item.get("document"),
            "page": item.get("page"),
            "section": item.get("section"),
            "coverage_at_k": round(coverage, 4),
            "first_cover_rank": first_cover_rank,
        })

    covered_count = sum(detail["first_cover_rank"] is not None for detail in details)
    first_relevant_rank = min(
        (detail["first_cover_rank"] for detail in details if detail["first_cover_rank"] is not None),
        default=None,
    )
    return {
        "evidence_hit": covered_count > 0 if evidence else None,
        "evidence_recall_at_k": round(covered_count / len(evidence), 4) if evidence else None,
        "evidence_coverage_at_k": _average(coverages),
        "evidence_mrr": round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        "evidence_macro_mrr": _average(reciprocal_ranks),
        "evidence_count": len(evidence),
        "evidence_covered_count": covered_count,
        "evidence_details": details,
    }


def measure_retrieval_hits(
    sample: dict,
    response: dict,
) -> dict:
    citations = response.get("citations") or []
    ranked_chunks = _ranked_retrieved_chunks(response, citations)
    ranked_document_ids = _ordered_unique(response.get("retrieved_document_ids")) or _ordered_unique([
        citation.get("document_id") for citation in citations
    ])
    ranked_chunk_ids = _ordered_unique(response.get("retrieved_chunk_ids")) or _ordered_unique([
        chunk.get("chunk_id") for chunk in ranked_chunks
    ])
    retrieved_document_ids = set(ranked_document_ids)
    retrieved_chunk_ids = set(ranked_chunk_ids)
    expected_document_ids = set(sample.get("source_document_ids") or [])
    expected_chunk_ids = set(sample.get("source_chunk_ids") or [])
    evidence = [
        item for item in sample.get("evidence") or []
        if isinstance(item, dict) and str(item.get("text_span") or "").strip()
    ]

    expected_document_names = _ordered_unique([item.get("document") for item in evidence])
    ranked_document_names = _ordered_unique([
        chunk.get("title") or chunk.get("file_name") or chunk.get("relative_path")
        for chunk in ranked_chunks
    ])
    if evidence and expected_document_names:
        expected_document_keys = {_document_key(name) for name in expected_document_names if _document_key(name)}
        retrieved_document_keys = {_document_key(name) for name in ranked_document_names if _document_key(name)}
        document_hit = bool(expected_document_keys & retrieved_document_keys)
        document_metrics = precision_recall(expected_document_keys, retrieved_document_keys)
        document_reciprocal_rank = reciprocal_rank(
            expected_document_keys,
            [_document_key(name) for name in ranked_document_names if _document_key(name)],
        )
    else:
        document_hit = (
            bool(expected_document_ids & retrieved_document_ids)
            if expected_document_ids
            else None
        )
        document_metrics = precision_recall(expected_document_ids, retrieved_document_ids)
        document_reciprocal_rank = reciprocal_rank(expected_document_ids, ranked_document_ids)

    if evidence:
        retrieval_k = _retrieval_k(
            response,
            [str(chunk.get("chunk_id") or index) for index, chunk in enumerate(ranked_chunks, start=1)],
        )
        evidence_metrics = _measure_evidence_retrieval(evidence, ranked_chunks, retrieval_k)
        chunk_hit = None
        chunk_metrics = {"precision": None, "recall": None}
        chunk_reciprocal_rank = None
        retrieval_recall_at_k = evidence_metrics["evidence_recall_at_k"]
        retrieval_reciprocal_rank = evidence_metrics["evidence_mrr"]
        retrieval_basis = "evidence"
    else:
        evidence_metrics = {
            "evidence_hit": None,
            "evidence_recall_at_k": None,
            "evidence_coverage_at_k": None,
            "evidence_mrr": None,
            "evidence_count": 0,
            "evidence_covered_count": 0,
            "evidence_details": [],
        }
        chunk_hit = (
            bool(expected_chunk_ids & retrieved_chunk_ids)
            if expected_chunk_ids
            else None
        )
        chunk_metrics = precision_recall(expected_chunk_ids, retrieved_chunk_ids)
        chunk_reciprocal_rank = reciprocal_rank(expected_chunk_ids, ranked_chunk_ids)
        expected_retrieval_ids = expected_chunk_ids or expected_document_ids
        ranked_retrieval_ids = ranked_chunk_ids if expected_chunk_ids else ranked_document_ids
        retrieval_k = _retrieval_k(response, ranked_retrieval_ids)
        ranked_at_k = ranked_retrieval_ids[:retrieval_k]
        retrieval_recall_at_k = (
            round(len(expected_retrieval_ids & set(ranked_at_k)) / len(expected_retrieval_ids), 4)
            if expected_retrieval_ids
            else None
        )
        retrieval_reciprocal_rank = reciprocal_rank(expected_retrieval_ids, ranked_at_k)
        retrieval_basis = "chunk" if expected_chunk_ids else "document" if expected_document_ids else None
    generated_answer = str(response.get("answer") or "")

    return {
        "question_id": sample.get("question_id"),
        "question": sample.get("question"),
        "question_type": sample.get("question_type"),
        "difficulty": sample.get("difficulty"),
        "document_hit": document_hit,
        "chunk_hit": chunk_hit,
        **evidence_metrics,
        "retrieval_basis": retrieval_basis,
        "document_precision": document_metrics["precision"],
        "document_recall": document_metrics["recall"],
        "document_reciprocal_rank": document_reciprocal_rank,
        "chunk_precision": chunk_metrics["precision"],
        "chunk_recall": chunk_metrics["recall"],
        "chunk_reciprocal_rank": chunk_reciprocal_rank,
        "retrieval_k": retrieval_k,
        "retrieval_recall_at_k": retrieval_recall_at_k,
        "retrieval_reciprocal_rank": retrieval_reciprocal_rank,
        "mrr": retrieval_reciprocal_rank,
        "answer": generated_answer,
        "retrieval_used": bool(response.get("retrieval_used")),
        "retrieved_count": int(response.get("retrieved_count") or 0),
        "retrieved_document_ids": ranked_document_ids,
        "retrieved_chunk_ids": ranked_chunk_ids,
        "expected_document_ids": sorted(expected_document_ids),
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "retrieved_documents": ranked_document_names,
        "expected_documents": expected_document_names,
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
    judgments = [
        result["judge"]
        for result in completed
        if isinstance(result.get("judge"), dict)
    ]
    judge_errors = [
        result["judge_error"]
        for result in completed
        if result.get("judge_error")
    ]
    judge_reported_usage = [
        result.get("judge_token_usage") or {}
        for result in completed
        if (result.get("judge_token_usage") or {}).get("available")
    ]
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
        "evidence_sample_count": sum(
            1 for result in completed if int(result.get("evidence_count") or 0) > 0
        ),
        "evidence_hit_rate": _rate([
            result["evidence_hit"]
            for result in completed
            if result.get("evidence_hit") is not None
        ]),
        "evidence_recall_at_k": _average_metric(completed, "evidence_recall_at_k"),
        "evidence_coverage_at_k": _average_metric(completed, "evidence_coverage_at_k"),
        "evidence_mrr": _average_metric(completed, "evidence_mrr"),
        "document_precision": _average_metric(completed, "document_precision"),
        "document_recall": _average_metric(completed, "document_recall"),
        "document_mrr": _average_metric(completed, "document_reciprocal_rank"),
        "chunk_precision": _average_metric(completed, "chunk_precision"),
        "chunk_recall": _average_metric(completed, "chunk_recall"),
        "chunk_mrr": _average_metric(completed, "chunk_reciprocal_rank"),
        "retrieval_k": _summary_retrieval_k(completed),
        "retrieval_recall_at_k": _average_metric(completed, "retrieval_recall_at_k"),
        "mrr": _average_metric(completed, "retrieval_reciprocal_rank"),
        "judge_sample_count": len(judgments),
        "judge_error_count": len(judge_errors),
        "answer_pass_rate": _rate([
            judgment["passed"]
            for judgment in judgments
            if judgment.get("passed") is not None
        ]),
        "average_answer_score": _average_metric(judgments, "score"),
        "average_correctness_score": _average_metric(judgments, "correctness_score"),
        "average_completeness_score": _average_metric(judgments, "completeness_score"),
        "average_faithfulness_score": _average_metric(judgments, "faithfulness_score"),
        "judge_token_usage_sample_count": len(judge_reported_usage),
        "judge_input_tokens": sum(
            int(usage.get("input_tokens") or 0)
            for usage in judge_reported_usage
        ),
        "judge_output_tokens": sum(
            int(usage.get("output_tokens") or 0)
            for usage in judge_reported_usage
        ),
        "judge_total_tokens": sum(
            int(usage.get("total_tokens") or 0)
            for usage in judge_reported_usage
        ),
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


def _summary_retrieval_k(results: list[dict]) -> int | None:
    values = [int(result["retrieval_k"]) for result in results if result.get("retrieval_k") is not None]
    if not values or len(set(values)) != 1:
        return None
    return values[0]


def load_dataset(
    path: Path,
    include_non_approved: bool = False,
    *,
    allow_empty: bool = False,
) -> list[dict]:
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
            if not isinstance(sample, dict):
                raise EvaluationError(f"Dataset line {line_number} is not a JSON object")
            sample = normalize_dataset_sample(sample)
            if not sample.get("question_id") or not sample.get("question"):
                raise EvaluationError(
                    f"数据集第 {line_number} 行缺少 question_id 或 question"
                )
            if include_non_approved or sample.get("status") in {None, "approved"}:
                samples.append(sample)
    if not samples and not allow_empty:
        raise EvaluationError("dataset contains no evaluable samples")
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
    normalized_samples = []
    for index, sample in enumerate(samples, start=1):
        sample = normalize_dataset_sample(sample) if isinstance(sample, dict) else sample
        if not isinstance(sample, dict) or not sample.get("question_id") or not sample.get("question"):
            raise EvaluationError(
                f"Test-set tool export sample {index} is missing question_id or question"
            )
    samples = select_dataset_samples(normalized_samples, selected_ids)
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
    judge_agent: AnswerJudgeAgent | None = None,
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
                "include_retrieved_content": bool(sample.get("evidence")),
            },
        )
        response["client_response_time_ms"] = round((perf_counter() - started_at) * 1_000, 2)
        result = measure_retrieval_hits(sample, response)
        if judge_agent is not None:
            with collect_model_usage() as judge_usage:
                try:
                    result["judge"] = judge_agent.run(
                        sample,
                        result["answer"],
                    ).as_dict()
                except Exception as exc:
                    result["judge_error"] = f"{type(exc).__name__}: {exc}"
            result["judge_token_usage"] = judge_usage.summary()
        return result
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
    judge_agent: AnswerJudgeAgent | None = None,
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
                    judge_agent,
                )
                if result.get("retrieval_basis") == "evidence":
                    retrieval_metrics = (
                        f"evidence_hit={result.get('evidence_hit')} "
                        f"evidence_recall_at_k={result.get('evidence_recall_at_k')} "
                        f"evidence_mrr={result.get('evidence_mrr')} "
                        f"evidence_coverage_at_k={result.get('evidence_coverage_at_k')}"
                    )
                else:
                    retrieval_metrics = (
                        f"chunk_hit={result.get('chunk_hit')} "
                        f"retrieval_recall_at_k={result.get('retrieval_recall_at_k')} "
                        f"mrr={result.get('mrr')}"
                    )
                print(
                    f"  doc_hit={result['document_hit']} "
                    f"{retrieval_metrics} "
                    f"answer_score={(result.get('judge') or {}).get('score')} "
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
        "judge_model": judge_agent.model_name if judge_agent is not None else None,
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
    parser.add_argument(
        "--judge-model",
        default=None,
        help="可选的 Judge 模型名称；默认使用 RAG_JUDGE_MODEL 或回答模型",
    )
    parser.add_argument("--no-judge", action="store_true", help="关闭答案质量模型评分")
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
        judge_agent = None
        if not args.no_judge:
            from .config import Settings
            from .model_gateway_factory import build_judge_gateway, build_model_gateway

            settings = Settings()
            if args.judge_model:
                settings = settings.model_copy(update={
                    "rag_judge_enabled": True,
                    "rag_judge_model": args.judge_model,
                })
            models = build_model_gateway(settings)
            judge_models = build_judge_gateway(settings, models)
            if judge_models is not None:
                judge_agent = AnswerJudgeAgent(
                    judge_models,
                    pass_threshold=settings.rag_judge_pass_threshold,
                    max_evidence_chars=settings.rag_judge_max_evidence_chars,
                )

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
            judge_agent,
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
