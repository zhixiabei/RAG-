from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
import unicodedata
from urllib.parse import quote

import httpx

from agent import AnswerJudgeAgent
from agent.telemetry import collect_model_usage

DEFAULT_DATASET = "heishanliang_rag_eval_v1.0.0.jsonl"
DEFAULT_EVIDENCE_SIMILARITY_THRESHOLD = 0.72
EVIDENCE_WINDOW_MAX_CHARS = 4_000


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
        gold_answer_items = (
            [str(item).strip() for item in normalized.get("gold_answer") or [] if str(item).strip()]
            if isinstance(normalized.get("gold_answer"), (list, tuple))
            else []
        )
        evidence_items = []
        evidence_texts = []
        evidence_facts = []
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            text_span = str(item.get("text_span") or "").strip()
            if not text_span:
                continue
            fact = str(item.get("fact") or item.get("answer_fact") or "").strip()
            if not fact and len(gold_answer_items) == len(evidence):
                fact = gold_answer_items[index]
            threshold = item.get("similarity_threshold")
            if threshold is not None:
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    threshold = None
            evidence_items.append({
                "document": str(item.get("document") or "").strip() or None,
                "page": item.get("page"),
                "section": str(item.get("section") or "").strip() or None,
                "text_span": text_span,
                "fact": fact or None,
                "similarity_threshold": threshold,
            })
            evidence_texts.append(text_span)
            if fact:
                evidence_facts.append(fact)
        normalized["evidence"] = evidence_items
        normalized["evidence_texts"] = evidence_texts
        normalized["evidence_facts"] = evidence_facts
    return normalized

def _validate_evidence_schema(sample: dict, label: str) -> None:
    evidence = [
        item for item in sample.get("evidence") or []
        if isinstance(item, dict) and str(item.get("text_span") or "").strip()
    ]
    if not evidence:
        return
    missing_facts = [
        str(index)
        for index, item in enumerate(evidence, start=1)
        if not str(item.get("fact") or "").strip()
    ]
    if missing_facts:
        raise EvaluationError(
            f"{label}: evidence items {', '.join(missing_facts)} are missing fact; "
            "text_span is provenance only and cannot be used as the scoring text"
        )
    for index, item in enumerate(evidence, start=1):
        threshold = item.get("similarity_threshold")
        if threshold is not None and not 0 <= float(threshold) <= 1:
            raise EvaluationError(
                f"{label}: evidence item {index} similarity_threshold must be between 0 and 1"
            )


def _ranked_retrieved_chunks(response: dict, citations: list[dict]) -> list[dict]:
    chunks = response.get("retrieved_chunks")
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return [
        {
            "chunk_id": citation.get("chunk_id"),
            "document_id": citation.get("document_id"),
            "title": citation.get("title"),
            "section_path": citation.get("section_path"),
            "chunk_index": citation.get("chunk_index"),
            "text": citation.get("excerpt") or "",
        }
        for citation in citations
        if isinstance(citation, dict)
    ]


_CHUNK_INDEX_PATTERN = re.compile(r":(\d+)$")


def _chunk_index(chunk_id: object) -> int | None:
    match = _CHUNK_INDEX_PATTERN.search(str(chunk_id or ""))
    return int(match.group(1)) if match else None


def _build_evidence_windows(ranked_chunks: list[dict], retrieval_k: int) -> list[dict]:
    chunks = []
    for rank, chunk in enumerate(ranked_chunks[:retrieval_k], start=1):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        document_name = (
            chunk.get("title")
            or chunk.get("file_name")
            or chunk.get("relative_path")
        )
        chunks.append({
            "rank": rank,
            "chunk_id": str(chunk.get("chunk_id") or rank),
            "chunk_index": (
                int(chunk["chunk_index"])
                if chunk.get("chunk_index") is not None
                else _chunk_index(chunk.get("chunk_id"))
            ),
            "document_group": str(chunk.get("document_id") or _document_key(document_name)),
            "document_key": _document_key(document_name),
            "text": text,
        })

    windows = []
    seen = set()

    def add_window(parts: list[dict]) -> None:
        chunk_ids = tuple(part["chunk_id"] for part in parts)
        if chunk_ids in seen:
            return
        seen.add(chunk_ids)
        windows.append({
            "rank": max(part["rank"] for part in parts),
            "retrieval_ranks": sorted(part["rank"] for part in parts),
            "chunk_ids": list(chunk_ids),
            "document_key": parts[0]["document_key"],
            "text": "\n".join(part["text"] for part in parts)[:EVIDENCE_WINDOW_MAX_CHARS],
        })

    for chunk in chunks:
        add_window([chunk])

    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk["document_group"], []).append(chunk)
    for group in grouped.values():
        indexed = [chunk for chunk in group if chunk["chunk_index"] is not None]
        indexed.sort(key=lambda chunk: chunk["chunk_index"])
        for size in (2, 3):
            for start in range(len(indexed) - size + 1):
                parts = indexed[start:start + size]
                indices = [part["chunk_index"] for part in parts]
                if indices == list(range(indices[0], indices[0] + size)):
                    add_window(parts)

    windows.sort(key=lambda window: (window["rank"], len(window["chunk_ids"])))
    return windows


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return round(max(0.0, min(1.0, similarity)), 4)


def _evidence_threshold(item: dict, default_threshold: float) -> float:
    value = item.get("similarity_threshold")
    try:
        threshold = float(value) if value is not None else default_threshold
    except (TypeError, ValueError):
        threshold = default_threshold
    return max(0.0, min(1.0, threshold))


def _measure_evidence_retrieval(
    evidence: list[dict],
    ranked_chunks: list[dict],
    retrieval_k: int,
    embedding_models,
    default_threshold: float,
) -> dict:
    if embedding_models is None:
        raise EvaluationError("Evidence Embedding evaluator is not configured")
    facts = [str(item.get("fact") or "").strip() for item in evidence]
    if any(not fact for fact in facts):
        raise EvaluationError(
            "Evidence evaluation requires a fact for every evidence item; "
            "text_span is provenance only"
        )

    windows = _build_evidence_windows(ranked_chunks, retrieval_k)
    texts = [*facts, *[window["text"] for window in windows]]
    vectors = embedding_models.embed(texts)
    if len(vectors) != len(texts):
        raise EvaluationError(
            f"Embedding service returned {len(vectors)} vectors for {len(texts)} texts"
        )
    fact_vectors = vectors[:len(facts)]
    window_vectors = vectors[len(facts):]

    details = []
    for index, (item, fact_vector) in enumerate(zip(evidence, fact_vectors), start=1):
        expected_document = _document_key(item.get("document"))
        candidates = []
        for window, window_vector in zip(windows, window_vectors):
            if expected_document and window["document_key"] != expected_document:
                continue
            candidates.append({
                **window,
                "similarity": _cosine_similarity(fact_vector, window_vector),
            })

        threshold = _evidence_threshold(item, default_threshold)
        best = max(
            candidates,
            key=lambda candidate: (candidate["similarity"], -candidate["rank"]),
            default=None,
        )
        matched = [
            candidate for candidate in candidates
            if candidate["similarity"] >= threshold
        ]
        first = min(
            matched,
            key=lambda candidate: (candidate["rank"], -candidate["similarity"]),
            default=None,
        )
        best_similarity = best["similarity"] if best else 0.0
        details.append({
            "index": index,
            "fact": facts[index - 1],
            "document": item.get("document"),
            "page": item.get("page"),
            "section": item.get("section"),
            "similarity_threshold": threshold,
            "best_similarity": best_similarity,
            "coverage_at_k": best_similarity,
            "best_match_rank": best["rank"] if best else None,
            "best_match_chunk_ids": best["chunk_ids"] if best else [],
            "first_cover_rank": first["rank"] if first else None,
            "supporting_chunk_ids": first["chunk_ids"] if first else [],
            "supporting_retrieval_ranks": first["retrieval_ranks"] if first else [],
        })

    covered_count = sum(detail["first_cover_rank"] is not None for detail in details)
    first_relevant_rank = min(
        (detail["first_cover_rank"] for detail in details if detail["first_cover_rank"] is not None),
        default=None,
    )
    reciprocal_ranks = [
        1 / detail["first_cover_rank"] if detail["first_cover_rank"] else 0.0
        for detail in details
    ]
    return {
        "evidence_hit": covered_count > 0 if evidence else None,
        "evidence_recall_at_k": round(covered_count / len(evidence), 4) if evidence else None,
        "evidence_coverage_at_k": _average([
            detail["best_similarity"] for detail in details
        ]),
        "evidence_mrr": round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        "evidence_macro_mrr": _average(reciprocal_ranks),
        "evidence_count": len(evidence),
        "evidence_covered_count": covered_count,
        "evidence_window_count": len(windows),
        "evidence_details": details,
    }


def measure_retrieval_hits(
    sample: dict,
    response: dict,
    evidence_embedding_models=None,
    evidence_similarity_threshold: float = DEFAULT_EVIDENCE_SIMILARITY_THRESHOLD,
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
        evidence_metrics = _measure_evidence_retrieval(
            evidence,
            ranked_chunks,
            retrieval_k,
            evidence_embedding_models,
            evidence_similarity_threshold,
        )
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
        "timing": response.get("timing") or {"available": False},
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


def _aggregate_timing(results: list[dict], bucket: str) -> dict:
    totals: dict[str, dict[str, float]] = {}
    total_ms = 0.0
    sample_count = 0
    for result in results:
        timing = result.get("timing")
        if not isinstance(timing, dict) or not timing.get("available"):
            continue
        request_ms = _optional_number(timing.get("total_ms"))
        if request_ms is None:
            continue
        sample_count += 1
        total_ms += request_ms
        stages = timing.get(bucket)
        if not isinstance(stages, dict):
            continue
        for stage, value in stages.items():
            if not isinstance(value, dict):
                continue
            elapsed_ms = _optional_number(value.get("total_ms"))
            if elapsed_ms is None:
                continue
            aggregate = totals.setdefault(stage, {"calls": 0.0, "total_ms": 0.0})
            aggregate["calls"] += float(value.get("calls") or 0)
            aggregate["total_ms"] += elapsed_ms

    stages = {}
    for stage, value in totals.items():
        elapsed_ms = round(value["total_ms"], 2)
        stages[stage] = {
            "calls": int(value["calls"]),
            "total_ms": elapsed_ms,
            "average_ms": round(elapsed_ms / sample_count, 2) if sample_count else None,
            "share_percent": round(elapsed_ms * 100 / total_ms, 2) if total_ms else 0.0,
        }
    return {
        "sample_count": sample_count,
        "total_ms": round(total_ms, 2),
        "stages": stages,
    }


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
    judge_latencies = [
        float(result["judge_time_ms"])
        for result in completed
        if _optional_number(result.get("judge_time_ms")) is not None
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
    top_level_timing = _aggregate_timing(completed, "top_level")
    nested_timing = _aggregate_timing(completed, "by_stage")
    return {
        "sample_count": len(results),
        "completed_count": len(completed),
        "error_count": len(results) - len(completed),
        "retrieval_metric_sample_count": sum(
            1 for result in completed if result.get("document_hit") is not None
        ),
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
        "average_judge_time_ms": (
            round(sum(judge_latencies) / len(judge_latencies), 2)
            if judge_latencies
            else None
        ),
        "p95_judge_time_ms": _percentile(judge_latencies, 0.95),
        "average_response_time_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_response_time_ms": _percentile(latencies, 0.50),
        "p95_response_time_ms": _percentile(latencies, 0.95),
        "p99_response_time_ms": _percentile(latencies, 0.99),
        "timing_sample_count": top_level_timing["sample_count"],
        "timing_total_ms": top_level_timing["total_ms"],
        "timing_top_level": top_level_timing["stages"],
        "timing_by_stage": nested_timing["stages"],
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
            _validate_evidence_schema(sample, f"dataset line {line_number}")
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
        _validate_evidence_schema(sample, f"Test-set tool export sample {index}")
        normalized_samples.append(sample)
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
    evidence_embedding_models=None,
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
                "force_retrieval": True,
            },
        )
        response["client_response_time_ms"] = round((perf_counter() - started_at) * 1_000, 2)
        result = measure_retrieval_hits(
            sample,
            response,
            evidence_embedding_models,
        )
        if judge_agent is not None:
            judge_started_at = perf_counter()
            with collect_model_usage() as judge_usage:
                try:
                    result["judge"] = judge_agent.run(
                        sample,
                        result["answer"],
                    ).as_dict()
                except Exception as exc:
                    result["judge_error"] = f"{type(exc).__name__}: {exc}"
            result["judge_time_ms"] = round((perf_counter() - judge_started_at) * 1_000, 2)
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


def _print_evaluation_result(completed_count: int, total_count: int, result: dict) -> None:
    question_id = str(result.get("question_id") or "")
    print(
        f"[{completed_count}/{total_count}] {question_id} {result.get('question') or ''}",
        flush=True,
    )
    if result.get("error"):
        print(f"  ERROR {result['error']}", file=sys.stderr, flush=True)
        return

    if result.get("retrieval_basis") == "evidence":
        retrieval_metrics = (
            f"evidence_hit={result.get('evidence_hit')} "
            f"evidence_recall_at_k={result.get('evidence_recall_at_k')} "
            f"evidence_mrr={result.get('evidence_mrr')} "
            f"evidence_coverage_at_k={result.get('evidence_coverage_at_k')} "
            f"evidence_facts={result.get('evidence_covered_count')}/{result.get('evidence_count')}"
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
        f"latency_ms={result.get('response_time_ms')} "
        f"judge_ms={result.get('judge_time_ms')}",
        flush=True,
    )
    if result.get("judge_error"):
        print(
            f"  JUDGE_ERROR {result['judge_error']}",
            file=sys.stderr,
            flush=True,
        )


def _print_summary_line(summary: dict) -> None:
    # The two rates are aggregate values, so use the document metric's sample
    # count when available rather than treating non-retrieval control samples
    # as retrieval failures.
    retrieval_metric_samples = summary.get("retrieval_metric_sample_count")
    if retrieval_metric_samples is None:
        retrieval_metric_samples = summary.get("completed_count", 0)

    def percent(value: object) -> str:
        if value is None:
            return "NA"
        return f"{float(value) * 100:.2f}%"

    def number(value: object, suffix: str = "") -> str:
        return "NA" if value is None else f"{value}{suffix}"

    print(
        "SUMMARY "
        f"samples={summary.get('sample_count', 0)} "
        f"retrieval_metric_samples={retrieval_metric_samples} "
        f"completed={summary.get('completed_count', 0)} "
        f"errors={summary.get('error_count', 0)} "
        f"doc_hit_rate={percent(summary.get('document_hit_rate'))} "
        f"chunk_hit_rate={percent(summary.get('chunk_hit_rate'))} "
        f"recall={percent(summary.get('retrieval_recall_at_k'))} "
        f"mrr={percent(summary.get('mrr'))} "
        f"answer_score={percent(summary.get('average_answer_score'))} "
        f"avg_latency={number(summary.get('average_response_time_ms'), 'ms')} "
        f"p50={number(summary.get('p50_response_time_ms'), 'ms')} "
        f"p95={number(summary.get('p95_response_time_ms'), 'ms')}",
        flush=True,
    )


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
    evidence_embedding_models=None,
    max_concurrency: int = 1,
) -> dict:
    if timeout <= 0:
        raise EvaluationError("单题请求超时必须大于 0 秒")
    if max_concurrency <= 0:
        raise EvaluationError("评测并发数必须大于 0")

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
    evaluation_started_at = perf_counter()
    worker_count = max(1, min(max_concurrency, len(samples)))
    ordered_results: list[dict | None] = [None] * len(samples)

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

        active_concurrency = worker_count
        concurrency_reduced = False
        next_sample_index = 0

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="rag-evaluation",
        ) as executor:
            future_items = {}

            def submit_available_samples() -> None:
                nonlocal next_sample_index
                while (
                    next_sample_index < len(samples)
                    and len(future_items) < active_concurrency
                ):
                    index = next_sample_index
                    sample = samples[index]
                    future = executor.submit(
                        evaluate_sample,
                        client,
                        base_url,
                        knowledge_base_id,
                        sample,
                        model,
                        cleanup,
                        judge_agent,
                        evidence_embedding_models,
                    )
                    future_items[future] = (index, sample)
                    next_sample_index += 1

            submit_available_samples()
            completed_count = 0
            while future_items:
                completed_futures, _ = wait(
                    future_items,
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    completed_futures,
                    key=lambda item: future_items[item][0],
                ):
                    index, sample = future_items.pop(future)
                    question_id = str(sample["question_id"])
                    try:
                        result = future.result()
                    except httpx.TimeoutException:
                        result = {
                            "question_id": question_id,
                            "question": sample.get("question"),
                            "error": f"单题请求超时（{timeout:g} 秒）",
                        }
                        if active_concurrency > 1:
                            active_concurrency = 1
                            concurrency_reduced = True
                    except (EvaluationError, httpx.HTTPError) as exc:
                        result = {
                            "question_id": question_id,
                            "question": sample.get("question"),
                            "error": str(exc),
                        }
                    except Exception as exc:
                        result = {
                            "question_id": question_id,
                            "question": sample.get("question"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ordered_results[index] = result
                    completed_count += 1
                    _print_evaluation_result(completed_count, len(samples), result)

                submit_available_samples()

    results = [result for result in ordered_results if result is not None]
    evaluation_time_ms = round((perf_counter() - evaluation_started_at) * 1_000, 2)
    summary = summarize(results)
    summary.update({
        "requested_count": len(samples),
        "remaining_count": len(samples) - len(results),
        "stopped_early": False,
        "concurrency": worker_count,
        "requested_concurrency": worker_count,
        "final_concurrency": active_concurrency,
        "concurrency_reduced": concurrency_reduced,
        "evaluation_time_ms": evaluation_time_ms,
        "throughput_per_minute": (
            round(len(results) * 60_000 / evaluation_time_ms, 2)
            if evaluation_time_ms > 0
            else None
        ),
    })
    _print_summary_line(summary)

    return {
        "dataset": dataset_source,
        "dataset_metadata": dataset_metadata,
        "question_ids": question_ids or [],
        "evidence_embedding_model": getattr(evidence_embedding_models, "embedding_model", None),
        "knowledge_base_id": knowledge_base_id,
        "base_url": base_url,
        "model": model,
        "judge_model": judge_agent.model_name if judge_agent is not None else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "stop_reason": None,
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
    parser.add_argument("--timeout", type=float, default=None, help="单题 HTTP 请求超时秒数")
    parser.add_argument("--concurrency", type=int, default=None, help="同时评测的题目数")
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
        from .config import Settings
        from .model_gateway_factory import build_judge_gateway, build_model_gateway

        settings = Settings()
        models = build_model_gateway(settings)
        if not args.no_judge:
            if args.judge_model:
                settings = settings.model_copy(update={
                    "rag_judge_enabled": True,
                    "rag_judge_model": args.judge_model,
                })
            judge_models = build_judge_gateway(settings, models)
            if judge_models is not None:
                judge_agent = AnswerJudgeAgent(
                    judge_models,
                    pass_threshold=settings.rag_judge_pass_threshold,
                    max_evidence_chars=settings.rag_judge_max_evidence_chars,
                    max_output_tokens=settings.rag_judge_max_output_tokens,
                    max_concurrency=settings.rag_judge_max_concurrency,
                )

        report = run_evaluation(
            dataset_path,
            args.knowledge_base_id,
            args.base_url,
            args.model,
            args.timeout or settings.evaluation_request_timeout_seconds,
            not args.keep_conversations,
            args.include_non_approved,
            args.testset_tool_url,
            args.question_ids,
            judge_agent,
            models,
            args.concurrency or settings.evaluation_max_concurrency,
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
