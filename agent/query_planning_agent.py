from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Sequence

from .context import select_history_messages
from .contracts import ModelGateway
from .telemetry import model_usage_stage, timed_stage


logger = logging.getLogger(__name__)
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["single", "rewrite", "decompose"]},
        "standalone_query": {"type": "string"},
        "subqueries": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["strategy", "standalone_query", "subqueries"],
}
_HISTORY_TOKENS = 1_200
_MAX_HISTORY_MESSAGES = 4
_MAX_HISTORY_CHARS = 800
_MAX_QUERY_CHARS = 500
_MAX_SUBQUERIES = 4
_QUERY_CONTAMINATION_MARKERS = (
    "trigger reason",
    "current question:",
    "standalone query:",
    "rewrite:",
    "触发原因：",
    "对话历史：",
    "当前问题：",
    "独立问题：",
    "改写：",
)
_REFERENCE_PATTERN = re.compile(r"(?:它|这个|那个|这些|那些|上述|上面|前面|刚才|前者|后者|其中|该方案|该文件|该井|该层|this|that|it|they|above)", re.IGNORECASE)
_SHORT_FOLLOW_UP_PATTERN = re.compile(r"(?:呢|怎么样|如何|多少|是什么|怎么办|有何影响|有什么区别)[？?]?$", re.IGNORECASE)
_STRONG_COMPLEX_PATTERN = re.compile(r"(?:综合|分别|对比|比较|异同|各自|逐项|多个方面|哪些方面)", re.IGNORECASE)
_WEAK_COMPLEX_PATTERN = re.compile(r"(?:以及|并且|同时|还要|还需|并说明|并分析|并给出|及其|又有哪些)", re.IGNORECASE)
_QUOTED_SOURCE_PATTERN = re.compile(r"\u300a[^\u300b]{2,160}\u300b")
_FILE_SOURCE_PATTERN = re.compile(
    r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff._()\uFF08\uFF09-]{1,100}"
    r"\.(?:docx?|pdf|xlsx?|pptx?|txt|csv|gdb|att|md)",
    re.IGNORECASE,
)
_MULTI_SOURCE_LANGUAGE_PATTERN = re.compile(
    r"(?:\u6839\u636e|\u7ed3\u5408|\u57fa\u4e8e|\u4f9d\u636e|\u53c2\u7167|\u5bf9\u7167|\u4ece).{0,180}"
    r"(?:\u548c|\u4e0e|\u53ca|\u4ee5\u53ca|\u3001).{0,180}"
    r"(?:\u8bbe\u8ba1|\u5b9e\u9645|\u62a5\u544a|\u6587\u4ef6|\u6587\u6863|\u65b9\u6848|\u76d1\u6d4b|\u9636\u6bb5|\u5360\u6bd4|\u5224\u65ad|\u8bf4\u660e|\u6570\u91cf)",
    re.IGNORECASE,
)
_MULTI_DOCUMENT_COUNT_PATTERN = re.compile(
    r"(?:\u4e24|\u4e8c|\u4e09|\u56db|\u4e94|\u516d|\u4e03|\u516b|\u4e5d|\u5341|\d+|\u591a|\u5404|\u82e5\u5e72)"
    r"\s*(?:\u4efd|\u4e2a|\u7bc7|\u5f20)?\s*"
    r"(?:\u62a5\u544a|\u6587\u6863|\u6587\u4ef6|\u8d44\u6599|\u8868|\u53f0\u8d26|\u8bb0\u5f55|\u6570\u636e)",
    re.IGNORECASE,
)
_MULTI_DOCUMENT_WORD_PATTERN = re.compile(
    r"(?:\u591a\u4efd|\u5404\u4efd|\u5404\u6587\u6863|\u5404\u62a5\u544a|\u4e0d\u540c\u62a5\u544a|"
    r"\u4e0d\u540c\u6587\u6863|\u591a\u4e2a\u6587\u4ef6|\u591a\u4efd\u8d44\u6599|\u591a\u5f20\u8868)",
    re.IGNORECASE,
)
_DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"(?:\u62a5\u544a|\u6587\u6863|\u6587\u4ef6|\u8d44\u6599|\u6570\u636e\u8868|\u7edf\u8ba1\u8868|\u53f0\u8d26|\u8bb0\u5f55)",
    re.IGNORECASE,
)
_SOURCE_CAPTURE_PATTERN = re.compile(r"\u300a([^\u300b]{2,160})\u300b")

QUERY_PLANNING_PROMPT = """你负责生成知识库检索查询，不负责回答问题。
策略：single 表示问题独立且只有一个目标；rewrite 表示依赖历史指代；decompose 表示有多个取证目标，需要生成 2 到 4 个独立子问题。
必须保留井号、层位、年份、标准号、数值和专有名词。只能改写或拆分已有目标，不得补充答案、结论或新事实。
standalone_query 是消除指代后的完整原问题。只输出符合 schema 的 JSON。"""


@dataclass(frozen=True)
class QueryPlan:
    strategy: str
    standalone_query: str
    subqueries: tuple[str, ...] = ()
    trigger: str | None = None
    fallback: bool = False

    @classmethod
    def single(cls, question: str, *, trigger: str | None = None, fallback: bool = False) -> "QueryPlan":
        return cls("single", _clean_query(question), trigger=trigger, fallback=fallback)

    @property
    def model_invoked(self) -> bool:
        return self.trigger is not None

    def retrieval_queries(self, original_question: str) -> list[str]:
        queries = [_clean_query(original_question)]
        if self.strategy == "rewrite":
            queries.append(self.standalone_query)
        elif self.strategy == "decompose":
            queries.extend(self.subqueries)
        return _unique_queries(queries)

    def rerank_query(self, original_question: str) -> str:
        return self.standalone_query or _clean_query(original_question)

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "standalone_query": self.standalone_query, "subqueries": list(self.subqueries), "trigger": self.trigger, "model_invoked": self.model_invoked, "fallback": self.fallback}


class QueryPlanningAgent:
    """Conditionally rewrites follow-ups and decomposes multi-evidence queries."""

    name = "query_planning"

    def __init__(self, models: ModelGateway):
        self.models = models

    def run(self, question: str, history: Sequence[dict[str, Any]]) -> QueryPlan:
        trigger = query_planning_trigger(question, history)
        if trigger is None:
            return QueryPlan.single(question)
        try:
            with timed_stage("planning.generation"), model_usage_stage("query_planning"):
                output = self.models.complete(
                    query_planning_messages(question, history, trigger),
                    temperature=0, max_tokens=384, reasoning=False, response_schema=_PLAN_SCHEMA,
                )
            return parse_query_plan(output, question, trigger)
        except Exception as exc:
            logger.warning(
                "Query planning output invalid; using the original question (%s: %s)",
                type(exc).__name__,
                exc,
            )
            logger.debug("Query planning failure details", exc_info=True)
            return QueryPlan.single(question, trigger=trigger, fallback=True)


def query_planning_trigger(question: str, history: Sequence[dict[str, Any]]) -> str | None:
    normalized = _clean_query(question)
    compact_length = len(re.sub(r"\s+", "", normalized))
    has_history = any(item.get("role") in {"user", "assistant"} and str(item.get("content") or "").strip() for item in history)
    if has_history and (_REFERENCE_PATTERN.search(normalized) or compact_length <= 18 and _SHORT_FOLLOW_UP_PATTERN.search(normalized)):
        return "context_reference"
    if compact_length >= 18 and (
        _STRONG_COMPLEX_PATTERN.search(normalized)
        or _looks_like_multi_source_query(normalized)
    ):
        return "complex_query"
    if compact_length >= 30 and (
        len(_WEAK_COMPLEX_PATTERN.findall(normalized)) >= 2
        or _looks_like_multi_source_query(normalized)
        and _WEAK_COMPLEX_PATTERN.search(normalized)
    ):
        return "complex_query"
    return None


def _looks_like_multi_source_query(question: str) -> bool:
    source_references = [
        *_QUOTED_SOURCE_PATTERN.findall(question),
        *_FILE_SOURCE_PATTERN.findall(question),
    ]
    if len({reference.casefold() for reference in source_references}) >= 2:
        return True
    if (
        _MULTI_DOCUMENT_COUNT_PATTERN.search(question)
        or _MULTI_DOCUMENT_WORD_PATTERN.search(question)
    ):
        return True
    document_reference_count = len(_DOCUMENT_REFERENCE_PATTERN.findall(question))
    return bool(
        _MULTI_SOURCE_LANGUAGE_PATTERN.search(question)
        or document_reference_count >= 2
        and re.search(
            r"(?:\u6839\u636e|\u7ed3\u5408|\u5bf9\u6bd4|\u6bd4\u8f83|\u6838\u5bf9|"
            r"\u5206\u522b|\u5404|\u4e0d\u540c|\u548c|\u4e0e|\u53ca|\u3001)",
            question,
            re.IGNORECASE,
        )
    )


def query_planning_messages(question: str, history: Sequence[dict[str, Any]], trigger: str) -> list[dict[str, str]]:
    history_view = select_history_messages(
        history,
        _HISTORY_TOKENS if trigger == "context_reference" else 0,
    )
    selected_history = history_view.messages[-_MAX_HISTORY_MESSAGES:]
    transcript = "\n".join(
        f"{item['role']}: {_clip_history_text(item['content'])}"
        for item in selected_history
    )
    if history_view.omission_notice:
        transcript = f"{history_view.omission_notice}\n{transcript}"
    return [
        {"role": "system", "content": QUERY_PLANNING_PROMPT},
        {"role": "user", "content": f"触发原因：{trigger}\n对话历史：\n{transcript or '（无）'}\n\n当前问题：\n{question}"},
    ]


def parse_query_plan(output: str, question: str, trigger: str) -> QueryPlan:
    payload = _load_json_object(output)
    if not isinstance(payload, dict):
        raise ValueError("Query planner output must be a JSON object")
    strategy = str(payload.get("strategy") or "").strip().lower()
    if strategy not in {"single", "rewrite", "decompose"}:
        raise ValueError("Query planner returned an unsupported strategy")
    original = _clean_query(question)
    standalone_query = _validated_generated_query(
        payload.get("standalone_query") or original,
        fallback=original,
    )
    standalone_query = _preserve_explicit_sources(original, standalone_query)
    raw_subqueries = payload.get("subqueries") or []
    if not isinstance(raw_subqueries, list):
        raise ValueError("Query planner subqueries must be a list")
    subqueries = tuple(_unique_queries([
        _validated_generated_query(item)
        for item in raw_subqueries
        if str(item).strip()
    ]))[:_MAX_SUBQUERIES]
    subqueries = _ensure_subquery_source_anchors(original, subqueries)
    if strategy == "rewrite" and standalone_query == original:
        strategy = "single"
    if len(subqueries) >= 2:
        # Small models sometimes emit valid subqueries but forget to update
        # the strategy label. The payload is stronger evidence of intent.
        strategy = "decompose"
    if strategy == "decompose" and len(subqueries) < 2:
        strategy = "rewrite" if standalone_query != original else "single"
        subqueries = ()
    if trigger == "complex_query":
        # Complex questions are retrieval tasks, not answer-rewriting tasks.
        # A model can accidentally place guessed values in standalone_query,
        # so never use its free-form rewrite as the rerank query.
        deterministic_subqueries = deterministic_source_subqueries(original)
        if len(deterministic_subqueries) >= 2:
            return QueryPlan(
                "decompose",
                original,
                deterministic_subqueries,
                trigger=trigger,
            )
        if strategy == "decompose" and len(subqueries) >= 2:
            return QueryPlan(
                "decompose",
                original,
                subqueries,
                trigger=trigger,
            )
        return QueryPlan.single(original, trigger=trigger)
    if strategy != "decompose":
        subqueries = ()
    return QueryPlan(strategy, standalone_query, subqueries, trigger=trigger)


def fallback_query_plan(question: str, trigger: str | None) -> QueryPlan:
    """Return a retrieval-safe plan when structured model output is unusable."""
    original = _clean_query(question)
    if trigger == "complex_query":
        subqueries = (
            deterministic_source_subqueries(original)
            or deterministic_topic_subqueries(original)
        )
        if len(subqueries) >= 2:
            return QueryPlan(
                "decompose",
                _preserve_explicit_sources(original, original),
                tuple(subqueries[:_MAX_SUBQUERIES]),
                trigger=trigger,
                fallback=True,
            )
    return QueryPlan.single(question, trigger=trigger, fallback=True)


def deterministic_topic_subqueries(question: str) -> tuple[str, ...]:
    """Split a clearly multi-target question without inventing new facts."""
    normalized = _clean_query(question)
    clauses = [
        _clean_query(clause)
        for clause in re.split(r"[；;。！？?]+", normalized)
        if _clean_query(clause)
    ]
    if len(clauses) >= 2:
        return tuple(_unique_queries(clauses))[:_MAX_SUBQUERIES]

    separator = re.search(r"(和|与|及|以及|、)", normalized)
    if not separator:
        return ()
    left = normalized[:separator.start()].strip()
    right = normalized[separator.end():].strip()
    if len(left) < 4 or len(right) < 4:
        return ()

    # Move a shared predicate such as “的绝对吸水量” or “有哪些要求”
    # onto both targets. This keeps each query independently searchable.
    tail_match = re.search(
        r"(的(?:[^，。；！？?]{1,24})|有哪些[^，。；！？?]{0,20}|"
        r"是什么[^，。；！？?]{0,20}|是多少[^，。；！？?]{0,20}|"
        r"如何[^，。；！？?]{0,20})",
        right,
    )
    if tail_match:
        right_target = right[:tail_match.start()].strip()
        shared_tail = right[tail_match.start():].strip()
    else:
        right_target = right
        shared_tail = ""
    if len(right_target) < 2:
        return ()

    left_query = _clean_query(f"{left}{shared_tail}")
    right_query = _clean_query(f"{right_target}{shared_tail}")
    if not left_query or not right_query or left_query.casefold() == right_query.casefold():
        return ()
    return tuple(_unique_queries((left_query, right_query)))


def extract_explicit_source_names(question: str) -> tuple[str, ...]:
    """Extract quoted titles and explicit file names in first-seen order."""
    sources: list[str] = []
    seen: set[str] = set()
    for source in (
        *_SOURCE_CAPTURE_PATTERN.findall(question),
        *_FILE_SOURCE_PATTERN.findall(question),
    ):
        source = _clean_query(source)
        key = re.sub(r"\s+", "", source).casefold()
        if not source or key in seen:
            continue
        seen.add(key)
        sources.append(source)
        if len(sources) >= _MAX_SUBQUERIES:
            break
    return tuple(sources)


def _preserve_explicit_sources(original: str, generated: str) -> str:
    sources = extract_explicit_source_names(original)
    if not sources:
        return generated
    normalized_generated = re.sub(r"\s+", "", generated).casefold()
    missing = [
        source for source in sources
        if re.sub(r"\s+", "", source).casefold() not in normalized_generated
    ]
    if not missing:
        return generated
    return _append_source_anchors(generated, missing)


def _ensure_subquery_source_anchors(
    original: str,
    subqueries: Sequence[str],
) -> tuple[str, ...]:
    sources = extract_explicit_source_names(original)
    if len(sources) < 2 or not subqueries:
        return tuple(subqueries)
    anchored: list[str] = []
    one_to_one = len(subqueries) == len(sources)
    for index, subquery in enumerate(subqueries):
        normalized = re.sub(r"\s+", "", subquery).casefold()
        if any(re.sub(r"\s+", "", source).casefold() in normalized for source in sources):
            anchored.append(subquery)
            continue
        selected = (sources[index],) if one_to_one else tuple(sources)
        anchored.append(_append_source_anchors(subquery, selected))
    return tuple(_unique_queries(anchored))[:_MAX_SUBQUERIES]


def _append_source_anchors(query: str, sources: Sequence[str]) -> str:
    """Append source names while reserving space so the anchors survive clipping."""
    anchors = "、".join(f"《{source}》" for source in sources)
    suffix = f"；来源：{anchors}"
    if len(suffix) >= _MAX_QUERY_CHARS:
        return _clean_query(suffix)
    prefix_limit = _MAX_QUERY_CHARS - len(suffix)
    prefix = _clean_query(query)[:prefix_limit].rstrip(" ，,；;")
    return f"{prefix}{suffix}"


def deterministic_source_subqueries(question: str) -> tuple[str, ...]:
    """Build source-anchored queries when a planner misses explicit file targets."""
    sources = list(extract_explicit_source_names(question))
    if len(sources) < 2:
        return ()
    base_query = question
    for source in sources:
        base_query = base_query.replace(f"《{source}》", " ")
        base_query = base_query.replace(source, " ")
    base_query = re.sub(
        r"(?:根据|结合|基于|依据)\s*(?:(?:和|与|及|以及|、)\s*)+",
        " ",
        base_query,
    )
    base_query = _clean_query(re.sub(r"[，,、；;]+", " ", base_query))
    clauses = [
        _clean_query(clause)
        for clause in re.split(r"[，,；;。！？?]+", base_query)
        if _clean_query(clause)
    ]
    subqueries = []
    for source in sources:
        if re.search(r"(?:设计|方案)", source):
            topic_pattern = re.compile(r"(?:设计|阶段|对应|采油|井数)")
        else:
            topic_pattern = re.compile(r"(?:实际|监测|示踪剂|见剂|占比|连通)")
        selected = [clause for clause in clauses if topic_pattern.search(clause)]
        target = " ".join(selected) or base_query
        subqueries.append(_clean_query(f"《{source}》；{target}"))
    return tuple(subqueries)


def _load_json_object(output: str) -> dict[str, Any]:
    """Parse strict JSON while tolerating a model's prose or code-fence wrapper."""
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Query planner returned empty output")
    normalized = output.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[^\r\n]*\r?\n?", "", normalized, count=1)
        normalized = re.sub(r"\r?\n?```\s*$", "", normalized).strip()

    candidates = [normalized]
    extracted = _extract_json_object(normalized)
    if extracted and extracted != normalized:
        candidates.append(extracted)
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise ValueError("Query planner output must be a JSON object")
        return payload
    if last_error is not None:
        raise last_error
    raise ValueError("Query planner output must be a JSON object")


def _extract_json_object(value: str) -> str | None:
    """Return the first balanced JSON object, ignoring braces inside strings."""
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == chr(92):
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return value[start:index + 1]
    return None


def _validated_generated_query(value: object, fallback: str = "") -> str:
    query = _clean_query(str(value or ""))
    lowered = query.casefold()
    if any(marker in lowered or marker in query for marker in _QUERY_CONTAMINATION_MARKERS):
        raise ValueError("Query planner copied prompt text into a query")
    if not query:
        if fallback:
            return fallback
        raise ValueError("Query planner returned an empty query")
    return query


def _clip_history_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= _MAX_HISTORY_CHARS:
        return text
    half = (_MAX_HISTORY_CHARS - 18) // 2
    return f"{text[:half]} ...[省略]... {text[-half:]}"


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:_MAX_QUERY_CHARS]


def _unique_queries(queries: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for query in queries:
        cleaned = _clean_query(query)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
