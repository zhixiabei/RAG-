from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Sequence

from ..domain.models import ParsedChunk
from ..domain.ports import ModelGateway

logger = logging.getLogger(__name__)

_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
    },
    "required": ["summary", "topics"],
}
_MAX_SAMPLE_CHUNKS = 12
_MAX_CHARS_PER_CHUNK = 900
_MAX_SUMMARY_CHARS = 2000
_MAX_TOPIC_CHARS = 80
_MAX_SECTION_ROUTES = 64


@dataclass(frozen=True)
class DocumentProfile:
    summary: str
    topics: tuple[str, ...]

    def routing_text(self, file_name: str, relative_path: str) -> str:
        return (
            f"文件名: {file_name}\n"
            f"完整路径: {relative_path}\n"
            f"摘要: {self.summary}\n"
            f"主题: {'、'.join(self.topics)}"
        )

    def route_texts(self, file_name: str, relative_path: str, chunks: Sequence[ParsedChunk]) -> list[tuple[str, str]]:
        routes = [
            (
                "identity",
                f"文件名: {file_name}\n"
                f"完整路径: {relative_path}\n"
                f"文件类型: {Path(file_name).suffix.lower() or '无后缀'}",
            ),
            (
                "summary",
                f"文件名: {file_name}\n"
                f"完整路径: {relative_path}\n"
                f"文件摘要: {self.summary}",
            ),
            (
                "topics",
                f"文件名: {file_name}\n"
                f"完整路径: {relative_path}\n"
                f"文件主题: {'、'.join(self.topics)}",
            ),
        ]
        section_names = []
        seen = set()
        for chunk in chunks:
            for part in (chunk.section_path or "").split("/"):
                section = _clean_text(part, _MAX_TOPIC_CHARS)
                key = section.casefold()
                if section and key not in seen:
                    seen.add(key)
                    section_names.append(section)
                    if len(section_names) >= _MAX_SECTION_ROUTES:
                        break
            if len(section_names) >= _MAX_SECTION_ROUTES:
                break
        if section_names:
            routes.append((
                "sections",
                f"文件名: {file_name}\n"
                f"完整路径: {relative_path}\n"
                f"章节主题: {'、'.join(section_names)}",
            ))
        return routes


def build_document_profile(models: ModelGateway, file_name: str, relative_path: str, chunks: Sequence[ParsedChunk]) -> DocumentProfile:
    fallback = _fallback_profile(file_name, relative_path, chunks)
    sample = _sample_document(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "你负责为文档检索生成文件级画像。根据文件名、路径、章节和正文样本，"
                "写一段覆盖文档用途、对象、关键内容和适用范围的摘要，并提取能够区分"
                "相似文件的主题词。主题词必须优先包含对象、井号、报告类型、测试方法、"
                "年份和关键业务术语；不要把表格中的随机数字当作主题。不要编造信息，只输出 JSON。"
            ),
        },
        {
            "role": "user",
            "content": f"文件名：{file_name}\n完整路径：{relative_path}\n章节与正文样本：\n{sample}",
        },
    ]
    try:
        output = models.complete(
            messages,
            temperature=0,
            max_tokens=600,
            reasoning=False,
            response_schema=_PROFILE_SCHEMA,
        )
        profile = _parse_profile(output)
        if profile.summary:
            return DocumentProfile(
                summary=profile.summary,
                topics=_merge_topics(profile.topics, fallback.topics),
            )
    except Exception:
        logger.warning("Document profile generation failed for %s; using deterministic fallback", relative_path, exc_info=True)
    return fallback


def _sample_document(chunks: Sequence[ParsedChunk]) -> str:
    if not chunks:
        return ""
    sample_count = min(len(chunks), _MAX_SAMPLE_CHUNKS)
    indices = [0] if sample_count == 1 else sorted({
        round(index * (len(chunks) - 1) / (sample_count - 1))
        for index in range(sample_count)
    })
    return "\n\n".join(
        f"[章节] {chunks[index].section_path or '无'}\n[内容] {chunks[index].text[:_MAX_CHARS_PER_CHUNK]}"
        for index in indices
    )


def _fallback_profile(file_name: str, relative_path: str, chunks: Sequence[ParsedChunk]) -> DocumentProfile:
    topics = []
    seen = set()

    def add_topic(value: str) -> None:
        topic = _clean_text(value, _MAX_TOPIC_CHARS)
        key = topic.casefold()
        if _is_useful_topic(topic) and key not in seen:
            seen.add(key)
            topics.append(topic)

    stem = Path(file_name).stem
    add_topic(stem)
    for match in re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}|\d+(?:[-./]\d+)+", stem):
        if not match.isdigit():
            add_topic(match)
    for part in relative_path.replace("\\", "/").split("/"):
        if part and part != file_name:
            add_topic(Path(part).stem)
    for chunk in chunks:
        for part in (chunk.section_path or "").split("/"):
            add_topic(part)
            if len(topics) >= 16:
                break
        if len(topics) >= 16:
            break

    summary_parts = []
    remaining = _MAX_SUMMARY_CHARS
    for chunk in chunks:
        text = _clean_text(chunk.text, min(600, remaining))
        if text:
            summary_parts.append(text)
            remaining -= len(text)
        if remaining <= 0 or len(summary_parts) >= 4:
            break
    return DocumentProfile(summary=" ".join(summary_parts)[:_MAX_SUMMARY_CHARS], topics=tuple(topics[:16]))


def _parse_profile(output: str) -> DocumentProfile:
    normalized = output.strip()
    fence = chr(96) * 3
    if normalized.startswith(fence):
        normalized = normalized.split("\n", 1)[-1]
        if normalized.endswith(fence):
            normalized = normalized[:-len(fence)]
        normalized = normalized.strip()
    payload = json.loads(normalized)
    summary = _clean_text(str(payload.get("summary") or ""), _MAX_SUMMARY_CHARS)
    raw_topics = payload.get("topics") or []
    if not isinstance(raw_topics, list):
        raw_topics = [raw_topics]
    topics = []
    seen = set()
    for value in raw_topics:
        topic = _clean_text(str(value), _MAX_TOPIC_CHARS)
        key = topic.casefold()
        if topic and key not in seen:
            seen.add(key)
            topics.append(topic)
            if len(topics) >= 16:
                break
    return DocumentProfile(summary=summary, topics=tuple(topics))


def _merge_topics(*groups: Sequence[str]) -> tuple[str, ...]:
    topics = []
    seen = set()
    for group in groups:
        for value in group:
            topic = _clean_text(value, _MAX_TOPIC_CHARS)
            key = topic.casefold()
            if _is_useful_topic(topic) and key not in seen:
                seen.add(key)
                topics.append(topic)
                if len(topics) >= 16:
                    return tuple(topics)
    return tuple(topics)


def _is_useful_topic(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 2:
        return False
    digit_count = sum(char.isdigit() for char in compact)
    return not (digit_count >= 10 and digit_count / len(compact) > 0.5)


def _clean_text(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]
