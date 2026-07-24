from typing import Mapping, Sequence

from .contracts import SearchHit


def format_retrieved_context(
    hits: Sequence[SearchHit],
    text_by_chunk_id: Mapping[str, str] | None = None,
) -> str:
    text_by_chunk_id = text_by_chunk_id or {}
    return "\n\n".join(
        (
            f"[文档] {hit.title}\n"
            f"[页码] {hit.page_number or '未知'}\n"
            f"[内容] {text_by_chunk_id.get(hit.chunk_id, hit.text)}"
        )
        for hit in hits
    )
