from typing import Mapping, Sequence

from .contracts import SearchHit


def format_retrieved_context(
    hits: Sequence[SearchHit],
    text_by_chunk_id: Mapping[str, str] | None = None,
) -> str:
    text_by_chunk_id = text_by_chunk_id or {}
    parts = []
    for hit in hits:
        header = f"[文档] {hit.title}"
        if hit.folder_path:
            header += f" ({hit.folder_path})"
        lines = [
            header,
            f"[页码] {hit.page_number or '未知'}",
            f"[内容] {text_by_chunk_id.get(hit.chunk_id, hit.text)}",
        ]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
