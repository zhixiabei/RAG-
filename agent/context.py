from typing import Mapping, Sequence

from .contracts import SearchHit


def retrieved_file_path(hit: SearchHit) -> str:
    if hit.relative_path:
        return hit.relative_path.replace("\\", "/")
    file_name = hit.file_name or hit.title
    folder_path = hit.folder_path.replace("\\", "/").strip("/")
    return f"{folder_path}/{file_name}" if folder_path else file_name


def format_retrieved_context(
    hits: Sequence[SearchHit],
    text_by_chunk_id: Mapping[str, str] | None = None,
) -> str:
    text_by_chunk_id = text_by_chunk_id or {}
    parts = []
    for hit in hits:
        file_name = hit.file_name or hit.title
        lines = [
            f"[完整路径] {retrieved_file_path(hit)}",
            f"[文件名] {file_name}",
            f"[页码] {hit.page_number or '未知'}",
            f"[内容] {text_by_chunk_id.get(hit.chunk_id, hit.text)}",
        ]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
