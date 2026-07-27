from typing import Any, Mapping, Sequence

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


def format_knowledge_catalog(documents: Sequence[Mapping[str, Any]], max_chars: int = 12_000) -> str:
    """Render a bounded folder and file catalog from ready document metadata."""
    file_paths = set()
    folder_paths = set()
    for document in documents:
        if document.get("status") != "ready":
            continue
        file_name = str(document.get("file_name") or document.get("title") or "").strip()
        folder_path = str(document.get("folder_path") or "").replace("\\", "/").strip("/")
        if not file_name:
            continue
        file_paths.add(f"{folder_path}/{file_name}" if folder_path else file_name)
        if folder_path:
            parts = folder_path.split("/")
            folder_paths.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))

    sorted_folders = sorted(folder_paths, key=str.casefold)
    sorted_files = sorted(file_paths, key=str.casefold)
    lines = [
        f"已入库目录：{len(sorted_folders)} 个文件夹，{len(sorted_files)} 个文件。",
        "[文件夹]",
    ]
    entries = [*(f"- {path}" for path in sorted_folders), "[文件]", *(f"- {path}" for path in sorted_files)]
    omitted = 0
    for index, entry in enumerate(entries):
        candidate = "\n".join([*lines, entry])
        if len(candidate) > max_chars:
            omitted = len(entries) - index
            break
        lines.append(entry)
    if omitted:
        suffix = f"... 另有 {omitted} 项因目录上下文长度限制未显示。"
        while len("\n".join([*lines, suffix])) > max_chars and len(lines) > 2:
            lines.pop()
            omitted += 1
            suffix = f"... 另有 {omitted} 项因目录上下文长度限制未显示。"
        lines.append(suffix)
    return "\n".join(lines)
