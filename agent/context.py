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
    entries = _ready_document_catalog_entries(documents)
    file_paths = {path for _folder, path in entries}
    folder_paths = set()
    for folder_path, _path in entries:
        if folder_path:
            parts = folder_path.split("/")
            folder_paths.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))

    sorted_folders = sorted(folder_paths, key=str.casefold)
    sorted_files = sorted(file_paths, key=str.casefold)
    lines = [
        f"已入库目录：{len(sorted_folders)} 个文件夹，{len(sorted_files)} 个文件。",
        "[文件夹]",
    ]
    catalog_lines = [*(f"- {path}" for path in sorted_folders), "[文件]", *(f"- {path}" for path in sorted_files)]
    omitted = 0
    for index, entry in enumerate(catalog_lines):
        candidate = "\n".join([*lines, entry])
        if len(candidate) > max_chars:
            omitted = len(catalog_lines) - index
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


def format_knowledge_catalog_answer(
    question: str,
    documents: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]] = (),
    file_lookup: bool = False,
) -> str:
    """Build an exact Markdown file listing without relying on a language model."""
    entries = _ready_document_catalog_entries(documents)
    if not entries:
        return "当前知识库还没有已完成入库的文件。"

    if file_lookup:
        lookup_text = "\n".join([
            question,
            *(
                str(message.get("content") or "")
                for message in reversed(history)
                if message.get("role") == "user"
            ),
        ]).replace("\\", "/").casefold()
        matches = sorted(
            {
                path
                for _folder, path in entries
                if len(file_name := path.rsplit("/", 1)[-1]) >= 3
                and (file_name.casefold() in lookup_text or path.casefold() in lookup_text)
            },
            key=str.casefold,
        )
        if not matches:
            return "没有在当前知识库的已入库文件中找到该文件。"
        lines = [f"找到了 **{len(matches)}** 个匹配的已入库文件："]
        lines.extend(f"- `{_escape_markdown_code(path)}`" for path in matches)
        return "\n".join(lines)

    folders = sorted({folder for folder, _path in entries if folder}, key=lambda value: (-len(value), value.casefold()))
    targets = _mentioned_catalog_folders(question, folders)
    if not targets:
        for message in reversed(history):
            if message.get("role") != "user":
                continue
            targets = _mentioned_catalog_folders(str(message.get("content") or ""), folders)
            if targets:
                break

    selected = [
        path
        for folder, path in entries
        if not targets or any(folder == target or folder.startswith(f"{target}/") for target in targets)
    ]
    selected = sorted(set(selected), key=str.casefold)
    if targets and not selected:
        return "没有找到该文件夹中的已入库文件。"

    scope = "、".join(f"`{_escape_markdown_code(target)}`" for target in targets) if targets else "知识库"
    lines = [f"{scope}中共有 **{len(selected)}** 个已入库文件："]
    lines.extend(f"- `{_escape_markdown_code(path)}`" for path in selected)
    return "\n".join(lines)


def _ready_document_catalog_entries(documents: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    entries = set()
    for document in documents:
        if document.get("status") != "ready":
            continue
        file_name = str(document.get("file_name") or document.get("title") or "").strip()
        folder_path = str(document.get("folder_path") or "").replace("\\", "/").strip("/")
        if not file_name:
            continue
        path = f"{folder_path}/{file_name}" if folder_path else file_name
        entries.add((folder_path, path))
    return sorted(entries, key=lambda item: item[1].casefold())


def _mentioned_catalog_folders(text: str, folders: Sequence[str]) -> list[str]:
    normalized = text.replace("\\", "/").casefold()
    full_matches = [folder for folder in folders if folder.casefold() in normalized]
    if full_matches:
        return [
            folder
            for folder in full_matches
            if not any(other != folder and other.startswith(f"{folder}/") for other in full_matches)
        ]
    return [folder for folder in folders if folder.rsplit("/", 1)[-1].casefold() in normalized]


def _escape_markdown_code(value: str) -> str:
    return value.replace("`", "\\`")
