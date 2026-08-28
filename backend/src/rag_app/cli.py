from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    __package__ = "rag_app"

from .config import Settings
from .domain.models import ParsedChunk
from .infrastructure.parsing.document_parser import SUPPORTED_SUFFIXES
from .main import build_services, initialize_services


def import_folder(knowledge_base_id: str, folder: Path) -> int:
    settings = Settings()
    services = build_services(settings)
    try:
        initialize_services(services)
        if not services.repository.knowledge_base_exists(knowledge_base_id):
            raise SystemExit(f"知识库不存在: {knowledge_base_id}")
        if not folder.is_dir():
            raise SystemExit(f"文件夹不存在: {folder}")

        files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
        if not files:
            print("没有找到支持的文件")
            return 0

        succeeded = 0
        failed = 0
        for path in files:
            try:
                with path.open("rb") as stream:
                    services.ingestion.ingest_stream(
                        knowledge_base_id,
                        path.name,
                        "application/octet-stream",
                        stream,
                        settings.max_document_bytes,
                    )
                succeeded += 1
                print(f"[ready] {path}")
            except Exception as exc:
                failed += 1
                print(f"[failed] {path}: {exc}")
        print(f"完成：成功 {succeeded} 个，失败 {failed} 个，总计 {len(files)} 个")
        return 1 if failed else 0
    finally:
        services.repository.close()

def rebuild_document_index(knowledge_base_id: str) -> int:
    settings = Settings()
    services = build_services(settings)
    try:
        initialize_services(services)
        if not services.repository.knowledge_base_exists(knowledge_base_id):
            raise SystemExit(f"知识库不存在: {knowledge_base_id}")

        documents = [
            document
            for document in services.repository.list_documents(knowledge_base_id)
            if document.get("status") == "ready"
        ]
        succeeded = 0
        failed = 0
        for document in documents:
            document_id = str(document["id"])
            try:
                rows = services.repository.list_document_chunks(document_id)
                chunks = [
                    ParsedChunk(
                        index=int(row["chunk_index"]),
                        text=str(row["text"]),
                        page_number=row.get("page_number"),
                        section_path=row.get("section_path"),
                    )
                    for row in rows
                ]
                if not chunks:
                    raise RuntimeError("没有可用于生成文件画像的 chunk")
                file_name = str(
                    document.get("file_name")
                    or document.get("title")
                    or document_id
                )
                folder_path = str(document.get("folder_path") or "")
                relative_path = (
                    f"{folder_path}/{file_name}"
                    if folder_path
                    else file_name
                )
                services.ingestion.index_document_profile(
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    title=str(document.get("title") or file_name),
                    file_name=file_name,
                    folder_path=folder_path,
                    relative_path=relative_path,
                    chunks=chunks,
                )
                succeeded += 1
                print(f"[indexed] {relative_path} ({document_id})")
            except Exception as exc:
                failed += 1
                print(f"[failed] {document_id}: {exc}")
        print(
            f"文件索引重建完成：成功 {succeeded} 个，"
            f"失败 {failed} 个，总计 {len(documents)} 个"
        )
        return 1 if failed else 0
    finally:
        services.repository.close()

def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 知识库批量导入工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    folder_parser = subparsers.add_parser("import-folder", help="递归导入本机文件夹")
    folder_parser.add_argument("--knowledge-base-id", required=True)
    folder_parser.add_argument("--folder", required=True, type=Path)
    rebuild_parser = subparsers.add_parser(
        "rebuild-document-index",
        help="根据已有文档和 chunk 重建独立文件画像索引",
    )
    rebuild_parser.add_argument("--knowledge-base-id", required=True)
    args = parser.parse_args()
    if args.command == "import-folder":
        return import_folder(args.knowledge_base_id, args.folder)
    if args.command == "rebuild-document-index":
        return rebuild_document_index(args.knowledge_base_id)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
