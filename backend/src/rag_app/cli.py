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


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 知识库批量导入工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    folder_parser = subparsers.add_parser("import-folder", help="递归导入本机文件夹")
    folder_parser.add_argument("--knowledge-base-id", required=True)
    folder_parser.add_argument("--folder", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "import-folder":
        return import_folder(args.knowledge_base_id, args.folder)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
