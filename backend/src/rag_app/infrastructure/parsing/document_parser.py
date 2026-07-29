from __future__ import annotations

import csv
from html.parser import HTMLParser
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import posixpath
import re
import struct
import sys
import tempfile
from typing import Any, BinaryIO, Iterable
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from ...domain.models import ParsedChunk


SUPPORTED_SUFFIXES = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xlsm",
        ".xls",
        ".csv",
        ".md",
        ".markdown",
        ".txt",
        ".html",
        ".htm",
        ".xml",
        ".json",
        ".dll",
        ".gdb",
        ".att",
        ".ptpt",
        ".jcpt",
        ".stpt",
        ".ppt",
        ".lst",
        "",
    }
)

GEOMAP_LAYER_STYLE_MAGIC = b"Geomap v3.60 LayerStyle\x00"
GEOMAP_LAYER_STYLE_HEADER_SIZE = 516
GEOMAP_LAYER_STYLE_RECORD_SIZE = 524
GEOMAP_LAYER_STYLE_TEXT_SLOTS = (
    (0, 64),
    (80, 64),
    (160, 64),
    (240, 64),
    (320, 64),
    (400, 64),
    (480, 32),
)
GEOMAP_ALBUM_MAGIC = b"GeoMap Album Information\n"
GEOMAP_ALBUM_TREE_OFFSET = 2048
PACKAGE_RELATIONSHIPS_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_RELATIONSHIPS_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PRESENTATIONML_NAMESPACE = "http://schemas.openxmlformats.org/presentationml/2006/main"
DRAWINGML_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
PPTX_MAX_SLIDE_XML_BYTES = 32 * 1024 * 1024
OPTIONAL_OFFICE_UI_PART_PREFIXES = ("customUI/", "userCustomization/")


class UnsupportedDocumentTypeError(ValueError):
    pass


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _relationship_source_directory(relationship_file: str) -> str:
    directory, file_name = posixpath.split(relationship_file)
    if posixpath.basename(directory) != "_rels" or not file_name.endswith(".rels"):
        return ""
    source_directory = posixpath.dirname(directory)
    return source_directory


def _repair_missing_optional_docx_parts(content: bytes) -> bytes:
    """Remove broken relationships to optional Office UI customization parts."""
    with ZipFile(BytesIO(content)) as source:
        archive_names = set(source.namelist())
        rewritten_relationships: dict[str, bytes] = {}
        relationship_tag = f"{{{PACKAGE_RELATIONSHIPS_NAMESPACE}}}Relationship"

        for entry in source.infolist():
            if not entry.filename.endswith(".rels"):
                continue
            relationship_xml = source.read(entry)
            try:
                root = ElementTree.fromstring(relationship_xml)
            except ElementTree.ParseError:
                continue

            source_directory = _relationship_source_directory(entry.filename)
            changed = False
            for relationship in list(root):
                if relationship.tag != relationship_tag or relationship.get("TargetMode") == "External":
                    continue
                target = relationship.get("Target", "").replace("\\", "/")
                target_part = posixpath.normpath(
                    target.lstrip("/") if target.startswith("/") else posixpath.join(source_directory, target)
                )
                is_optional_ui_part = target_part.startswith(OPTIONAL_OFFICE_UI_PART_PREFIXES)
                if is_optional_ui_part and target_part not in archive_names:
                    root.remove(relationship)
                    changed = True

            if changed:
                rewritten_relationships[entry.filename] = ElementTree.tostring(
                    root,
                    encoding="utf-8",
                    xml_declaration=True,
                )

        if not rewritten_relationships:
            return content

        repaired = BytesIO()
        with ZipFile(repaired, "w") as destination:
            for entry in source.infolist():
                destination.writestr(
                    entry,
                    rewritten_relationships.get(entry.filename, source.read(entry)),
                )
        return repaired.getvalue()


class DocumentParser:
    def supports(self, file_name: str) -> bool:
        return Path(file_name).suffix.lower() in SUPPORTED_SUFFIXES

    def parse(self, file_name: str, content: bytes) -> list[ParsedChunk]:
        return self.parse_stream(file_name, BytesIO(content))

    def parse_stream(self, file_name: str, stream: BinaryIO) -> list[ParsedChunk]:
        suffix = Path(file_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedDocumentTypeError(f"暂不支持的文件类型: {suffix or '无扩展名'}")

        stream.seek(0)
        if suffix == ".pdf":
            sources = self._parse_pdf_stream(stream)
        elif suffix == ".docx":
            sources = self._parse_docx_stream(stream)
        elif suffix == ".doc":
            sources = self._parse_doc(stream.read())
        elif suffix == ".pptx":
            sources = self._parse_pptx_stream(stream)
        elif suffix in {".xlsx", ".xlsm"}:
            sources = self._parse_xlsx_stream(stream)
        elif suffix == ".xls":
            sources = self._parse_xls(stream.read())
        elif suffix == ".csv":
            sources = self._parse_csv(stream.read())
        elif suffix == ".json":
            sources = self._parse_json(stream.read())
        elif suffix == ".xml":
            sources = self._parse_xml(stream.read())
        elif suffix in {".html", ".htm"}:
            sources = self._parse_html(stream.read())
        elif suffix in {".ptpt", ".jcpt", ".stpt"}:
            sources = self._parse_zip_container_stream(stream)
        elif suffix == ".ppt":
            sources = self._parse_ppt(stream.read())
        elif suffix == ".lst":
            sources = self._parse_lst(file_name, stream.read())
        elif suffix == ".att":
            sources = self._parse_att(file_name, stream.read())
        elif suffix in {".dll", ".gdb"}:
            sources = self._parse_binary(file_name, stream.read())
        elif not suffix:
            sources = self._parse_without_extension(file_name, stream.read())
        else:
            sources = [(None, None, _decode_text(stream.read()))]

        return self._chunk_sources(sources)

    def _parse_pdf(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        return self._parse_pdf_stream(BytesIO(content))

    def _parse_pdf_stream(self, stream: BinaryIO) -> list[tuple[int | None, str | None, str]]:
        from pypdf import PdfReader

        return [
            (index + 1, None, page.extract_text() or "")
            for index, page in enumerate(PdfReader(stream).pages)
        ]

    def _parse_docx(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        return self._parse_docx_stream(BytesIO(content))

    def _parse_docx_stream(self, stream: BinaryIO) -> list[tuple[int | None, str | None, str]]:
        from docx import Document

        try:
            document = Document(stream)
        except KeyError:
            stream.seek(0)
            content = stream.read()
            repaired_content = _repair_missing_optional_docx_parts(content)
            if repaired_content is content:
                raise
            document = Document(BytesIO(repaired_content))
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                if any(values):
                    parts.append("\t".join(values))
        return [(None, None, "\n".join(parts))]

    def _parse_doc(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        if sys.platform != "win32":
            return self._parse_binary("legacy-word.doc", content)

        path = ""
        word = None
        document = None
        pythoncom = None
        import threading

        result: list[tuple[int | None, str | None, str]] = []
        error: Exception | None = None
        done = threading.Event()

        def _com_parse() -> None:
            nonlocal result, error, word, document, pythoncom
            try:
                import pythoncom
                from win32com.client import DispatchEx

                pythoncom.CoInitialize()
                with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as temporary:
                    temporary.write(content)
                    path = temporary.name

                word = DispatchEx("Word.Application")
                word.Visible = False
                word.DisplayAlerts = 0
                word.AutomationSecurity = 3
                document = word.Documents.Open(
                    path,
                    ConfirmConversions=False,
                    ReadOnly=True,
                    AddToRecentFiles=False,
                    OpenAndRepair=True,
                    NoEncodingDialog=True,
                )
                result = [(None, None, document.Content.Text)]
            except Exception as exc:
                error = exc
            finally:
                if document is not None:
                    try:
                        document.Close(False)
                    except Exception:
                        pass
                if word is not None:
                    try:
                        word.Quit()
                    except Exception:
                        pass
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                if pythoncom is not None:
                    pythoncom.CoUninitialize()
                done.set()

        thread = threading.Thread(target=_com_parse, daemon=True)
        thread.start()
        if not done.wait(timeout=60):
            # COM 解析超时，回退到二进制提取
            return self._parse_binary("legacy-word.doc", content)
        if error or not result:
            return self._parse_binary("legacy-word.doc", content)
        return result

    def _parse_ppt(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        if sys.platform != "win32":
            return self._parse_binary("legacy-powerpoint.ppt", content)

        path = ""
        app = None
        presentation = None
        pythoncom = None
        import threading

        result: list[tuple[int | None, str | None, str]] = []
        error: Exception | None = None
        done = threading.Event()

        def _com_parse() -> None:
            nonlocal result, error, app, presentation, pythoncom
            try:
                import pythoncom
                from win32com.client import DispatchEx

                pythoncom.CoInitialize()
                with tempfile.NamedTemporaryFile(suffix=".ppt", delete=False) as temporary:
                    temporary.write(content)
                    path = temporary.name

                app = DispatchEx("PowerPoint.Application")
                app.Visible = False
                app.DisplayAlerts = 0
                presentation = app.Presentations.Open(path, WithWindow=False)
                parts: list[str] = []
                for slide in presentation.Slides:
                    for shape in slide.Shapes:
                        if shape.HasTextFrame and shape.TextFrame.HasText:
                            text = shape.TextFrame.TextRange.Text.strip()
                            if text:
                                parts.append(text)
                result = [(None, None, "\n".join(parts))]
            except Exception as exc:
                error = exc
            finally:
                if presentation is not None:
                    try:
                        presentation.Close()
                    except Exception:
                        pass
                if app is not None:
                    try:
                        app.Quit()
                    except Exception:
                        pass
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                if pythoncom is not None:
                    pythoncom.CoUninitialize()
                done.set()

        thread = threading.Thread(target=_com_parse, daemon=True)
        thread.start()
        if not done.wait(timeout=60):
            return self._parse_binary("legacy-powerpoint.ppt", content)
        if error or not result:
            return self._parse_binary("legacy-powerpoint.ppt", content)
        return result

    def _parse_pptx(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        return self._parse_pptx_stream(BytesIO(content))

    def _parse_pptx_stream(self, stream: BinaryIO) -> list[tuple[int | None, str | None, str]]:
        try:
            with ZipFile(stream) as archive:
                sources = []
                for slide_number, slide_name in enumerate(self._pptx_slide_names(archive), start=1):
                    info = archive.getinfo(slide_name)
                    if info.file_size > PPTX_MAX_SLIDE_XML_BYTES:
                        raise ValueError(
                            f"PPTX 第 {slide_number} 页 XML 过大: {info.file_size // (1024 * 1024)} MB"
                        )
                    root = ElementTree.fromstring(archive.read(info))
                    parts = []
                    for paragraph in root.iter(f"{{{DRAWINGML_NAMESPACE}}}p"):
                        text_parts = []
                        for element in paragraph.iter():
                            if element.tag == f"{{{DRAWINGML_NAMESPACE}}}t" and element.text:
                                text_parts.append(element.text)
                            elif element.tag == f"{{{DRAWINGML_NAMESPACE}}}tab":
                                text_parts.append("\t")
                            elif element.tag == f"{{{DRAWINGML_NAMESPACE}}}br":
                                text_parts.append("\n")
                        text = "".join(text_parts).strip()
                        if text:
                            parts.append(text)
                    sources.append((slide_number, f"第 {slide_number} 页", "\n".join(parts)))
                return sources
        except BadZipFile as exc:
            raise ValueError("PPTX 文件损坏或格式不兼容") from exc

    @staticmethod
    def _pptx_slide_names(archive: ZipFile) -> list[str]:
        try:
            presentation = ElementTree.fromstring(archive.read("ppt/presentation.xml"))
            relationships = ElementTree.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
            targets = {
                relationship.get("Id"): posixpath.normpath(
                    posixpath.join("ppt", relationship.get("Target", "").replace("\\", "/"))
                )
                for relationship in relationships
                if relationship.tag == f"{{{PACKAGE_RELATIONSHIPS_NAMESPACE}}}Relationship"
                and relationship.get("TargetMode") != "External"
                and relationship.get("Type", "").endswith("/slide")
            }
            slide_names = []
            relationship_id_key = f"{{{OFFICE_RELATIONSHIPS_NAMESPACE}}}id"
            for slide_id in presentation.iter(f"{{{PRESENTATIONML_NAMESPACE}}}sldId"):
                target = targets.get(slide_id.get(relationship_id_key))
                if target and target in archive.NameToInfo:
                    slide_names.append(target)
            if slide_names:
                return slide_names
        except (KeyError, ElementTree.ParseError):
            pass

        def slide_number(name: str) -> int:
            match = re.search(r"slide(\d+)\.xml$", name)
            return int(match.group(1)) if match else sys.maxsize

        return sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ),
            key=slide_number,
        )

    def _parse_xlsx(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        return self._parse_xlsx_stream(BytesIO(content))

    def _parse_xlsx_stream(self, stream: BinaryIO) -> list[tuple[int | None, str | None, str]]:
        from openpyxl import load_workbook

        workbook = load_workbook(stream, read_only=True, data_only=True)
        try:
            sources = []
            for worksheet in workbook.worksheets:
                rows = self._tabular_rows(worksheet.iter_rows(values_only=True))
                sources.append((None, worksheet.title, "\n".join(rows)))
            return sources
        finally:
            workbook.close()

    def _parse_xls(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        import xlrd

        workbook = xlrd.open_workbook(file_contents=content, on_demand=True)
        try:
            sources = []
            for worksheet in workbook.sheets():
                rows = self._tabular_rows(
                    worksheet.row_values(row_index)
                    for row_index in range(worksheet.nrows)
                )
                sources.append((None, worksheet.name, "\n".join(rows)))
            return sources
        finally:
            workbook.release_resources()

    def _parse_csv(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        text = _decode_text(content)
        sample = text[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
        rows = self._tabular_rows(csv.reader(StringIO(text), dialect))
        return [(None, None, "\n".join(rows))]

    def _parse_json(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        value = json.loads(_decode_text(content))
        return [(None, None, json.dumps(value, ensure_ascii=False, indent=2, default=str))]

    def _parse_xml(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        root = ElementTree.fromstring(content)
        lines = []
        seen = set()
        for element in root.iter():
            values = [element.text, *element.attrib.values()]
            for value in values:
                if not value:
                    continue
                text = str(value).replace("\\r\\n", "\n").strip()
                if text and text not in seen:
                    seen.add(text)
                    lines.append(text)
        return [(None, root.tag, "\n".join(lines))]

    def _parse_html(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        extractor = _TextExtractor()
        extractor.feed(_decode_text(content))
        return [(None, None, "\n".join(extractor.parts))]

    def _parse_zip_container(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        return self._parse_zip_container_stream(BytesIO(content))

    def _parse_zip_container_stream(self, stream: BinaryIO) -> list[tuple[int | None, str | None, str]]:
        sources = []
        total_size = 0
        try:
            with ZipFile(stream) as archive:
                for entry in archive.infolist()[:200]:
                    if entry.is_dir() or entry.file_size > 20 * 1024 * 1024:
                        continue
                    total_size += entry.file_size
                    if total_size > 50 * 1024 * 1024:
                        break
                    suffix = Path(entry.filename).suffix.lower()
                    if suffix not in {".xml", ".json", ".csv", ".txt", ".md", ".html", ".htm"}:
                        continue
                    data = archive.read(entry)
                    if suffix == ".xml":
                        nested = self._parse_xml(data)
                    elif suffix == ".json":
                        nested = self._parse_json(data)
                    elif suffix == ".csv":
                        nested = self._parse_csv(data)
                    elif suffix in {".html", ".htm"}:
                        nested = self._parse_html(data)
                    else:
                        nested = [(None, None, _decode_text(data))]
                    sources.extend(
                        (page_number, entry.filename, text)
                        for page_number, _section, text in nested
                    )
        except BadZipFile as exc:
            raise ValueError("压缩容器损坏或格式不兼容") from exc
        return sources

    def _parse_without_extension(self, file_name: str, content: bytes) -> list[tuple[int | None, str | None, str]]:
        if self._looks_like_text(content):
            return [(None, None, _decode_text(content))]
        if content.startswith(b"PK\x03\x04"):
            return self._parse_zip_container(content)
        return self._parse_binary(file_name, content)

    def _parse_att(self, file_name: str, content: bytes) -> list[tuple[int | None, str | None, str]]:
        if content.startswith(GEOMAP_LAYER_STYLE_MAGIC):
            return self._parse_geomap_layer_style(file_name, content)
        return self._parse_without_extension(file_name, content)

    def _parse_lst(self, file_name: str, content: bytes) -> list[tuple[int | None, str | None, str]]:
        if content.startswith(GEOMAP_ALBUM_MAGIC):
            return self._parse_geomap_album(file_name, content)
        return [(None, None, _decode_text(content))]

    def _parse_geomap_album(
        self,
        file_name: str,
        content: bytes,
    ) -> list[tuple[int | None, str | None, str]]:
        if len(content) <= GEOMAP_ALBUM_TREE_OFFSET:
            raise ValueError("GeoMap Album Information 文件损坏：缺少图册树数据")

        cursor = GEOMAP_ALBUM_TREE_OFFSET

        def read_u32() -> int:
            nonlocal cursor
            if cursor + 4 > len(content):
                raise ValueError("读取 uint32 时超出文件边界")
            value = struct.unpack_from("<I", content, cursor)[0]
            cursor += 4
            return value

        def read_string() -> str:
            nonlocal cursor
            size = read_u32()
            if size > 1024 * 1024 or cursor + size > len(content):
                raise ValueError(f"非法的 GBK 字符串长度: {size}")
            raw = content[cursor:cursor + size]
            cursor += size
            try:
                return raw.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise ValueError("GeoMap 图册字符串不是有效的 GBK/GB18030") from exc

        try:
            stored_file_name = read_string()
            flags = read_u32()
            album_name = read_string()
            category_count = read_u32()
            if category_count > 10_000:
                raise ValueError(f"非法的图册分类数: {category_count}")

            categories: list[tuple[str, list[tuple[str, str, str]]]] = []
            total_items = 0
            for _ in range(category_count):
                node_type = read_u32()
                if node_type != 3:
                    raise ValueError(f"未知的图册分类节点类型: {node_type}")
                category_name = read_string()
                item_count = read_u32()
                if item_count > 100_000:
                    raise ValueError(f"非法的图件数: {item_count}")
                items = []
                for _ in range(item_count):
                    item_type = read_u32()
                    if item_type != 1:
                        raise ValueError(f"未知的图册图件节点类型: {item_type}")
                    display_name = read_string()
                    folder_path = read_string().replace("\\", "/").strip("/")
                    database_name = read_string()
                    items.append((display_name, folder_path, database_name))
                total_items += len(items)
                categories.append((category_name, items))

            if cursor != len(content):
                raise ValueError(f"图册树结束后仍有 {len(content) - cursor} 字节未解析数据")
        except ValueError as exc:
            raise ValueError(f"GeoMap Album Information 文件损坏: {exc}") from exc

        sources: list[tuple[int | None, str | None, str]] = [
            (
                None,
                "GeoMap Album Information",
                "\n".join(
                    (
                        f"文件名: {file_name}",
                        "格式: GeoMap Album Information 图册信息文件",
                        "字符编码: GBK/GB18030",
                        f"图册名: {album_name or stored_file_name}",
                        f"分类数: {len(categories)}",
                        f"图件数: {total_items}",
                        f"图册标志: {flags}",
                    )
                ),
            )
        ]
        for category_name, items in categories:
            for display_name, folder_path, database_name in items:
                relative_path = f"{folder_path}/{database_name}" if folder_path else database_name
                sources.append(
                    (
                        None,
                        f"GeoMap图册/{category_name}/{display_name}",
                        "\n".join(
                            (
                                f"图册: {album_name or stored_file_name}",
                                f"分类: {category_name}",
                                f"图件显示名: {display_name}",
                                f"相对目录: {folder_path or '.'}",
                                f"GeoMap 数据库文件: {database_name}",
                                f"完整相对路径: {relative_path}",
                            )
                        ),
                    )
                )
        return sources

    def _parse_geomap_layer_style(
        self,
        file_name: str,
        content: bytes,
    ) -> list[tuple[int | None, str | None, str]]:
        body_size = len(content) - GEOMAP_LAYER_STYLE_HEADER_SIZE
        if body_size <= 0 or body_size % GEOMAP_LAYER_STYLE_RECORD_SIZE:
            raise ValueError(
                "Geomap v3.60 LayerStyle 文件损坏："
                f"文件头后的数据长度不是 {GEOMAP_LAYER_STYLE_RECORD_SIZE} 字节记录的整数倍"
            )

        record_count = body_size // GEOMAP_LAYER_STYLE_RECORD_SIZE
        sources: list[tuple[int | None, str | None, str]] = [
            (
                None,
                "Geomap v3.60 LayerStyle",
                "\n".join(
                    (
                        f"文件名: {file_name}",
                        "格式: Geomap v3.60 LayerStyle 属性文件",
                        "字符编码: GBK/GB18030",
                        f"LayerStyle 记录数: {record_count}",
                        f"固定记录长度: {GEOMAP_LAYER_STYLE_RECORD_SIZE} 字节",
                    )
                ),
            )
        ]

        for index in range(record_count):
            start = GEOMAP_LAYER_STYLE_HEADER_SIZE + index * GEOMAP_LAYER_STYLE_RECORD_SIZE
            record = content[start:start + GEOMAP_LAYER_STYLE_RECORD_SIZE]
            fields = [
                value
                for offset, width in GEOMAP_LAYER_STYLE_TEXT_SLOTS
                if (value := self._decode_geomap_text_slot(record[offset:offset + width]))
            ]
            if not fields:
                continue
            group_name = fields[0]
            lines = [
                f"LayerStyle 记录: {index + 1}/{record_count}",
                f"图层、对象或属性组: {group_name}",
            ]
            if len(fields) > 1:
                lines.append("原始字段序列: " + " | ".join(fields[1:]))
            sources.append((None, f"LayerStyle/{index + 1:03d}/{group_name}", "\n".join(lines)))
        return sources

    @staticmethod
    def _decode_geomap_text_slot(slot: bytes) -> str:
        raw = slot.split(b"\x00", 1)[0]
        if not raw:
            return ""
        try:
            value = raw.decode("gb18030").strip()
        except UnicodeDecodeError:
            return ""
        if not value or any(ord(character) < 32 for character in value):
            return ""
        meaningful = sum(
            character.isalnum()
            or "\u4e00" <= character <= "\u9fff"
            or character in "._- /"
            for character in value
        )
        return value if meaningful / len(value) >= 0.7 else ""

    def _looks_like_text(self, content: bytes) -> bool:
        sample = content[:8192]
        if not sample:
            return False
        if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
            return True
        zero_ratio = sample.count(0) / len(sample)
        control_count = sum(byte < 9 or 13 < byte < 32 for byte in sample)
        return zero_ratio < 0.02 and control_count / len(sample) < 0.05

    def _parse_binary(self, file_name: str, content: bytes) -> list[tuple[int | None, str | None, str]]:
        strings: list[str] = []
        seen = set()
        total_characters = 0

        def append(value: str) -> None:
            nonlocal total_characters
            value = re.sub(r"\s+", " ", value).strip()
            if len(value) < 4 or value in seen:
                return
            meaningful = sum(character.isalnum() or "\u4e00" <= character <= "\u9fff" for character in value)
            if meaningful / len(value) < 0.35:
                return
            seen.add(value)
            strings.append(value)
            total_characters += len(value)

        for match in re.finditer(rb"[\x20-\x7e]{4,}", content):
            append(match.group().decode("ascii"))
            if total_characters >= 100_000:
                break

        if total_characters < 100_000:
            for match in re.finditer(rb"(?:[\x20-\x7e]\x00){4,}", content):
                append(match.group().decode("utf-16-le"))
                if total_characters >= 100_000:
                    break

        if total_characters < 100_000:
            for match in re.finditer(rb"(?:[\x20-\x7e]|[\x81-\xfe][\x40-\xfe]){4,}", content):
                try:
                    append(match.group().decode("gb18030"))
                except UnicodeDecodeError:
                    continue
                if total_characters >= 100_000:
                    break

        header = f"文件名: {file_name}\n文件大小: {len(content)} 字节\n可读字符串:"
        return [(None, "二进制内容", f"{header}\n" + "\n".join(strings))]

    def _tabular_rows(self, rows: Iterable[Iterable[Any]]) -> list[str]:
        result = []
        for row in rows:
            values = [_cell_text(value) for value in row]
            while values and not values[-1]:
                values.pop()
            if any(values):
                result.append("\t".join(values))
        return result

    def _chunk_sources(
        self,
        sources: Iterable[tuple[int | None, str | None, str]],
    ) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        for page_number, section_path, raw_text in sources:
            text = re.sub(
                r"\n{3,}",
                "\n\n",
                re.sub(r"[ \t]+", " ", raw_text.replace("\x00", "")),
            ).strip()
            start = 0
            while start < len(text):
                end = min(start + 1800, len(text))
                segment = text[start:end].strip()
                if segment:
                    chunks.append(
                        ParsedChunk(
                            index=len(chunks),
                            text=segment,
                            page_number=page_number,
                            section_path=section_path,
                        )
                    )
                if end >= len(text):
                    break
                start = max(start + 1, end - 240)
        return chunks
