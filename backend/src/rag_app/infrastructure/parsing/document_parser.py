from __future__ import annotations

import csv
from html.parser import HTMLParser
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable
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
        "",
    }
)


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


class DocumentParser:
    def supports(self, file_name: str) -> bool:
        return Path(file_name).suffix.lower() in SUPPORTED_SUFFIXES

    def parse(self, file_name: str, content: bytes) -> list[ParsedChunk]:
        suffix = Path(file_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedDocumentTypeError(f"暂不支持的文件类型: {suffix or '无扩展名'}")

        if suffix == ".pdf":
            sources = self._parse_pdf(content)
        elif suffix == ".docx":
            sources = self._parse_docx(content)
        elif suffix == ".doc":
            sources = self._parse_doc(content)
        elif suffix == ".pptx":
            sources = self._parse_pptx(content)
        elif suffix in {".xlsx", ".xlsm"}:
            sources = self._parse_xlsx(content)
        elif suffix == ".xls":
            sources = self._parse_xls(content)
        elif suffix == ".csv":
            sources = self._parse_csv(content)
        elif suffix == ".json":
            sources = self._parse_json(content)
        elif suffix == ".xml":
            sources = self._parse_xml(content)
        elif suffix in {".html", ".htm"}:
            sources = self._parse_html(content)
        elif suffix in {".ptpt", ".jcpt", ".stpt"}:
            sources = self._parse_zip_container(content)
        elif suffix in {".dll", ".gdb", ".att"}:
            sources = self._parse_binary(file_name, content)
        elif not suffix:
            sources = self._parse_without_extension(file_name, content)
        else:
            sources = [(None, None, _decode_text(content))]

        return self._chunk_sources(sources)

    def _parse_pdf(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        from pypdf import PdfReader

        return [
            (index + 1, None, page.extract_text() or "")
            for index, page in enumerate(PdfReader(BytesIO(content)).pages)
        ]

    def _parse_docx(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        from docx import Document

        document = Document(BytesIO(content))
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
            return [(None, None, document.Content.Text)]
        except Exception:
            return self._parse_binary("legacy-word.doc", content)
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

    def _parse_pptx(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        from pptx import Presentation

        sources = []
        for slide_number, slide in enumerate(Presentation(BytesIO(content)).slides, start=1):
            parts: list[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    text = shape.text.strip()
                    if text:
                        parts.append(text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        values = [cell.text.strip() for cell in row.cells]
                        if any(values):
                            parts.append("\t".join(values))
            sources.append((slide_number, f"第 {slide_number} 页", "\n".join(parts)))
        return sources

    def _parse_xlsx(self, content: bytes) -> list[tuple[int | None, str | None, str]]:
        from openpyxl import load_workbook

        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
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
        sources = []
        total_size = 0
        try:
            with ZipFile(BytesIO(content)) as archive:
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
