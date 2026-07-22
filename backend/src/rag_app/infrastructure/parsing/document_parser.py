from io import BytesIO
from pathlib import Path
import re

from ...domain.models import ParsedChunk


class DocumentParser:
    def parse(self, file_name: str, content: bytes) -> list[ParsedChunk]:
        suffix = Path(file_name).suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(BytesIO(content)).pages)]
        elif suffix == ".docx":
            from docx import Document
            pages = [(None, "\n".join(paragraph.text for paragraph in Document(BytesIO(content)).paragraphs))]
        elif suffix in {".md", ".markdown", ".txt", ".html", ".htm"}:
            pages = [(None, content.decode("utf-8", errors="ignore"))]
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix or 'unknown'}")

        chunks: list[ParsedChunk] = []
        for page_number, raw_text in pages:
            text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", raw_text.replace("\x00", ""))).strip()
            start = 0
            while start < len(text):
                end = min(start + 1800, len(text))
                segment = text[start:end].strip()
                if segment:
                    chunks.append(ParsedChunk(index=len(chunks), text=segment, page_number=page_number))
                if end >= len(text):
                    break
                start = max(start + 1, end - 240)
        return chunks

