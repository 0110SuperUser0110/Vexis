from __future__ import annotations

import csv
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class IngestResult:
    filepath: str
    filename: str
    success: bool
    file_type: str
    extracted_text: str = ""
    extracted_data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class FileIngest:
    """
    Deterministic file ingest pipeline.

    Current support:
    - .txt / .md / .log
    - .json
    - .csv
    - .pdf (machine-readable text extraction with scanned-page OCR fallback)
    - common image formats via local OCR
    """

    SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".log"}
    SUPPORTED_JSON_EXTENSIONS = {".json"}
    SUPPORTED_CSV_EXTENSIONS = {".csv"}
    SUPPORTED_PDF_EXTENSIONS = {".pdf"}
    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, max_pdf_ocr_pages: int = 60) -> None:
        self._rapidocr_engine: Any = None
        self._rapidocr_attempted = False
        self.max_pdf_ocr_pages = max_pdf_ocr_pages

    def ingest_file(self, filepath: str) -> IngestResult:
        path = Path(filepath)

        if not path.exists():
            return IngestResult(
                filepath=str(path),
                filename=path.name,
                success=False,
                file_type="missing",
                error="File does not exist.",
            )

        ext = path.suffix.lower()

        try:
            if ext in self.SUPPORTED_TEXT_EXTENSIONS:
                return self._ingest_text(path)

            if ext in self.SUPPORTED_JSON_EXTENSIONS:
                return self._ingest_json(path)

            if ext in self.SUPPORTED_CSV_EXTENSIONS:
                return self._ingest_csv(path)

            if ext in self.SUPPORTED_PDF_EXTENSIONS:
                return self._ingest_pdf(path)

            if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                return self._ingest_image(path)

            return self._register_unknown(path)

        except Exception as exc:
            return IngestResult(
                filepath=str(path),
                filename=path.name,
                success=False,
                file_type=ext or "unknown",
                error=str(exc),
            )

    def ingest_files(self, filepaths: list[str]) -> list[IngestResult]:
        return [self.ingest_file(filepath) for filepath in filepaths]

    def _ingest_text(self, path: Path) -> IngestResult:
        text = path.read_text(encoding="utf-8", errors="ignore")

        return IngestResult(
            filepath=str(path),
            filename=path.name,
            success=True,
            file_type="text",
            extracted_text=text,
            extracted_data={
                "char_count": len(text),
                "line_count": len(text.splitlines()),
            },
        )

    def _ingest_json(self, path: Path) -> IngestResult:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        parsed = json.loads(raw)

        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)

        top_level_type = type(parsed).__name__
        top_level_keys = list(parsed.keys())[:50] if isinstance(parsed, dict) else []

        return IngestResult(
            filepath=str(path),
            filename=path.name,
            success=True,
            file_type="json",
            extracted_text=pretty,
            extracted_data={
                "top_level_type": top_level_type,
                "top_level_keys": top_level_keys,
                "item_count": len(parsed) if hasattr(parsed, "__len__") else None,
            },
        )

    def _ingest_csv(self, path: Path) -> IngestResult:
        rows: list[dict[str, Any]] = []
        headers: list[str] = []

        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx >= 24:
                    break

        preview_lines = []
        if headers:
            preview_lines.append(", ".join(headers))
            for row in rows:
                preview_lines.append(", ".join(str(row.get(h, "")) for h in headers))

        preview_text = "\n".join(preview_lines)

        return IngestResult(
            filepath=str(path),
            filename=path.name,
            success=True,
            file_type="csv",
            extracted_text=preview_text,
            extracted_data={
                "headers": headers,
                "preview_row_count": len(rows),
            },
        )

    def _ingest_pdf(self, path: Path) -> IngestResult:
        pypdf = self._import_optional_module("pypdf")
        reader = pypdf.PdfReader(str(path))

        page_entries: list[tuple[int, str]] = []
        pages_with_text = 0
        pages_needing_ocr: list[int] = []

        for index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            cleaned = page_text.strip()
            if cleaned:
                pages_with_text += 1
                page_entries.append((index, cleaned))
            else:
                pages_needing_ocr.append(index)

        ocr_entries: list[tuple[int, str]] = []
        ocr_truncated = False
        if pages_needing_ocr:
            ocr_entries, ocr_truncated = self._ocr_pdf_pages(path, pages_needing_ocr)
            page_entries.extend(ocr_entries)

        page_entries.sort(key=lambda item: item[0])
        extracted_text = "\n\n".join(
            f"[page {page_index + 1}]\n{page_text}"
            for page_index, page_text in page_entries
            if page_text.strip()
        )

        metadata = {
            "page_count": len(reader.pages),
            "pages_with_text": pages_with_text,
            "pages_with_ocr": len(ocr_entries),
            "parser": "pypdf+rapidocr" if ocr_entries else "pypdf",
            "ocr_truncated": ocr_truncated,
        }
        if not extracted_text.strip():
            metadata["message"] = "PDF parsed, but no readable text was extracted."

        return IngestResult(
            filepath=str(path),
            filename=path.name,
            success=True,
            file_type="pdf",
            extracted_text=extracted_text,
            extracted_data=metadata,
        )

    def _ingest_image(self, path: Path) -> IngestResult:
        image_module = self._import_optional_module("PIL.Image")
        image = image_module.open(path)
        width, height = image.size
        image_format = image.format
        image.close()

        lines, confidences = self._ocr_image(path)
        extracted_text = "\n".join(lines).strip()
        average_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        metadata = {
            "width": width,
            "height": height,
            "image_format": image_format,
            "ocr_engine": "rapidocr_onnxruntime",
            "ocr_line_count": len(lines),
            "ocr_average_confidence": average_confidence,
        }
        if not extracted_text:
            metadata["message"] = "Image OCR completed, but no readable text was detected."

        return IngestResult(
            filepath=str(path),
            filename=path.name,
            success=True,
            file_type="image",
            extracted_text=extracted_text,
            extracted_data=metadata,
        )

    def _ocr_image(self, path: Path) -> tuple[list[str], list[float]]:
        return self._ocr_from_source(str(path))

    def _ocr_pdf_pages(self, path: Path, page_indexes: list[int]) -> tuple[list[tuple[int, str]], bool]:
        if not page_indexes:
            return [], False

        pymupdf = self._import_optional_module("pymupdf")
        document = pymupdf.open(str(path))
        entries: list[tuple[int, str]] = []
        truncated = False

        try:
            for count, page_index in enumerate(page_indexes):
                if count >= self.max_pdf_ocr_pages:
                    truncated = True
                    break

                page = document.load_page(page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                lines, _ = self._ocr_from_source(pix.tobytes("png"))
                page_text = "\n".join(lines).strip()
                if page_text:
                    entries.append((page_index, page_text))
        finally:
            document.close()

        return entries, truncated

    def _ocr_from_source(self, source: Any) -> tuple[list[str], list[float]]:
        engine = self._get_rapidocr_engine()
        result, _ = engine(source)

        if not result:
            return [], []

        lines: list[str] = []
        confidences: list[float] = []
        for item in result:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            lines.append(text)
            try:
                confidences.append(float(item[2]))
            except Exception:
                pass

        return lines, confidences

    def _get_rapidocr_engine(self) -> Any:
        if self._rapidocr_engine is not None:
            return self._rapidocr_engine
        if self._rapidocr_attempted:
            raise RuntimeError("rapidocr engine unavailable")

        self._rapidocr_attempted = True
        rapidocr_module = self._import_optional_module("rapidocr_onnxruntime")
        self._rapidocr_engine = rapidocr_module.RapidOCR()
        return self._rapidocr_engine

    def _import_optional_module(self, module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            self._ensure_local_site_packages()
            return importlib.import_module(module_name)

    def _ensure_local_site_packages(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        local_site = project_root / ".venv" / "Lib" / "site-packages"
        if local_site.exists() and str(local_site) not in sys.path:
            sys.path.insert(0, str(local_site))

    def _register_unknown(self, path: Path) -> IngestResult:
        return IngestResult(
            filepath=str(path),
            filename=path.name,
            success=True,
            file_type=path.suffix.lower() or "unknown",
            extracted_text="",
            extracted_data={
                "message": "File registered but parser not yet implemented for this type."
            },
        )
