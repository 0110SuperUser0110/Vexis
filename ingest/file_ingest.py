from __future__ import annotations

import csv
import json
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
    First-pass file ingest pipeline.

    Current support:
    - .txt
    - .json
    - .csv

    Everything else is registered but not deeply parsed yet.
    """

    SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".log"}
    SUPPORTED_JSON_EXTENSIONS = {".json"}
    SUPPORTED_CSV_EXTENSIONS = {".csv"}

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