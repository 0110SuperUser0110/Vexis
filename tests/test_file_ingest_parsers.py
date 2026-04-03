from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

site_packages = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))

from PIL import Image, ImageDraw, ImageFont

from ingest.file_ingest import FileIngest


class TestFileIngestParsers(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"file_ingest_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.ingest = FileIngest()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_pdf_ingest_extracts_machine_readable_text(self) -> None:
        pdf_path = self.temp_root / "physics.pdf"
        self._write_simple_pdf(pdf_path, "Force equals mass times acceleration")

        result = self.ingest.ingest_file(str(pdf_path))

        self.assertTrue(result.success)
        self.assertEqual(result.file_type, "pdf")
        self.assertIn("force equals mass times acceleration", result.extracted_text.lower())
        self.assertEqual(result.extracted_data.get("parser"), "pypdf")
        self.assertEqual(result.extracted_data.get("pages_with_text"), 1)

    def test_scanned_pdf_ingest_uses_ocr_fallback(self) -> None:
        pdf_path = self.temp_root / "scan.pdf"
        image_path = self.temp_root / "scan_source.png"
        self._write_text_image(image_path, "ENERGY IS CONSERVED")
        image = Image.open(image_path)
        image.save(pdf_path, "PDF")
        image.close()

        result = self.ingest.ingest_file(str(pdf_path))

        self.assertTrue(result.success)
        self.assertEqual(result.file_type, "pdf")
        lowered = result.extracted_text.lower()
        self.assertTrue("energy" in lowered or "conserved" in lowered)
        self.assertEqual(result.extracted_data.get("parser"), "pypdf+rapidocr")
        self.assertGreaterEqual(result.extracted_data.get("pages_with_ocr", 0), 1)

    def test_image_ingest_ocr_extracts_text(self) -> None:
        image_path = self.temp_root / "law.png"
        self._write_text_image(image_path, "ENERGY IS CONSERVED")

        result = self.ingest.ingest_file(str(image_path))

        self.assertTrue(result.success)
        self.assertEqual(result.file_type, "image")
        lowered = result.extracted_text.lower()
        self.assertTrue("energy" in lowered or "conserved" in lowered)
        self.assertEqual(result.extracted_data.get("ocr_engine"), "rapidocr_onnxruntime")
        self.assertGreaterEqual(result.extracted_data.get("ocr_line_count", 0), 1)

    def _write_text_image(self, path: Path, text: str) -> None:
        image = Image.new("RGB", (1200, 240), color="white")
        draw = ImageDraw.Draw(image)

        font = None
        for candidate in (
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ):
            try:
                font = ImageFont.truetype(candidate, 72)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        draw.text((40, 70), text, fill="black", font=font)
        image.save(path)

    def _write_simple_pdf(self, path: Path, text: str) -> None:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT\n/F1 24 Tf\n72 720 Td\n({escaped}) Tj\nET"

        objects = [
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
            "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
            f"5 0 obj\n<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream\nendobj\n",
        ]

        parts: list[bytes] = [b"%PDF-1.4\n"]
        offsets: list[int] = [0]
        current_offset = len(parts[0])
        for obj in objects:
            offsets.append(current_offset)
            encoded = obj.encode("latin-1")
            parts.append(encoded)
            current_offset += len(encoded)

        xref_start = current_offset
        xref_lines = ["xref", "0 6", "0000000000 65535 f "]
        for offset in offsets[1:]:
            xref_lines.append(f"{offset:010d} 00000 n ")
        xref_block = ("\n".join(xref_lines) + "\n").encode("latin-1")
        trailer = (
            f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
        ).encode("latin-1")

        path.write_bytes(b"".join(parts + [xref_block, trailer]))


if __name__ == "__main__":
    unittest.main()
