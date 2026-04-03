from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class KnowledgeChunk:
    chunk_index: int
    title: str
    text: str
    section_name: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "title": self.title,
            "text": self.text,
            "section_name": self.section_name,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeIngestResult:
    success: bool
    source_path: str
    title: str
    source_type: str
    summary: str = ""
    key_terms: list[str] = field(default_factory=list)
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_path": self.source_path,
            "title": self.title,
            "source_type": self.source_type,
            "summary": self.summary,
            "key_terms": self.key_terms,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata,
            "error": self.error,
        }


class KnowledgeIngest:
    """
    Deterministic ingest for long-form knowledge documents.

    Goals:
    - preserve retrievable textbook/document content as chunked memory
    - derive a lightweight source summary and key terms
    - avoid dumping a full book into one memory cell
    """

    def __init__(
        self,
        chunk_target_chars: int = 1200,
        chunk_overlap_chars: int = 160,
        max_chunks: int = 180,
    ) -> None:
        self.chunk_target_chars = chunk_target_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_chunks = max_chunks

    def ingest_text(
        self,
        text: str,
        source_path: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> KnowledgeIngestResult:
        metadata = metadata or {}
        normalized = self._normalize(text)
        if not normalized:
            return KnowledgeIngestResult(
                success=False,
                source_path=source_path,
                title=metadata.get("title", "") or self._title_from_path(source_path),
                source_type=metadata.get("source_type", "unknown"),
                error="empty_text",
                metadata=metadata,
            )

        provided_title = str(metadata.get("title") or "").strip()
        title = provided_title
        if not title or self._looks_like_filename(title):
            title = self._extract_title(normalized, source_path)
        source_type = metadata.get("source_type") or self._infer_source_type(title, normalized, source_path)
        summary = self._build_summary(normalized)
        key_terms = self._extract_key_terms(normalized)
        chunks = self._chunk_text(title=title, text=normalized)

        return KnowledgeIngestResult(
            success=True,
            source_path=source_path,
            title=title,
            source_type=source_type,
            summary=summary,
            key_terms=key_terms,
            chunks=chunks,
            metadata={
                **metadata,
                "char_count": len(normalized),
                "chunk_count": len(chunks),
                "truncated": len(chunks) >= self.max_chunks,
            },
        )

    def _normalize(self, text: str) -> str:
        text = text or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_title(self, text: str, source_path: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            candidate = lines[0]
            if 4 <= len(candidate) <= 180:
                return candidate
        return self._title_from_path(source_path)

    def _title_from_path(self, source_path: str) -> str:
        if source_path:
            return Path(source_path).stem or "Untitled Knowledge Source"
        return "Untitled Knowledge Source"

    def _looks_like_filename(self, text: str) -> bool:
        candidate = text.strip().lower()
        return bool(re.fullmatch(r".+\.[a-z0-9]{2,5}", candidate))

    def _infer_source_type(self, title: str, text: str, source_path: str) -> str:
        lowered_title = title.lower()
        lowered_path = source_path.lower()
        header_text = text[:5000].lower()

        journal_markers = (
            "abstract",
            "methods",
            "materials and methods",
            "results",
            "discussion",
            "doi",
            "conclusion",
        )
        if sum(1 for marker in journal_markers if marker in header_text) >= 3:
            return "peer_reviewed_journal"

        technical_markers = (
            "api",
            "configuration",
            "install",
            "usage",
            "endpoint",
            "specification",
            "protocol",
            "reference",
        )
        if any(marker in header_text for marker in technical_markers) or any(
            marker in lowered_title for marker in ("manual", "guide", "reference", "documentation")
        ):
            return "technical_documentation"

        if any(marker in lowered_title for marker in ("textbook", "handbook", "principles", "introduction", "chapter")):
            return "book"
        if any(marker in lowered_path for marker in ("textbook", "handbook", "principles", "chapter")):
            return "book"

        return "book"

    def _build_summary(self, text: str) -> str:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not paragraphs:
            return text[:800].strip()

        summary_parts: list[str] = []
        char_total = 0
        for paragraph in paragraphs[:4]:
            compact = " ".join(paragraph.split())
            if not compact:
                continue
            if char_total + len(compact) > 900:
                break
            summary_parts.append(compact)
            char_total += len(compact)

        summary = " ".join(summary_parts).strip()
        if len(summary) > 900:
            summary = summary[:897].rstrip() + "..."
        return summary

    def _extract_key_terms(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b", text.lower())
        stop = {
            "this", "that", "with", "from", "were", "have", "been", "their",
            "which", "these", "those", "into", "between", "after", "before",
            "using", "used", "chapter", "introduction", "summary", "conclusion",
            "example", "examples", "figure", "table", "therefore", "because",
        }
        freq: dict[str, int] = {}
        for word in words:
            if word in stop:
                continue
            freq[word] = freq.get(word, 0) + 1

        ordered = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in ordered[:25]]

    def _chunk_text(self, title: str, text: str) -> list[KnowledgeChunk]:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: list[KnowledgeChunk] = []
        current_parts: list[str] = []
        current_start = 0
        current_length = 0
        consumed = 0
        current_section = self._detect_section_name(paragraphs[0]) if paragraphs else "full_text"

        for paragraph in paragraphs:
            section_name = self._detect_section_name(paragraph)
            paragraph_text = paragraph.strip()
            paragraph_length = len(paragraph_text)

            if not current_parts:
                current_start = consumed
                current_section = section_name

            projected = current_length + paragraph_length + (2 if current_parts else 0)
            if current_parts and projected > self.chunk_target_chars:
                chunk_text = "\n\n".join(current_parts).strip()
                chunks.append(
                    KnowledgeChunk(
                        chunk_index=len(chunks),
                        title=title,
                        text=chunk_text,
                        section_name=current_section,
                        start_offset=current_start,
                        end_offset=current_start + len(chunk_text),
                        metadata={
                            "char_count": len(chunk_text),
                        },
                    )
                )
                if len(chunks) >= self.max_chunks:
                    break

                overlap_seed = chunk_text[-self.chunk_overlap_chars :].strip()
                current_parts = [overlap_seed] if overlap_seed else []
                current_length = len(overlap_seed)
                current_start = max(0, consumed - len(overlap_seed))
                current_section = section_name

            current_parts.append(paragraph_text)
            current_length += paragraph_length + (2 if len(current_parts) > 1 else 0)
            consumed += paragraph_length + 2

        if current_parts and len(chunks) < self.max_chunks:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(
                KnowledgeChunk(
                    chunk_index=len(chunks),
                    title=title,
                    text=chunk_text,
                    section_name=current_section,
                    start_offset=current_start,
                    end_offset=current_start + len(chunk_text),
                    metadata={
                        "char_count": len(chunk_text),
                    },
                )
            )

        return chunks

    def _detect_section_name(self, paragraph: str) -> str:
        line = paragraph.splitlines()[0].strip().lower()
        if len(line) <= 80 and re.fullmatch(r"[a-z0-9 ,:_\-]+", line):
            return line.replace(" ", "_")
        return "full_text"
