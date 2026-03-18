from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.methodology_engine import MethodologyAssessment, MethodologyEngine


@dataclass
class JournalSection:
    name: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
        }


@dataclass
class ExtractedFinding:
    statement: str
    category: str
    support_strength: str
    source_section: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "category": self.category,
            "support_strength": self.support_strength,
            "source_section": self.source_section,
            "metadata": self.metadata,
        }


@dataclass
class JournalIngestResult:
    success: bool
    source_path: str
    title: str
    source_type: str
    sections: list[JournalSection] = field(default_factory=list)
    extracted_findings: list[ExtractedFinding] = field(default_factory=list)
    methodology: Optional[MethodologyAssessment] = None
    abstract_summary: str = ""
    key_terms: list[str] = field(default_factory=list)
    belief_candidates: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_path": self.source_path,
            "title": self.title,
            "source_type": self.source_type,
            "sections": [section.to_dict() for section in self.sections],
            "extracted_findings": [finding.to_dict() for finding in self.extracted_findings],
            "methodology": self.methodology.to_dict() if self.methodology else None,
            "abstract_summary": self.abstract_summary,
            "key_terms": self.key_terms,
            "belief_candidates": self.belief_candidates,
            "open_questions": self.open_questions,
            "error": self.error,
            "metadata": self.metadata,
        }


class JournalIngest:
    """
    Selective journal ingest layer for VEX.

    Goals:
    - extract useful scientific structure
    - do not store every sentence as active cognition
    - identify methodology quality
    - generate findings, candidate beliefs, and open questions
    """

    def __init__(self, methodology_engine: Optional[MethodologyEngine] = None) -> None:
        self.methodology_engine = methodology_engine or MethodologyEngine()

    def ingest_text(
        self,
        text: str,
        source_path: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> JournalIngestResult:
        metadata = metadata or {}
        normalized = self._normalize(text)
        if not normalized:
            return JournalIngestResult(
                success=False,
                source_path=source_path,
                title=metadata.get("title", ""),
                source_type=metadata.get("source_type", "unknown"),
                error="empty_text",
            )

        title = metadata.get("title") or self._extract_title(normalized)
        source_type = metadata.get("source_type", "peer_reviewed_journal")
        sections = self._split_sections(normalized)
        abstract_summary = self._build_abstract_summary(sections)
        key_terms = self._extract_key_terms(normalized)
        findings = self._extract_findings(sections)
        methodology = self.methodology_engine.assess(
            text=normalized,
            metadata={
                "source_type": source_type,
                "title": title,
                "year": metadata.get("year"),
            },
        )
        belief_candidates = self._build_belief_candidates(findings, methodology)
        open_questions = self._build_open_questions(findings, methodology)

        return JournalIngestResult(
            success=True,
            source_path=source_path,
            title=title,
            source_type=source_type,
            sections=sections,
            extracted_findings=findings,
            methodology=methodology,
            abstract_summary=abstract_summary,
            key_terms=key_terms,
            belief_candidates=belief_candidates,
            open_questions=open_questions,
            metadata=metadata,
        )

    def ingest_file(
        self,
        path: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> JournalIngestResult:
        metadata = metadata or {}
        source_path = str(Path(path))

        try:
            file_path = Path(path)
            if not file_path.exists():
                return JournalIngestResult(
                    success=False,
                    source_path=source_path,
                    title=metadata.get("title", ""),
                    source_type=metadata.get("source_type", "unknown"),
                    error=f"file not found: {source_path}",
                )

            text = file_path.read_text(encoding="utf-8", errors="replace")
            return self.ingest_text(text=text, source_path=source_path, metadata=metadata)

        except Exception as exc:
            return JournalIngestResult(
                success=False,
                source_path=source_path,
                title=metadata.get("title", ""),
                source_type=metadata.get("source_type", "unknown"),
                error=str(exc),
                metadata=metadata,
            )

    def _normalize(self, text: str) -> str:
        text = text or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_title(self, text: str) -> str:
        first_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not first_lines:
            return "Untitled Journal Source"
        title = first_lines[0]
        if len(title) > 180:
            title = title[:180].rstrip()
        return title

    def _split_sections(self, text: str) -> list[JournalSection]:
        known_headings = [
            "abstract",
            "introduction",
            "background",
            "methods",
            "materials and methods",
            "results",
            "discussion",
            "conclusion",
            "limitations",
            "references",
        ]

        lines = text.splitlines()
        sections: list[JournalSection] = []
        current_name = "full_text"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            normalized = stripped.lower()

            if normalized in known_headings:
                if current_lines:
                    sections.append(
                        JournalSection(
                            name=current_name,
                            text="\n".join(current_lines).strip(),
                        )
                    )
                current_name = normalized
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                JournalSection(
                    name=current_name,
                    text="\n".join(current_lines).strip(),
                )
            )

        return sections

    def _build_abstract_summary(self, sections: list[JournalSection]) -> str:
        for section in sections:
            if section.name == "abstract":
                return self._compress_text(section.text, 800)

        if sections:
            return self._compress_text(sections[0].text, 800)

        return ""

    def _extract_key_terms(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b", text.lower())
        stop = {
            "this", "that", "with", "from", "were", "have", "been", "their",
            "which", "these", "those", "into", "between", "after", "before",
            "using", "used", "results", "study", "method", "methods", "conclusion",
            "discussion", "introduction", "abstract",
        }

        freq: dict[str, int] = {}
        for word in words:
            if word in stop:
                continue
            freq[word] = freq.get(word, 0) + 1

        ordered = sorted(freq.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in ordered[:20]]

    def _extract_findings(self, sections: list[JournalSection]) -> list[ExtractedFinding]:
        findings: list[ExtractedFinding] = []
        target_sections = {"abstract", "results", "discussion", "conclusion", "full_text"}

        for section in sections:
            if section.name not in target_sections:
                continue

            sentences = self._split_sentences(section.text)
            for sentence in sentences:
                category = self._sentence_category(sentence)
                if category is None:
                    continue

                findings.append(
                    ExtractedFinding(
                        statement=self._compress_text(sentence, 260),
                        category=category,
                        support_strength=self._finding_strength(sentence),
                        source_section=section.name,
                    )
                )

        return findings[:50]

    def _build_belief_candidates(
        self,
        findings: list[ExtractedFinding],
        methodology: MethodologyAssessment,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for finding in findings:
            if finding.category not in {"result", "conclusion", "association"}:
                continue

            base_confidence = {
                "strong": 0.26,
                "moderate": 0.18,
                "weak": 0.10,
            }.get(finding.support_strength, 0.08)

            confidence = min(base_confidence + (methodology.max_support_contribution * 0.35), 0.55)

            candidates.append(
                {
                    "statement": finding.statement,
                    "category": finding.category,
                    "candidate_confidence": round(confidence, 3),
                    "source_section": finding.source_section,
                    "study_type": methodology.study_type,
                    "rigor_label": methodology.rigor_label,
                }
            )

        return candidates[:20]

    def _build_open_questions(
        self,
        findings: list[ExtractedFinding],
        methodology: MethodologyAssessment,
    ) -> list[str]:
        questions: list[str] = []

        if methodology.overreach_detected:
            questions.append("Does the paper's conclusion exceed the strength of its reported evidence?")

        if methodology.sample_strength in {"weak", "unknown"}:
            questions.append("Would a larger sample materially change confidence in these findings?")

        if methodology.control_strength in {"weak", "unknown"}:
            questions.append("How much of the reported effect may be due to missing or limited controls?")

        for finding in findings[:8]:
            if finding.category == "association":
                questions.append(f'Is the reported association causal or only correlational: "{finding.statement}"?')

        deduped: list[str] = []
        seen: set[str] = set()
        for question in questions:
            if question not in seen:
                seen.add(question)
                deduped.append(question)

        return deduped[:15]

    def _split_sentences(self, text: str) -> list[str]:
        rough = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in rough if len(s.strip()) > 20]

    def _sentence_category(self, sentence: str) -> Optional[str]:
        lowered = sentence.lower()

        if any(phrase in lowered for phrase in ("we found", "our results show", "results showed", "was associated with")):
            return "result"

        if any(phrase in lowered for phrase in ("conclude", "suggest", "indicate", "support the idea")):
            return "conclusion"

        if any(phrase in lowered for phrase in ("associated with", "correlated with", "linked to", "relationship between")):
            return "association"

        if any(phrase in lowered for phrase in ("limitations include", "a limitation", "future research")):
            return "limitation"

        return None

    def _finding_strength(self, sentence: str) -> str:
        lowered = sentence.lower()

        if any(phrase in lowered for phrase in ("significant", "confidence interval", "effect size", "randomized")):
            return "strong"

        if any(phrase in lowered for phrase in ("associated with", "suggests", "indicates", "correlated")):
            return "moderate"

        return "weak"

    def _compress_text(self, text: str, limit: int) -> str:
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."