from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.schemas import MemoryRecord


@dataclass
class GroundedAnswerResult:
    resolved: bool
    answer_text: str
    confidence: float
    supporting_points: list[str] = field(default_factory=list)
    source_label: str = ""
    memory_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "answer_text": self.answer_text,
            "confidence": self.confidence,
            "supporting_points": self.supporting_points,
            "source_label": self.source_label,
            "memory_ids": self.memory_ids,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class GroundedAnswerEngine:
    """
    Deterministic grounded-answer builder for retrieved memories.

    This engine does not invent content.
    It selects and compresses already-stored evidence into a short answer.
    """

    def resolve(
        self,
        query_text: str,
        memories: list[MemoryRecord],
    ) -> GroundedAnswerResult:
        query = (query_text or "").strip()
        if not query:
            return GroundedAnswerResult(
                resolved=False,
                answer_text="",
                confidence=0.0,
                reasons=["empty_query"],
            )

        candidates = [
            memory
            for memory in memories
            if memory.kind in {"knowledge_chunk", "knowledge_source", "fact", "resolved_question"}
        ]
        if not candidates:
            return GroundedAnswerResult(
                resolved=False,
                answer_text="",
                confidence=0.0,
                reasons=["no_grounding_candidates"],
            )

        query_terms = self._content_terms(query)
        fact_result = self._resolve_from_fact_memories(query_terms, candidates)
        if fact_result is not None:
            return fact_result

        broad_request = self._is_broad_request(query, query_terms)
        ranked = self._rank_memories(query_terms, candidates)
        if not ranked:
            return GroundedAnswerResult(
                resolved=False,
                answer_text="",
                confidence=0.0,
                reasons=["no_ranked_memories"],
            )

        memory_window = self._memory_window(ranked, broad_request)
        selected_sentences = self._select_sentences(query_terms, memory_window, broad_request)
        source_label = self._source_label([memory for _, memory in ranked])

        if selected_sentences:
            answer_text = self._compose_answer_text(selected_sentences, source_label)
            return GroundedAnswerResult(
                resolved=True,
                answer_text=answer_text,
                confidence=min(0.58 + (0.05 * min(len(selected_sentences), 3)), 0.83),
                supporting_points=selected_sentences[:3],
                source_label=source_label,
                memory_ids=[memory.memory_id for _, memory in memory_window[:4]],
                reasons=["resolved_from_grounded_memory", "sentence_selection"],
                metadata={
                    "broad_request": broad_request,
                    "query_terms": sorted(query_terms),
                },
            )

        topic_summary = self._topic_summary(memory_window, source_label)
        if topic_summary:
            return GroundedAnswerResult(
                resolved=True,
                answer_text=topic_summary,
                confidence=0.54,
                supporting_points=[],
                source_label=source_label,
                memory_ids=[memory.memory_id for _, memory in memory_window[:4]],
                reasons=["resolved_from_topic_summary"],
                metadata={
                    "broad_request": broad_request,
                    "query_terms": sorted(query_terms),
                },
            )

        return GroundedAnswerResult(
            resolved=False,
            answer_text="",
            confidence=0.0,
            source_label=source_label,
            memory_ids=[memory.memory_id for _, memory in ranked[:4]],
            reasons=["grounding_present_but_not_answerable"],
        )

    def _resolve_from_fact_memories(
        self,
        query_terms: set[str],
        memories: list[MemoryRecord],
    ) -> Optional[GroundedAnswerResult]:
        best_match: Optional[tuple[float, MemoryRecord, str, str, str]] = None

        for memory in memories:
            metadata = memory.metadata or {}
            subject = str(metadata.get("subject", "")).strip()
            relation = str(metadata.get("relation", "")).strip()
            value = str(metadata.get("value", "")).strip()
            if not subject or not relation or not value:
                continue

            render_text = str(metadata.get("render_text", "")).strip()
            terms = self._content_terms(" ".join([subject, relation, value, render_text]))
            overlap = len(query_terms.intersection(terms))
            if overlap <= 0:
                continue

            score = float(overlap)
            subject_terms = self._content_terms(subject)
            relation_terms = self._content_terms(relation.replace("_", " "))
            if query_terms.intersection(subject_terms):
                score += 1.0
            if query_terms.intersection(relation_terms):
                score += 0.6

            if best_match is None or score > best_match[0]:
                best_match = (score, memory, subject, relation, value)

        if best_match is None or best_match[0] < 1.0:
            return None

        _, memory, subject, relation, value = best_match
        source_label = self._source_label([memory])
        fact_text = self._render_fact_text(memory, subject, relation, value)
        answer = fact_text
        if source_label:
            answer = f"According to {source_label}, {fact_text[0].lower() + fact_text[1:] if len(fact_text) > 1 else fact_text.lower()}"

        return GroundedAnswerResult(
            resolved=True,
            answer_text=answer,
            confidence=min(float(memory.metadata.get("confidence", 0.74)), 0.86),
            supporting_points=[fact_text],
            source_label=source_label,
            memory_ids=[memory.memory_id],
            reasons=["resolved_from_structured_fact"],
        )

    def _render_fact_text(self, memory: MemoryRecord, subject: str, relation: str, value: str) -> str:
        render_text = str((memory.metadata or {}).get("render_text", "")).strip()
        if render_text:
            return render_text.rstrip(".") + "."

        relation_key = relation.lower()
        if relation_key == "defined_as":
            return f"{subject} is {value}."
        if relation_key in {"equals", "equals_expression"}:
            return f"{subject} equals {value}."
        if relation_key == "depends_on":
            return f"{subject} depends on {value}."
        if relation_key == "proportional_to":
            return f"{subject} is proportional to {value}."
        if relation_key == "inversely_proportional_to":
            return f"{subject} is inversely proportional to {value}."
        if relation_key == "increases_with":
            return f"{subject} increases as {value} increases."
        if relation_key == "decreases_with":
            return f"{subject} decreases as {value} increases."
        if relation_key == "conserved_in":
            return f"{subject} is conserved in {value}."
        if relation_key == "holds_when":
            return f"{subject} holds when {value}."
        if relation_key == "cannot_exceed":
            return f"{subject} cannot exceed {value}."
        return f"{subject} {relation.replace('_', ' ')} {value}."

    def _is_broad_request(self, query_text: str, query_terms: set[str]) -> bool:
        normalized = query_text.strip().lower()
        if len(query_terms) <= 2:
            return True
        broad_patterns = (
            "what is this book about",
            "what does the book say",
            "what does the text say",
            "what is this text about",
            "give me an overview",
        )
        return any(pattern in normalized for pattern in broad_patterns)

    def _rank_memories(
        self,
        query_terms: set[str],
        memories: list[MemoryRecord],
    ) -> list[tuple[float, MemoryRecord]]:
        ranked: list[tuple[float, MemoryRecord]] = []
        source_counts: dict[str, int] = {}

        for memory in memories:
            source_key = self._memory_source_key(memory)
            if source_key:
                source_counts[source_key] = source_counts.get(source_key, 0) + 1

        dominant_source = max(source_counts.items(), key=lambda item: item[1])[0] if source_counts else ""

        for memory in memories:
            metadata = memory.metadata or {}
            combined = " ".join(
                [
                    str(metadata.get("render_text", "")),
                    memory.content,
                    str(metadata.get("title", "")),
                    str(metadata.get("section_name", "")),
                    str(memory.kind),
                ]
            )
            terms = self._content_terms(combined)
            overlap = len(query_terms.intersection(terms))
            score = float(overlap * 3.5)
            score += {
                "fact": 3.0,
                "knowledge_chunk": 2.4,
                "resolved_question": 1.8,
                "knowledge_source": 0.8,
            }.get(memory.kind, 0.2)
            if dominant_source and self._memory_source_key(memory) == dominant_source:
                score += 0.5
            if int(metadata.get("chunk_index", 0)) >= 8 and memory.kind == "knowledge_chunk":
                score += 0.6
            if self._looks_like_front_matter(memory.content):
                score -= 4.0
            ranked.append((score, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _memory_window(
        self,
        ranked: list[tuple[float, MemoryRecord]],
        broad_request: bool,
    ) -> list[tuple[float, MemoryRecord]]:
        if not broad_request:
            return ranked[:6]

        content_chunks = [
            item
            for item in ranked
            if item[1].kind == "knowledge_chunk" and int(item[1].metadata.get("chunk_index", 0)) >= 3
        ]
        if content_chunks:
            return content_chunks[:4]
        return ranked[:6]

    def _select_sentences(
        self,
        query_terms: set[str],
        memory_window: list[tuple[float, MemoryRecord]],
        broad_request: bool,
    ) -> list[str]:
        sentence_candidates: list[tuple[float, str, str]] = []

        for memory_score, memory in memory_window:
            cleaned = self._clean_text(memory.content)
            for sentence in self._split_sentences(cleaned):
                score = self._sentence_score(sentence, query_terms, memory_score, broad_request)
                if score <= 0:
                    continue
                sentence_candidates.append((score, sentence, memory.memory_id))

        sentence_candidates.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        seen_norms: set[str] = set()
        seen_memory_ids: set[str] = set()
        for _, sentence, memory_id in sentence_candidates:
            normalized = self._normalize_text(sentence)
            if normalized in seen_norms:
                continue
            if memory_id in seen_memory_ids and len(selected) >= 2:
                continue
            selected.append(sentence)
            seen_norms.add(normalized)
            seen_memory_ids.add(memory_id)
            if len(selected) >= 3:
                break
        return selected

    def _sentence_score(
        self,
        sentence: str,
        query_terms: set[str],
        memory_score: float,
        broad_request: bool,
    ) -> float:
        if len(sentence) < 28 or len(sentence) > 260:
            return -1.0

        lowered = sentence.lower()
        terms = self._content_terms(sentence)
        overlap = len(query_terms.intersection(terms))
        score = min(memory_score, 7.0) * 0.25
        score += overlap * 4.0

        if any(marker in lowered for marker in (" is ", " are ", " means ", " refers to ", " explains ", " describes ", " defines ", " develops ", " concerned with ", " measures ")):
            score += 0.8

        if sentence[0].islower() and overlap < 2:
            score -= 2.5

        if self._looks_like_front_matter(sentence):
            score -= 5.0

        if any(marker in lowered for marker in ("foreword", "acknowledg", "cover photograph", "table of contents", "contents")):
            score -= 4.0

        if broad_request and overlap == 0 and any(marker in lowered for marker in ("in this part", "this part is concerned with", "we develop", "we assume")):
            score += 1.0

        return score

    def _compose_answer_text(self, sentences: list[str], source_label: str) -> str:
        answer = " ".join(sentences)
        if source_label:
            return f"From {source_label}, {answer[0].lower() + answer[1:] if len(answer) > 1 else answer.lower()}"
        return answer

    def _topic_summary(
        self,
        memory_window: list[tuple[float, MemoryRecord]],
        source_label: str,
    ) -> str:
        topics = self._extract_topics([memory for _, memory in memory_window])
        if not topics:
            return ""
        if source_label:
            return f"{source_label} covers {self._format_series(topics[:5])}."
        return f"The retrieved material covers {self._format_series(topics[:5])}."

    def _extract_topics(self, memories: list[MemoryRecord]) -> list[str]:
        topic_counts: dict[str, int] = {}
        skip = {
            "basic mathematics",
            "serge lang",
            "columbia university",
            "publishing company",
            "reading massachusetts",
        }

        for memory in memories:
            section_name = str((memory.metadata or {}).get("section_name", "")).replace("_", " ").strip()
            if section_name and section_name not in {"full text", "full_text"}:
                topic = " ".join(word.capitalize() for word in section_name.split())
                if topic.lower() not in skip:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 2

            cleaned = self._clean_text(memory.content)
            for sentence in self._split_sentences(cleaned):
                if len(sentence) > 140:
                    continue
                pieces = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\b", sentence)
                for piece in pieces:
                    if piece.lower() in skip:
                        continue
                    topic_counts[piece] = topic_counts.get(piece, 0) + 1

        ordered = sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)
        return [topic for topic, _ in ordered[:6]]

    def _source_label(self, memories: list[MemoryRecord]) -> str:
        for memory in memories:
            title = str((memory.metadata or {}).get("title", "")).strip()
            if title and not self._looks_like_filename(title):
                if title.isupper():
                    return title.title()
                return title

        for memory in memories:
            cleaned = self._clean_text(memory.content)
            match = re.search(r"\b[A-Z]{4,}(?:\s+[A-Z]{4,})?\b", cleaned)
            if match and not self._looks_like_front_matter(match.group(0)):
                return match.group(0).title()
        return ""

    def _memory_source_key(self, memory: MemoryRecord) -> str:
        metadata = memory.metadata or {}
        return str(metadata.get("source_path") or metadata.get("title") or "")

    def _looks_like_front_matter(self, text: str) -> bool:
        lowered = self._clean_text(text).lower()
        markers = (
            "publishing company",
            "columbia university",
            "courtesy of",
            "cover photograph",
            "copyright",
            "isbn",
            "addison-wesley",
            "reading, massachusetts",
            "don mills",
            "menlo park",
            "london",
            "foreword",
            "acknowledg",
            "table of contents",
            "contents",
            "index",
        )
        return any(marker in lowered for marker in markers)

    def _looks_like_filename(self, text: str) -> bool:
        return bool(re.fullmatch(r".+\.[a-z0-9]{2,5}", text.strip().lower()))

    def _split_sentences(self, text: str) -> list[str]:
        raw_parts = re.split(r"(?<=[.!?])\s+", self._clean_text(text))
        parts: list[str] = []
        markers = (
            " In this part ",
            " This part ",
            " We assume ",
            " We then ",
            " The notion of ",
        )
        for raw in raw_parts:
            sentence = raw.strip(" \t\r\n.-")
            if not sentence:
                continue
            for marker in markers:
                if marker in sentence and not sentence.startswith(marker.strip()):
                    sentence = sentence[sentence.index(marker) + 1 :].strip()
                    break
            if sentence:
                parts.append(sentence)
        return parts

    def _clean_text(self, text: str) -> str:
        cleaned = (text or "").replace("\u00ad", "")
        cleaned = re.sub(r"\[page\s+\d+\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _normalize_text(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().lower().split())
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch.isspace())
        return " ".join(cleaned.split())

    def _content_terms(self, text: str) -> set[str]:
        stop = {
            "about",
            "again",
            "also",
            "book",
            "does",
            "from",
            "give",
            "into",
            "just",
            "more",
            "says",
            "tell",
            "text",
            "that",
            "them",
            "there",
            "these",
            "this",
            "through",
            "what",
            "which",
            "with",
            "would",
            "your",
        }
        terms = set(re.findall(r"[a-z0-9']{3,}", text.lower()))
        return {term for term in terms if term not in stop}

    def _format_series(self, items: list[str]) -> str:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            return "the retrieved material"
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"
