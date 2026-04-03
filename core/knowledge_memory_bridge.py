from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.belief_engine import BeliefRecord
from core.fact_extractor import FactExtractor
from core.fact_learning_engine import FactLearningEngine
from core.inquiry_engine import InquiryEngine
from core.state_manager import StateManager
from ingest.knowledge_ingest import KnowledgeIngestResult
from memory.memory_store import MemoryStore


@dataclass
class KnowledgeBridgeResult:
    success: bool
    source_path: str
    stored_memory_ids: list[str] = field(default_factory=list)
    stored_fact_count: int = 0
    belief_records: list[BeliefRecord] = field(default_factory=list)
    open_questions_added: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_path": self.source_path,
            "stored_memory_ids": self.stored_memory_ids,
            "stored_fact_count": self.stored_fact_count,
            "belief_records": [belief.to_dict() for belief in self.belief_records],
            "open_questions_added": self.open_questions_added,
            "notes": self.notes,
            "error": self.error,
            "metadata": self.metadata,
        }


class KnowledgeMemoryBridge:
    """
    Stores generic document knowledge as searchable source + chunk memories.

    Rules:
    - keep a compact source summary memory
    - store chunked passage memories for later recall
    - extract deterministic facts from the chunks
    - promote sourced facts into weighted belief candidates
    - generate explicit open questions where support is still thin
    """

    def __init__(
        self,
        state_manager: StateManager,
        memory_store: MemoryStore,
        fact_extractor: Optional[FactExtractor] = None,
        fact_learning_engine: Optional[FactLearningEngine] = None,
        inquiry_engine: Optional[InquiryEngine] = None,
        max_facts_per_chunk: int = 8,
    ) -> None:
        self.state_manager = state_manager
        self.memory_store = memory_store
        self.fact_extractor = fact_extractor or FactExtractor()
        self.fact_learning_engine = fact_learning_engine or FactLearningEngine()
        self.inquiry_engine = inquiry_engine or InquiryEngine()
        self.max_facts_per_chunk = max_facts_per_chunk

    def ingest_result_to_memory(self, result: KnowledgeIngestResult) -> KnowledgeBridgeResult:
        if not result.success:
            return KnowledgeBridgeResult(
                success=False,
                source_path=result.source_path,
                error=result.error or "knowledge_ingest_failed",
                metadata=result.metadata,
            )

        stored_memory_ids: list[str] = []
        stored_fact_count = 0
        belief_records: list[BeliefRecord] = []
        open_questions_added: list[str] = []
        notes: list[str] = []
        fact_memories: list[Any] = []

        try:
            source_memory = self.state_manager.add_memory(
                kind="knowledge_source",
                content=result.summary or f"Knowledge source loaded: {result.title}",
                source="knowledge_ingest",
                related_input_id=None,
                status="active",
                metadata={
                    "source_type": result.source_type,
                    "title": result.title,
                    "source_path": result.source_path,
                    "key_terms": result.key_terms,
                    "chunk_count": len(result.chunks),
                    "independence_group": result.source_path or result.title,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_knowledge_ingest",
                    "source": "knowledge_ingest",
                    "interaction_type": "knowledge_source",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            self.memory_store.save_memory(source_memory)
            stored_memory_ids.append(source_memory.memory_id)
            notes.append("stored knowledge source summary")

            for chunk in result.chunks:
                chunk_memory = self.state_manager.add_memory(
                    kind="knowledge_chunk",
                    content=chunk.text,
                    source="knowledge_ingest",
                    related_input_id=None,
                    status="active",
                    metadata={
                        "source_type": result.source_type,
                        "title": result.title,
                        "source_path": result.source_path,
                        "chunk_index": chunk.chunk_index,
                        "section_name": chunk.section_name,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "chunk_char_count": chunk.metadata.get("char_count", len(chunk.text)),
                        "independence_group": result.source_path or result.title,
                    },
                    interaction_context={
                        "speaker_role": "system",
                        "speaker_id": "vex_knowledge_ingest",
                        "source": "knowledge_ingest",
                        "interaction_type": "knowledge_chunk",
                        "importance": "normal",
                        "store_as_evidence": True,
                        "expects_reply": False,
                        "expects_reasoning": True,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_memory(chunk_memory)
                stored_memory_ids.append(chunk_memory.memory_id)

                extraction = self.fact_extractor.extract(chunk.text)
                if not extraction.success:
                    continue

                for fact in extraction.facts[: self.max_facts_per_chunk]:
                    fact_text = self._render_fact_text(fact.subject, fact.relation, fact.value)
                    fact_memory = self.state_manager.add_memory(
                        kind="fact",
                        content=fact_text,
                        source="knowledge_ingest",
                        related_input_id=None,
                        status="active",
                        metadata={
                            "source_type": result.source_type,
                            "title": result.title,
                            "source_path": result.source_path,
                            "subject": fact.subject,
                            "relation": fact.relation,
                            "value": fact.value,
                            "confidence": fact.confidence,
                            "fact_type": fact.fact_type,
                            "reasons": fact.reasons,
                            "render_text": fact_text,
                            "metadata": fact.metadata,
                            "chunk_index": chunk.chunk_index,
                            "independence_group": result.source_path or result.title,
                        },
                        interaction_context={
                            "speaker_role": "system",
                            "speaker_id": "vex_knowledge_ingest",
                            "source": "knowledge_ingest",
                            "interaction_type": "fact",
                            "importance": "high",
                            "store_as_evidence": True,
                            "expects_reply": False,
                            "expects_reasoning": True,
                            "personality_allowed": False,
                        },
                    )
                    self.memory_store.save_memory(fact_memory)
                    stored_memory_ids.append(fact_memory.memory_id)
                    fact_memories.append(fact_memory)
                    stored_fact_count += 1

            notes.append(f"stored {len(result.chunks)} knowledge chunks")
            if stored_fact_count:
                notes.append(f"extracted {stored_fact_count} structured facts from uploaded knowledge")

            belief_records = self.fact_learning_engine.build_beliefs_from_fact_memories(fact_memories)
            for belief in belief_records:
                belief_memory = self.state_manager.add_memory(
                    kind="belief_candidate",
                    content=belief.statement,
                    source="fact_learning_engine",
                    status=belief.status,
                    metadata={
                        "belief_id": belief.belief_id,
                        "confidence_score": belief.confidence_score,
                        "confidence_label": belief.confidence_label,
                        "status": belief.status,
                        "support_count": belief.support_count,
                        "contradiction_count": belief.contradiction_count,
                        "supporting_sources": belief.supporting_sources,
                        "contradicting_sources": belief.contradicting_sources,
                        "reasons": belief.reasons,
                        **dict(belief.metadata or {}),
                    },
                    interaction_context={
                        "speaker_role": "system",
                        "speaker_id": "vex_fact_learning",
                        "source": "fact_learning_engine",
                        "interaction_type": "belief_candidate",
                        "importance": "high",
                        "store_as_evidence": True,
                        "expects_reply": False,
                        "expects_reasoning": True,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_memory(belief_memory)
                stored_memory_ids.append(belief_memory.memory_id)

            if belief_records:
                notes.append(f"promoted {len(belief_records)} weighted belief candidates from sourced facts")

            open_questions_added = self.inquiry_engine.generate_questions_from_beliefs(belief_records, max_questions=6)
            for question in open_questions_added:
                self.state_manager.add_open_question(question)

            if open_questions_added:
                notes.append(f"generated {len(open_questions_added)} follow-up questions from weakly supported beliefs")

            return KnowledgeBridgeResult(
                success=True,
                source_path=result.source_path,
                stored_memory_ids=stored_memory_ids,
                stored_fact_count=stored_fact_count,
                belief_records=belief_records,
                open_questions_added=open_questions_added,
                notes=notes,
                metadata={
                    "title": result.title,
                    "source_type": result.source_type,
                    "chunk_count": len(result.chunks),
                    **result.metadata,
                },
            )

        except Exception as exc:
            return KnowledgeBridgeResult(
                success=False,
                source_path=result.source_path,
                stored_memory_ids=stored_memory_ids,
                stored_fact_count=stored_fact_count,
                belief_records=belief_records,
                open_questions_added=open_questions_added,
                notes=notes,
                error=str(exc),
                metadata={
                    "title": result.title,
                    "source_type": result.source_type,
                    **result.metadata,
                },
            )

    def _render_fact_text(self, subject: str, relation: str, value: str) -> str:
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
