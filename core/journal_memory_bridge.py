from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.belief_engine import BeliefEngine, BeliefRecord
from core.evidence_engine import EvidenceEngine
from ingest.journal_ingest import JournalIngestResult
from core.methodology_engine import MethodologyAssessment
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


@dataclass
class JournalBridgeResult:
    success: bool
    source_path: str
    stored_memory_ids: list[str] = field(default_factory=list)
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
            "belief_records": [belief.to_dict() for belief in self.belief_records],
            "open_questions_added": self.open_questions_added,
            "notes": self.notes,
            "error": self.error,
            "metadata": self.metadata,
        }


class JournalMemoryBridge:
    """
    Moves distilled journal ingest results into VEX memory and belief structures.

    Rules:
    - do not store every sentence as core memory
    - store compact scientific findings
    - store methodology assessment
    - store belief candidates only as scored candidates unless support is sufficient
    """

    def __init__(
        self,
        state_manager: StateManager,
        memory_store: MemoryStore,
        evidence_engine: Optional[EvidenceEngine] = None,
        belief_engine: Optional[BeliefEngine] = None,
    ) -> None:
        self.state_manager = state_manager
        self.memory_store = memory_store
        self.evidence_engine = evidence_engine or EvidenceEngine()
        self.belief_engine = belief_engine or BeliefEngine()

    def ingest_result_to_memory(self, result: JournalIngestResult) -> JournalBridgeResult:
        if not result.success:
            return JournalBridgeResult(
                success=False,
                source_path=result.source_path,
                error=result.error or "journal_ingest_failed",
            )

        stored_memory_ids: list[str] = []
        belief_records: list[BeliefRecord] = []
        open_questions_added: list[str] = []
        notes: list[str] = []

        source_type = result.source_type
        title = result.title
        methodology = result.methodology

        try:
            # 1. Store source summary memory.
            summary_memory = self.state_manager.add_memory(
                kind="journal_summary",
                content=result.abstract_summary or f"Journal source loaded: {title}",
                source="journal_ingest",
                related_input_id=None,
                status="active",
                metadata={
                    "source_type": source_type,
                    "title": title,
                    "source_path": result.source_path,
                    "year": result.metadata.get("year"),
                    "doi": result.metadata.get("doi"),
                    "authors": result.metadata.get("authors"),
                    "key_terms": result.key_terms,
                    "independence_group": result.metadata.get("doi") or result.title,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_journal_ingest",
                    "source": "journal_ingest",
                    "interaction_type": "journal_summary",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            self.memory_store.save_memory(summary_memory)
            stored_memory_ids.append(summary_memory.memory_id)
            notes.append("stored journal summary memory")

            # 2. Store methodology assessment memory.
            if methodology is not None:
                methodology_text = self._methodology_text(methodology)
                methodology_memory = self.state_manager.add_memory(
                    kind="methodology_assessment",
                    content=methodology_text,
                    source="journal_ingest",
                    related_input_id=None,
                    status="active",
                    metadata={
                        "source_type": source_type,
                        "title": title,
                        "source_path": result.source_path,
                        "methodology": methodology.to_dict(),
                        "independence_group": result.metadata.get("doi") or result.title,
                    },
                    interaction_context={
                        "speaker_role": "system",
                        "speaker_id": "vex_methodology_engine",
                        "source": "journal_ingest",
                        "interaction_type": "methodology_assessment",
                        "importance": "high",
                        "store_as_evidence": True,
                        "expects_reply": False,
                        "expects_reasoning": True,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_memory(methodology_memory)
                stored_memory_ids.append(methodology_memory.memory_id)
                notes.append("stored methodology assessment memory")

            # 3. Store extracted findings as distilled memory units.
            for finding in result.extracted_findings[:20]:
                finding_memory = self.state_manager.add_memory(
                    kind="journal_finding",
                    content=finding.statement,
                    source="journal_ingest",
                    related_input_id=None,
                    status="active",
                    metadata={
                        "source_type": source_type,
                        "title": title,
                        "source_path": result.source_path,
                        "finding_category": finding.category,
                        "support_strength": finding.support_strength,
                        "source_section": finding.source_section,
                        "doi": result.metadata.get("doi"),
                        "independence_group": result.metadata.get("doi") or result.title,
                    },
                    interaction_context={
                        "speaker_role": "system",
                        "speaker_id": "vex_journal_ingest",
                        "source": "journal_ingest",
                        "interaction_type": "journal_finding",
                        "importance": "normal",
                        "store_as_evidence": True,
                        "expects_reply": False,
                        "expects_reasoning": True,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_memory(finding_memory)
                stored_memory_ids.append(finding_memory.memory_id)

            if result.extracted_findings:
                notes.append(f"stored {min(len(result.extracted_findings), 20)} journal finding memories")

            # 4. Convert belief candidates into belief records using methodology + source scoring.
            for candidate in result.belief_candidates[:12]:
                if methodology is None:
                    continue

                evidence = self.evidence_engine.assess_evidence(
                    statement=candidate["statement"],
                    source_id=f'journal::{result.source_path}::{candidate["source_section"]}',
                    source_type=source_type,
                    methodology=methodology,
                    supports_statement=True,
                    independence_group=result.metadata.get("doi") or result.title,
                    contradiction_count=0,
                    metadata={
                        "title": title,
                        "source_path": result.source_path,
                        "source_section": candidate["source_section"],
                        "candidate_category": candidate["category"],
                    },
                )

                belief = self.belief_engine.build_belief(
                    belief_id=f'belief_{abs(hash(candidate["statement"])) % 10_000_000}',
                    statement=candidate["statement"],
                    evidence_items=[evidence],
                    metadata={
                        "origin": "journal_memory_bridge",
                        "title": title,
                        "source_path": result.source_path,
                        "source_type": source_type,
                    },
                )
                belief_records.append(belief)

            if belief_records:
                notes.append(f"generated {len(belief_records)} belief candidates from journal findings")

            # 5. Add open questions derived from the paper.
            for question in result.open_questions[:10]:
                self.state_manager.add_open_question(question)
                open_questions_added.append(question)

            if open_questions_added:
                notes.append(f"added {len(open_questions_added)} open questions from journal review")

            return JournalBridgeResult(
                success=True,
                source_path=result.source_path,
                stored_memory_ids=stored_memory_ids,
                belief_records=belief_records,
                open_questions_added=open_questions_added,
                notes=notes,
                metadata={
                    "title": title,
                    "source_type": source_type,
                    "doi": result.metadata.get("doi"),
                },
            )

        except Exception as exc:
            return JournalBridgeResult(
                success=False,
                source_path=result.source_path,
                stored_memory_ids=stored_memory_ids,
                belief_records=belief_records,
                open_questions_added=open_questions_added,
                notes=notes,
                error=str(exc),
                metadata={
                    "title": title,
                    "source_type": source_type,
                },
            )

    def _methodology_text(self, methodology: MethodologyAssessment) -> str:
        return (
            f"Study type: {methodology.study_type}. "
            f"Rigor: {methodology.rigor_label} ({methodology.rigor_score}). "
            f"Sample strength: {methodology.sample_strength}. "
            f"Control strength: {methodology.control_strength}. "
            f"Statistics strength: {methodology.statistics_strength}. "
            f"Overreach detected: {methodology.overreach_detected}. "
            f"Limitations acknowledged: {methodology.limitations_acknowledged}. "
            f"Replication signal: {methodology.replication_signal}."
        )