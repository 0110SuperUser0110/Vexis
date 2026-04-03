from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.state_manager import StateManager
from memory.memory_store import MemoryStore


BOOTSTRAP_ID = "epistemic_bootstrap_v1"
BOOTSTRAP_SOURCE_PATH = "vexis://epistemic/bootstrap"
BOOTSTRAP_TITLE = "Vexis Epistemic Bootstrap"


PRINCIPLES: list[dict[str, Any]] = [
    {
        "id": "provenance_required",
        "statement": "Claims require evidence with traceable provenance.",
        "subject": "claims",
        "relation": "require",
        "value": "evidence with traceable provenance",
        "category": "evidence",
        "confidence": 0.94,
        "label": "established",
        "status": "established",
    },
    {
        "id": "replication_increases_confidence",
        "statement": "Independent replication increases confidence in a claim.",
        "subject": "independent replication",
        "relation": "increases_confidence_in",
        "value": "a claim",
        "category": "evidence",
        "confidence": 0.93,
        "label": "established",
        "status": "established",
    },
    {
        "id": "correlation_not_causation",
        "statement": "Correlation alone does not establish causation.",
        "subject": "correlation",
        "relation": "does_not_establish",
        "value": "causation",
        "category": "causality",
        "confidence": 0.95,
        "label": "established",
        "status": "established",
    },
    {
        "id": "contradiction_lowers_confidence",
        "statement": "Contradictory evidence lowers confidence until the conflict is resolved.",
        "subject": "contradictory evidence",
        "relation": "lowers",
        "value": "confidence until the conflict is resolved",
        "category": "evidence",
        "confidence": 0.92,
        "label": "established",
        "status": "established",
    },
    {
        "id": "model_scope_matters",
        "statement": "Models must be evaluated within their scope and assumptions.",
        "subject": "models",
        "relation": "require",
        "value": "scope and assumption checks",
        "category": "modeling",
        "confidence": 0.92,
        "label": "established",
        "status": "established",
    },
    {
        "id": "measurement_limits_inference",
        "statement": "Measurement quality constrains inference quality.",
        "subject": "measurement quality",
        "relation": "constrains",
        "value": "inference quality",
        "category": "measurement",
        "confidence": 0.91,
        "label": "high",
        "status": "supported",
    },
    {
        "id": "definitions_not_empirical_proof",
        "statement": "Definitions organize meaning but do not constitute empirical proof.",
        "subject": "definitions",
        "relation": "do_not_constitute",
        "value": "empirical proof",
        "category": "logic",
        "confidence": 0.93,
        "label": "established",
        "status": "established",
    },
    {
        "id": "math_consistency_required",
        "statement": "Quantitative claims require mathematical consistency.",
        "subject": "quantitative claims",
        "relation": "require",
        "value": "mathematical consistency",
        "category": "mathematics",
        "confidence": 0.95,
        "label": "established",
        "status": "established",
    },
    {
        "id": "mechanism_strengthens_causal_claims",
        "statement": "Mechanistic explanation strengthens causal claims.",
        "subject": "mechanistic explanation",
        "relation": "strengthens",
        "value": "causal claims",
        "category": "causality",
        "confidence": 0.90,
        "label": "high",
        "status": "supported",
    },
    {
        "id": "extraordinary_claims_need_stronger_evidence",
        "statement": "Extraordinary claims require proportionally stronger evidence.",
        "subject": "extraordinary claims",
        "relation": "require",
        "value": "proportionally stronger evidence",
        "category": "evidence",
        "confidence": 0.93,
        "label": "established",
        "status": "established",
    },
]


@dataclass
class BootstrapResult:
    success: bool
    added_count: int = 0
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EpistemicBootstrap:
    def __init__(self, state_manager: StateManager, memory_store: MemoryStore) -> None:
        self.state_manager = state_manager
        self.memory_store = memory_store

    def ensure_bootstrapped(self) -> BootstrapResult:
        try:
            existing = self.memory_store.load_memories()
            if any((memory.metadata or {}).get("bootstrap_id") == BOOTSTRAP_ID for memory in existing):
                return BootstrapResult(
                    success=True,
                    added_count=0,
                    notes=["epistemic bootstrap already present"],
                    metadata={"bootstrap_id": BOOTSTRAP_ID},
                )

            added_count = 0
            notes: list[str] = []

            source_memory = self.state_manager.add_memory(
                kind="knowledge_source",
                content=(
                    "Bootstrapped methodological substrate for Vexis: evidence, causality, replication, "
                    "mathematical consistency, scope, measurement, and contradiction handling."
                ),
                source="epistemic_bootstrap",
                related_input_id=None,
                status="active",
                metadata={
                    "bootstrap_id": BOOTSTRAP_ID,
                    "source_type": "axiomatic_methodology",
                    "title": BOOTSTRAP_TITLE,
                    "source_path": BOOTSTRAP_SOURCE_PATH,
                    "independence_group": BOOTSTRAP_ID,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_epistemic_bootstrap",
                    "source": "epistemic_bootstrap",
                    "interaction_type": "knowledge_source",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            self.memory_store.save_memory(source_memory)
            added_count += 1

            for principle in PRINCIPLES:
                fact_memory = self.state_manager.add_memory(
                    kind="fact",
                    content=f"{principle['subject']} {principle['relation']} {principle['value']}",
                    source="epistemic_bootstrap",
                    related_input_id=None,
                    status="active",
                    metadata={
                        "bootstrap_id": BOOTSTRAP_ID,
                        "source_type": "axiomatic_methodology",
                        "title": BOOTSTRAP_TITLE,
                        "source_path": BOOTSTRAP_SOURCE_PATH,
                        "subject": principle["subject"],
                        "relation": principle["relation"],
                        "value": principle["value"],
                        "confidence": principle["confidence"],
                        "fact_type": "epistemic_principle",
                        "category": principle["category"],
                        "independence_group": BOOTSTRAP_ID,
                        "render_text": principle["statement"],
                    },
                    interaction_context={
                        "speaker_role": "system",
                        "speaker_id": "vex_epistemic_bootstrap",
                        "source": "epistemic_bootstrap",
                        "interaction_type": "fact",
                        "importance": "high",
                        "store_as_evidence": True,
                        "expects_reply": False,
                        "expects_reasoning": True,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_memory(fact_memory)
                added_count += 1

                belief_memory = self.state_manager.add_memory(
                    kind="belief_candidate",
                    content=principle["statement"],
                    source="epistemic_bootstrap",
                    related_input_id=None,
                    status=principle["status"],
                    metadata={
                        "bootstrap_id": BOOTSTRAP_ID,
                        "belief_id": f"belief_bootstrap_{principle['id']}",
                        "confidence_score": principle["confidence"],
                        "confidence_label": principle["label"],
                        "status": principle["status"],
                        "support_count": 1,
                        "contradiction_count": 0,
                        "supporting_sources": [BOOTSTRAP_SOURCE_PATH],
                        "contradicting_sources": [],
                        "reasons": ["bootstrapped_epistemic_principle"],
                        "origin": "epistemic_bootstrap",
                        "source_type": "axiomatic_methodology",
                        "title": BOOTSTRAP_TITLE,
                        "source_path": BOOTSTRAP_SOURCE_PATH,
                        "subject": principle["subject"],
                        "relation": principle["relation"],
                        "value": principle["value"],
                        "category": principle["category"],
                        "independence_group": BOOTSTRAP_ID,
                        "render_text": principle["statement"],
                    },
                    interaction_context={
                        "speaker_role": "system",
                        "speaker_id": "vex_epistemic_bootstrap",
                        "source": "epistemic_bootstrap",
                        "interaction_type": "belief_candidate",
                        "importance": "high",
                        "store_as_evidence": True,
                        "expects_reply": False,
                        "expects_reasoning": True,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_memory(belief_memory)
                added_count += 1

            notes.append(f"bootstrapped {len(PRINCIPLES)} epistemic principles")
            return BootstrapResult(
                success=True,
                added_count=added_count,
                notes=notes,
                metadata={"bootstrap_id": BOOTSTRAP_ID, "principle_count": len(PRINCIPLES)},
            )

        except Exception as exc:
            return BootstrapResult(
                success=False,
                added_count=0,
                notes=[f"epistemic bootstrap failed: {exc}"],
                metadata={"bootstrap_id": BOOTSTRAP_ID},
            )
