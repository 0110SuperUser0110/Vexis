from __future__ import annotations

from typing import Any


class InquiryEngine:
    """
    Generates explicit follow-up questions from weakly supported or conflicting beliefs.

    The goal is to externalize what VEX still needs: replication, mechanism,
    boundary conditions, contradiction resolution, or direct verification.
    """

    def generate_questions_from_beliefs(self, beliefs: list[Any], max_questions: int = 6) -> list[str]:
        questions: list[str] = []
        seen: set[str] = set()

        for belief in beliefs:
            metadata = dict(getattr(belief, "metadata", {}) or {})
            subject = str(metadata.get("subject", "")).strip()
            relation = str(metadata.get("relation", "")).strip().lower()
            value = str(metadata.get("value", "")).strip()
            source_type = str(metadata.get("source_type", "unknown")).strip().lower()
            support_count = int(getattr(belief, "support_count", 0))
            confidence = float(getattr(belief, "confidence_score", 0.0))
            conflicting_values = list(metadata.get("conflicting_values", []) or [])

            if subject and relation and value and support_count < 2:
                self._append(questions, seen, self._independent_support_question(subject, relation, value), max_questions)

            if conflicting_values:
                self._append(
                    questions,
                    seen,
                    self._conflict_resolution_question(subject, relation, value, conflicting_values),
                    max_questions,
                )

            if subject and relation and value and confidence < 0.40:
                mechanism_question = self._mechanism_or_boundary_question(subject, relation, value)
                if mechanism_question:
                    self._append(questions, seen, mechanism_question, max_questions)

            if source_type == "peer_reviewed_journal" and subject and relation and value and support_count < 2:
                self._append(
                    questions,
                    seen,
                    f"Has an independent study replicated the claim that {self._relation_statement(subject, relation, value)}?",
                    max_questions,
                )

            if source_type == "book" and subject and relation and value and support_count < 2:
                self._append(
                    questions,
                    seen,
                    f"What primary source, derivation, or experiment confirms that {self._relation_statement(subject, relation, value)}?",
                    max_questions,
                )

            if len(questions) >= max_questions:
                break

        return questions

    def _independent_support_question(self, subject: str, relation: str, value: str) -> str:
        if relation == "defined_as":
            return f"What independent source confirms that {subject} is {self._clip(value)}?"
        if relation in {"equals", "equals_expression"}:
            return f"What derivation, proof, or independent source supports that {subject} equals {self._clip(value)}?"
        if relation in {"causes", "increases", "decreases", "reduces", "raises", "lowers"}:
            return f"What controlled evidence supports that {subject} {relation} {self._clip(value)}?"
        return f"What independent evidence supports that {self._relation_statement(subject, relation, value)}?"

    def _mechanism_or_boundary_question(self, subject: str, relation: str, value: str) -> str:
        if relation == "defined_as":
            return f"What examples or counterexamples would test whether {subject} is best understood as {self._clip(value)}?"
        if relation in {"supports", "requires", "uses", "contains", "depends_on"}:
            return f"What mechanism explains how {self._relation_statement(subject, relation, value)}?"
        if relation in {"causes", "increases", "decreases", "reduces", "raises", "lowers"}:
            return f"What mechanism and what test would show whether {self._relation_statement(subject, relation, value)}?"
        if relation in {"proportional_to", "inversely_proportional_to", "equals_expression"}:
            return f"What derivation or measurement boundary supports the relation that {self._relation_statement(subject, relation, value)}?"
        if relation in {"conserved_in", "holds_when", "cannot_exceed"}:
            return f"Under what boundary conditions is it true that {self._relation_statement(subject, relation, value)}?"
        if relation in {"color", "property"}:
            return f"What direct observation would verify that {subject} has the property {self._clip(value)}?"
        return ""

    def _conflict_resolution_question(
        self,
        subject: str,
        relation: str,
        value: str,
        conflicting_values: list[str],
    ) -> str:
        alternatives = ", ".join(self._clip(item, 42) for item in conflicting_values[:3])
        if relation in {"equals", "equals_expression"}:
            return f"What evidence resolves whether {subject} equals {self._clip(value)} or {alternatives}?"
        return f"What evidence resolves whether {self._relation_statement(subject, relation, value)} or {alternatives}?"

    def _relation_statement(self, subject: str, relation: str, value: str) -> str:
        if relation == "defined_as":
            return f"{subject} is {self._clip(value)}"
        if relation in {"equals", "equals_expression"}:
            return f"{subject} equals {self._clip(value)}"
        if relation == "depends_on":
            return f"{subject} depends on {self._clip(value)}"
        if relation == "proportional_to":
            return f"{subject} is proportional to {self._clip(value)}"
        if relation == "inversely_proportional_to":
            return f"{subject} is inversely proportional to {self._clip(value)}"
        if relation == "increases_with":
            return f"{subject} increases as {self._clip(value)} increases"
        if relation == "decreases_with":
            return f"{subject} decreases as {self._clip(value)} increases"
        if relation == "conserved_in":
            return f"{subject} is conserved in {self._clip(value)}"
        if relation == "holds_when":
            return f"{subject} holds when {self._clip(value)}"
        if relation == "cannot_exceed":
            return f"{subject} cannot exceed {self._clip(value)}"
        return f"{subject} {relation.replace('_', ' ')} {self._clip(value)}"

    def _append(self, questions: list[str], seen: set[str], question: str, max_questions: int) -> None:
        normalized = question.strip()
        if not normalized or normalized in seen or len(questions) >= max_questions:
            return
        seen.add(normalized)
        questions.append(normalized)

    def _clip(self, text: str, limit: int = 84) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3].rstrip() + "..."
