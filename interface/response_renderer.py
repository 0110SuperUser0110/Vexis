from __future__ import annotations

from typing import Optional

from core.schemas import BlackboardEntry, Claim, Evidence, Question, StateSnapshot


class ResponseRenderer:
    """
    Converts internal VEXIS state objects into readable user-facing responses.

    This is intentionally simple in v1:
    - concise
    - explicit about uncertainty
    - separates observation from conclusion
    - suitable for both text display and speech output
    """

    def render_system_status(self, state: StateSnapshot) -> str:
        current_time = state.current_time_context

        time_text = "unknown time"
        date_text = "unknown date"
        weekday_text = "unknown weekday"
        day_part_text = "unknown part of day"

        if current_time is not None:
            time_text = current_time.time_label or time_text
            date_text = current_time.date_label or date_text
            weekday_text = current_time.weekday_name or weekday_text
            day_part_text = current_time.day_part or day_part_text

        return (
            f"System active. "
            f"It is {time_text} on {weekday_text}, {date_text}. "
            f"Current day part is {day_part_text}. "
            f"Boot count is {state.boot_count}."
        )

    def render_claim_summary(self, claim: Claim) -> str:
        bias_text = self._format_bias_flags(claim.bias_flags)
        confidence_text = claim.confidence.value
        evidence_strength_text = claim.evidence_strength.value

        response = (
            f"Claim under review: {claim.content} "
            f"Current confidence is {confidence_text}. "
            f"Evidence strength is {evidence_strength_text}."
        )

        if bias_text:
            response += f" Bias indicators: {bias_text}."

        if claim.uncertainty_reason:
            response += f" Uncertainty reason: {claim.uncertainty_reason}."

        return response

    def render_evidence_summary(self, evidence: Evidence) -> str:
        bias_text = self._format_bias_flags(evidence.bias_flags)
        source_type = evidence.provenance.source_type
        title = evidence.provenance.title or "untitled source"

        response = (
            f"Evidence reviewed from {source_type}: {title}. "
            f"Strength is {evidence.strength.value}. "
            f"Rigor score is {evidence.rigor_score:.2f}."
        )

        if bias_text:
            response += f" Bias indicators: {bias_text}."

        if evidence.notes:
            response += f" Notes: {evidence.notes}."

        return response

    def render_question(self, question: Question) -> str:
        status = "resolved" if question.resolved else "unresolved"
        return (
            f"Epistemic question: {question.content} "
            f"Priority is {question.priority:.2f}. "
            f"Status is {status}."
        )

    def render_blackboard_entry(self, entry: BlackboardEntry) -> str:
        return (
            f"Blackboard note from {entry.source_module}: "
            f"{entry.content} "
            f"Priority {entry.priority:.2f}."
        )

    def render_top_focus(
        self,
        question: Optional[Question] = None,
        claim: Optional[Claim] = None,
        entry: Optional[BlackboardEntry] = None,
    ) -> str:
        """
        Returns the highest-priority focus in a user-readable way.
        Preference order:
        1. unresolved epistemic question
        2. claim
        3. blackboard entry
        """

        if question is not None:
            return self.render_question(question)

        if claim is not None:
            return self.render_claim_summary(claim)

        if entry is not None:
            return self.render_blackboard_entry(entry)

        return "No active epistemic focus is currently available."

    def render_boot_summary(
        self,
        state: StateSnapshot,
        top_working_item: Optional[str] = None,
        top_blackboard_note: Optional[str] = None,
    ) -> str:
        status = self.render_system_status(state)

        details = []
        details.append(status)

        details.append(f"Tracked claims: {len(state.claims)}.")
        details.append(f"Tracked evidence items: {len(state.evidence)}.")
        details.append(f"Tracked questions: {len(state.questions)}.")
        details.append(f"Recent contradictions: {len(state.contradictions)}.")

        if state.crossed_day_boundary:
            details.append("A day boundary has been crossed since the prior run.")

        if state.previous_time_context is not None:
            details.append(
                "Previous observed time was "
                f"{state.previous_time_context.time_label} on "
                f"{state.previous_time_context.weekday_name}, "
                f"{state.previous_time_context.date_label}."
            )

        if top_working_item:
            details.append(f"Top working item: {top_working_item}.")

        if top_blackboard_note:
            details.append(f"Top blackboard note: {top_blackboard_note}.")

        return " ".join(details)

    def render_epistemic_update(
        self,
        claim: Claim,
        evidence: Evidence,
        question: Question,
    ) -> str:
        claim_bias_text = self._format_bias_flags(claim.bias_flags)
        evidence_bias_text = self._format_bias_flags(evidence.bias_flags)

        response = (
            f"I have reviewed a claim and generated a question. "
            f"The current claim is: {claim.content} "
            f"I formed this question: {question.content} "
            f"Evidence strength is {evidence.strength.value} "
            f"with a rigor score of {evidence.rigor_score:.2f}."
        )

        if claim_bias_text:
            response += f" Claim bias indicators: {claim_bias_text}."

        if evidence_bias_text:
            response += f" Evidence bias indicators: {evidence_bias_text}."

        return response

    def render_speech_ready_status(
        self,
        state: StateSnapshot,
        epistemic_question: Optional[str] = None,
    ) -> str:
        """
        Shorter response for speech output.
        """

        current_time = state.current_time_context

        if current_time is None:
            base = "System active."
        else:
            base = (
                f"It is {current_time.time_label} on {current_time.weekday_name}. "
                f"It is currently {current_time.day_part}."
            )

        if epistemic_question:
            return f"{base} I have formed an epistemic question for review. {epistemic_question}"

        return f"{base} No urgent epistemic question is currently active."

    def render_uncertainty_notice(
        self,
        reason: str,
        suggestion: Optional[str] = None,
    ) -> str:
        response = f"Uncertainty remains. Reason: {reason}."
        if suggestion:
            response += f" Next useful step: {suggestion}."
        return response

    def render_contradiction_notice(
        self,
        contradiction_text: str,
        recommendation: Optional[str] = None,
    ) -> str:
        response = f"A contradiction has been detected. {contradiction_text}."
        if recommendation:
            response += f" Recommended next step: {recommendation}."
        return response

    def _format_bias_flags(self, bias_flags: list) -> str:
        if not bias_flags:
            return ""

        values = []
        for flag in bias_flags:
            try:
                values.append(flag.value.replace("_", " "))
            except AttributeError:
                values.append(str(flag).replace("_", " "))

        return ", ".join(values)