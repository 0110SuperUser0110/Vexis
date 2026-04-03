from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractedFact:
    subject: str
    relation: str
    value: str
    confidence: float
    fact_type: str
    source_text: str
    should_store: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "value": self.value,
            "confidence": self.confidence,
            "fact_type": self.fact_type,
            "source_text": self.source_text,
            "should_store": self.should_store,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass
class FactExtractionResult:
    success: bool
    facts: list[ExtractedFact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "facts": [fact.to_dict() for fact in self.facts],
            "notes": self.notes,
            "error": self.error,
            "metadata": self.metadata,
        }


class FactExtractor:
    """
    Deterministic fact extractor for VEX.

    Goal:
    - convert user teaching and uploaded knowledge into structured facts
    - identify runtime/self-model facts
    - identify scientific definitions, formulas, causal relations, and scope constraints
    - identify simple arithmetic equations and taught arithmetic rules
    - avoid pretending every sentence is a fact
    """

    def extract(self, text: str) -> FactExtractionResult:
        raw = (text or "").strip()
        normalized = self._normalize(raw)

        if not normalized:
            return FactExtractionResult(success=False, error="empty_input")

        facts: list[ExtractedFact] = []
        notes: list[str] = []

        arithmetic_facts = self._extract_arithmetic_facts(raw, normalized)
        if arithmetic_facts:
            facts.extend(arithmetic_facts)
            notes.extend(["arithmetic fact extracted" for _ in arithmetic_facts])

        runtime_fact = self._extract_runtime_fact(raw, normalized)
        if runtime_fact is not None:
            facts.append(runtime_fact)
            notes.append("runtime fact extracted")

        location_fact = self._extract_location_fact(raw, normalized)
        if location_fact is not None:
            facts.append(location_fact)
            notes.append("location fact extracted")

        identity_fact = self._extract_identity_fact(raw, normalized)
        if identity_fact is not None:
            facts.append(identity_fact)
            notes.append("identity fact extracted")

        definition_fact = self._extract_definition_fact(raw, normalized)
        if definition_fact is not None:
            facts.append(definition_fact)
            notes.append("definition fact extracted")

        formula_fact = self._extract_formula_fact(raw, normalized)
        if formula_fact is not None:
            facts.append(formula_fact)
            notes.append("formula fact extracted")

        causal_facts = self._extract_causal_facts(raw, normalized)
        if causal_facts:
            facts.extend(causal_facts)
            notes.extend(["causal fact extracted" for _ in causal_facts])

        scope_fact = self._extract_scope_fact(raw, normalized)
        if scope_fact is not None:
            facts.append(scope_fact)
            notes.append("scope fact extracted")

        object_property_fact = self._extract_object_property_fact(raw, normalized)
        if object_property_fact is not None:
            facts.append(object_property_fact)
            notes.append("object property fact extracted")

        relation_fact = self._extract_general_relation_fact(raw, normalized)
        if relation_fact is not None:
            facts.append(relation_fact)
            notes.append("general relation fact extracted")

        deduped = self._dedupe_facts(facts)
        if not deduped:
            return FactExtractionResult(
                success=False,
                facts=[],
                notes=["no structured fact extracted"],
                metadata={"normalized": normalized},
            )

        return FactExtractionResult(
            success=True,
            facts=deduped,
            notes=notes,
            metadata={"normalized": normalized},
        )

    def _extract_arithmetic_facts(self, raw: str, normalized: str) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        equation_patterns = [
            r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(\+|\-|\*|x|/)\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
            r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(\+|\-|\*|x|/)\s*(-?\d+(?:\.\d+)?)\s+(?:equals|is)\s+(-?\d+(?:\.\d+)?)",
            r"(?<!\w)(-?\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by)\s+(-?\d+(?:\.\d+)?)\s+(?:equals|is)\s+(-?\d+(?:\.\d+)?)",
        ]

        for pattern in equation_patterns:
            for match in re.finditer(pattern, normalized):
                left = self._parse_number_literal(match.group(1))
                operator = self._normalize_operator(match.group(2))
                right = self._parse_number_literal(match.group(3))
                stated_result = self._parse_number_literal(match.group(4))
                if left is None or operator is None or right is None or stated_result is None:
                    continue

                computed_result = self._compute_arithmetic_result(left, operator, right)
                if computed_result is None or not self._numbers_equal(computed_result, stated_result):
                    continue

                expression = f"{self._number_to_string(left)}{operator}{self._number_to_string(right)}"
                facts.append(
                    ExtractedFact(
                        subject=expression,
                        relation="equals",
                        value=self._number_to_string(computed_result),
                        confidence=0.98,
                        fact_type="arithmetic_equation",
                        source_text=raw,
                        reasons=["validated_arithmetic_equation"],
                        metadata={
                            "left_operand": left,
                            "right_operand": right,
                            "operator": operator,
                            "result": computed_result,
                        },
                    )
                )

        successor_patterns = [
            r"(?:any|every)\s+number\s*(?:\+|plus)\s*1\s+(?:equals|is)\s+(?:1|one)\s+higher(?:\s+number)?",
            r"adding\s+1\s+to\s+(?:any|every)\s+number\s+(?:makes|gives)\s+(?:it\s+)?(?:1|one)\s+higher",
        ]
        if any(re.search(pattern, normalized) for pattern in successor_patterns):
            facts.append(
                ExtractedFact(
                    subject="integer",
                    relation="successor_rule",
                    value="n+1 increases the integer by 1",
                    confidence=0.86,
                    fact_type="arithmetic_rule",
                    source_text=raw,
                    reasons=["validated_successor_rule"],
                    metadata={
                        "operator": "+",
                        "right_operand": 1,
                        "domain": "integer",
                        "transform": "increment_by_one",
                    },
                )
            )

        return facts

    def _extract_runtime_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = [
            (
                r"(?:you are on|youre on|you are running on|youre running on|runtime device label is)\s+([a-zA-Z0-9_\-]+)",
                "vex",
                "running_on",
                "runtime_device",
            ),
            (
                r"(?:device name is|system name is)\s+([a-zA-Z0-9_\-]+)",
                "vex",
                "running_on",
                "runtime_device",
            ),
        ]
        for pattern, subject, relation, fact_type in patterns:
            match = re.search(pattern, normalized)
            if match:
                value = match.group(1).strip().upper()
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.86,
                    fact_type=fact_type,
                    source_text=raw,
                    reasons=["runtime_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_location_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = [
            (r"(?:location label is)\s+([a-zA-Z0-9_\-]+)", "vex", "location_label", "runtime_location"),
            (r"(?:you are at|youre at|you are in|youre in)\s+([a-zA-Z0-9_\-]+)", "vex", "location_label", "runtime_location"),
            (r"(?:here means|here refers to)\s+([a-zA-Z0-9_\-]+)", "vex", "location_label", "runtime_location"),
        ]
        for pattern, subject, relation, fact_type in patterns:
            match = re.search(pattern, normalized)
            if match:
                value = match.group(1).strip().lower()
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.78,
                    fact_type=fact_type,
                    source_text=raw,
                    reasons=["location_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_identity_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = (
            r"(?:your name is|you are called|youre called)\s+([a-zA-Z0-9_\-]+)",
            r"(?:the active identity name is)\s+([a-zA-Z0-9_\-]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            value = match.group(1).strip()
            if value:
                return ExtractedFact(
                    subject="vex",
                    relation="identity_name",
                    value=value,
                    confidence=0.60,
                    fact_type="identity",
                    source_text=raw,
                    reasons=["identity_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_definition_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        if self._looks_like_arithmetic(normalized):
            return None

        patterns = [
            r"(?:a|an|the)\s+([a-z][a-z0-9_\- ]{1,36})\s+(?:is|means|refers to|denotes)\s+([^.;]{8,180})",
            r"(?:a|an|the)\s+([a-z][a-z0-9_\- ]{1,36})\s+(?:measures|describes)\s+([^.;]{8,180})",
            r"([a-z][a-z0-9_\- ]{1,36})\s+(?:is|means|refers to|denotes)\s+([^.;]{8,180})",
        ]
        blocked_value_starts = (
            "conserved ",
            "proportional to",
            "inversely proportional to",
            "caused by ",
            "causes ",
            "increases ",
            "decreases ",
            "reduces ",
            "raises ",
            "lowers ",
            "depends on ",
            "holds when ",
            "applies when ",
            "cannot exceed ",
        )

        for sentence in self._candidate_sentences(raw):
            normalized_sentence = self._normalize(sentence)
            for pattern in patterns:
                match = re.search(pattern, normalized_sentence)
                if not match:
                    continue
                subject = self._clean_phrase(match.group(1))
                value = self._clean_phrase(match.group(2))
                subject_words = subject.split()
                if not subject or not value:
                    continue
                if self._is_blocked_phrase(subject) or len(subject_words) > 5 or len(value.split()) < 3:
                    continue
                if value in {"a", "an", "the", "thing", "stuff"}:
                    continue
                if value.startswith(blocked_value_starts):
                    continue
                if any(operator in value for operator in (" plus ", " minus ", " times ", " multiplied by ", " divided by ")):
                    continue
                return ExtractedFact(
                    subject=subject,
                    relation="defined_as",
                    value=value,
                    confidence=0.74,
                    fact_type="definition",
                    source_text=sentence,
                    reasons=["definition_pattern_match"],
                    metadata={"normalized_value": value, "subject_word_count": len(subject_words)},
                )
        return None

    def _extract_formula_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        if self._looks_like_arithmetic(normalized):
            return None

        patterns = [
            r"(?:the )?([a-z][a-z0-9_\- ]{1,24})\s+(?:equals|is equal to|is)\s+([a-z][a-z0-9_\- ]{1,20})\s+(plus|minus|times|multiplied by|divided by)\s+([a-z][a-z0-9_\- ]{1,20})",
        ]
        for sentence in self._candidate_sentences(raw):
            normalized_sentence = self._normalize(sentence)
            for pattern in patterns:
                match = re.search(pattern, normalized_sentence)
                if not match:
                    continue
                subject = self._clean_phrase(match.group(1))
                left_term = self._clean_phrase(match.group(2))
                operator = self._normalize_operator(match.group(3))
                right_term = self._clean_phrase(match.group(4))
                if not subject or not left_term or not operator or not right_term:
                    continue
                if self._is_blocked_phrase(subject) or self._is_blocked_phrase(left_term) or self._is_blocked_phrase(right_term):
                    continue
                expression = f"{left_term} {operator} {right_term}"
                return ExtractedFact(
                    subject=subject,
                    relation="equals_expression",
                    value=expression,
                    confidence=0.72,
                    fact_type="formula",
                    source_text=sentence,
                    reasons=["formula_pattern_match"],
                    metadata={
                        "left_term": left_term,
                        "operator": operator,
                        "right_term": right_term,
                        "normalized_value": expression,
                    },
                )
        return None

    def _extract_causal_facts(self, raw: str, normalized: str) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        patterns = [
            (r"([a-z][a-z0-9_\- ]{1,36})\s+(causes|increase|increases|decrease|decreases|reduces|raises|lowers)\s+([a-z][a-z0-9_\- ]{1,48})", None),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+depends on\s+([a-z][a-z0-9_\- ]{1,48})", "depends_on"),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+is proportional to\s+([a-z][a-z0-9_\- ]{1,48})", "proportional_to"),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+is inversely proportional to\s+([a-z][a-z0-9_\- ]{1,48})", "inversely_proportional_to"),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+increases as\s+([a-z][a-z0-9_\- ]{1,48})\s+increases", "increases_with"),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+decreases as\s+([a-z][a-z0-9_\- ]{1,48})\s+increases", "decreases_with"),
        ]
        relation_map = {
            "cause": "causes",
            "causes": "causes",
            "increase": "increases",
            "increases": "increases",
            "decrease": "decreases",
            "decreases": "decreases",
            "reduces": "reduces",
            "raises": "raises",
            "lowers": "lowers",
        }

        for sentence in self._candidate_sentences(raw):
            normalized_sentence = self._normalize(sentence)
            for pattern, explicit_relation in patterns:
                for match in re.finditer(pattern, normalized_sentence):
                    subject = self._clean_phrase(match.group(1))
                    relation = explicit_relation or relation_map.get(match.group(2).strip().lower(), "")
                    value_index = 3 if explicit_relation is None else 2
                    value = self._clean_phrase(match.group(value_index))
                    if not subject or not relation or not value:
                        continue
                    if explicit_relation is None and " as " in normalized_sentence and relation in {"increases", "decreases"}:
                        continue
                    if self._is_blocked_phrase(subject) or self._is_blocked_phrase(value):
                        continue
                    if subject == value:
                        continue
                    facts.append(
                        ExtractedFact(
                            subject=subject,
                            relation=relation,
                            value=value,
                            confidence=0.70,
                            fact_type="causal_relation",
                            source_text=sentence,
                            reasons=["causal_pattern_match"],
                            metadata={"normalized_value": value},
                        )
                    )
        return facts

    def _extract_scope_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = [
            (r"([a-z][a-z0-9_\- ]{1,36})\s+is conserved in\s+([^.;]{4,72})", "conserved_in"),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+(?:holds|applies) when\s+([^.;]{4,72})", "holds_when"),
            (r"([a-z][a-z0-9_\- ]{1,36})\s+cannot exceed\s+([^.;]{2,48})", "cannot_exceed"),
        ]
        for sentence in self._candidate_sentences(raw):
            normalized_sentence = self._normalize(sentence)
            for pattern, relation in patterns:
                match = re.search(pattern, normalized_sentence)
                if not match:
                    continue
                subject = self._clean_phrase(match.group(1))
                value = self._clean_phrase(match.group(2))
                if not subject or not value:
                    continue
                if self._is_blocked_phrase(subject) or self._is_blocked_phrase(value):
                    continue
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.70,
                    fact_type="scope_constraint",
                    source_text=sentence,
                    reasons=["scope_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_object_property_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        if self._looks_like_arithmetic(normalized):
            return None
        patterns = [
            (r"(?:the )?([a-zA-Z0-9_\-]+)\s+(?:color is|is)\s+([a-zA-Z0-9_\-]+)", "attribute"),
            (r"(?:the )?([a-zA-Z0-9_\-]+)\s+(?:uses|has)\s+([a-zA-Z0-9_\-]+)", "attribute"),
        ]
        for pattern, fact_type in patterns:
            match = re.search(pattern, normalized)
            if match:
                subject = self._clean_phrase(match.group(1))
                value = self._clean_phrase(match.group(2))
                if self._is_blocked_phrase(subject) or subject.replace("-", "").isdigit():
                    return None
                if value in {"a", "an", "the", "note", "thing", "conserved", "proportional", "dependent"}:
                    return None
                relation = "color" if "color" in normalized else "property"
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.72,
                    fact_type=fact_type,
                    source_text=raw,
                    reasons=["object_property_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_general_relation_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        pattern = r"(?:the )?([a-z][a-z0-9_\- ]{1,36})\s+(orbits|contains|supports|contradicts|uses|requires|implies)\s+(?:the )?([^.;]{1,60})"
        for sentence in self._candidate_sentences(raw):
            normalized_sentence = self._normalize(sentence)
            match = re.search(pattern, normalized_sentence)
            if match:
                subject = self._clean_phrase(match.group(1))
                relation = match.group(2).strip().lower()
                value = self._clean_phrase(match.group(3))
                if self._is_blocked_phrase(subject) or self._is_blocked_phrase(value):
                    return None
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.68,
                    fact_type="relation",
                    source_text=sentence,
                    reasons=["general_relation_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _dedupe_facts(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[ExtractedFact] = []
        for fact in facts:
            key = (fact.subject, fact.relation, fact.value)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _candidate_sentences(self, text: str) -> list[str]:
        segments = re.split(r"[\r\n]+|(?<=[.!?;])\s+", text or "")
        candidates: list[str] = []
        for segment in segments:
            cleaned = " ".join(segment.strip().split())
            if len(cleaned) < 4:
                continue
            candidates.append(cleaned.strip())
        return candidates

    def _clean_phrase(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().lower().split())
        cleaned = cleaned.strip(" .,:;!?()[]{}\"'")
        return cleaned

    def _is_blocked_phrase(self, phrase: str) -> bool:
        blocked = {
            "you", "your", "i", "we", "they", "he", "she", "it",
            "this", "that", "these", "those", "here", "there",
            "something", "anything", "everything", "nothing",
        }
        words = [word for word in self._clean_phrase(phrase).split() if word]
        if not words:
            return True
        if len(words) > 8:
            return True
        return any(word in blocked for word in words)

    def _looks_like_arithmetic(self, normalized: str) -> bool:
        has_digits = bool(re.search(r"\d", normalized))
        has_operator = bool(
            re.search(r"\bplus\b|\bminus\b|\btimes\b|\bmultiplied by\b|\bdivided by\b", normalized)
            or re.search(r"\d\s*[=+\-*/x]\s*\d", normalized)
        )
        return has_digits and has_operator

    def _normalize_operator(self, operator: str) -> Optional[str]:
        mapping = {
            "+": "+",
            "-": "-",
            "*": "*",
            "x": "*",
            "/": "/",
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
        }
        return mapping.get(operator.strip().lower())

    def _parse_number_literal(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _compute_arithmetic_result(self, left: float, operator: str, right: float) -> Optional[float]:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            if right == 0:
                return None
            return left / right
        return None

    def _numbers_equal(self, left: float, right: float, tolerance: float = 1e-9) -> bool:
        return abs(float(left) - float(right)) <= tolerance

    def _number_to_string(self, value: float) -> str:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(round(numeric, 8)).rstrip("0").rstrip(".")
