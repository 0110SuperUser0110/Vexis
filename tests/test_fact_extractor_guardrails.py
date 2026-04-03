from __future__ import annotations

import unittest

from core.fact_extractor import FactExtractor


class TestFactExtractorGuardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = FactExtractor()

    def test_runtime_extractor_does_not_treat_bare_your_as_device_fact(self) -> None:
        result = self.extractor.extract("your pattern is odd")

        self.assertFalse(any(fact.relation == "running_on" for fact in result.facts))

    def test_identity_extractor_does_not_treat_generic_you_are_as_identity_fact(self) -> None:
        result = self.extractor.extract("you are persistent")

        self.assertFalse(any(fact.relation == "identity_name" for fact in result.facts))

    def test_location_extractor_does_not_treat_here_is_as_location_fact(self) -> None:
        result = self.extractor.extract("here is some text from the page")

        self.assertFalse(any(fact.relation == "location_label" for fact in result.facts))


if __name__ == "__main__":
    unittest.main()
