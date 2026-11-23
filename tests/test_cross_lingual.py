from __future__ import annotations

import json

from lingua_eval.cross_lingual import CrossLingualAuditor
from lingua_eval.models import EvaluationPrompt, ModelCandidate
from lingua_eval.rlhf_pipeline import RLHFRankingPipeline


def test_numeric_hallucination_detection() -> None:
    auditor = CrossLingualAuditor()
    issues = auditor.check_numeric_entity_preservation(
        "The total is 1.08 USD with a 2.5% fee.",
        "Le total est de 1.18 USD avec des frais de 5%.",
    )

    assert any("1.08" in issue.message for issue in issues)
    assert any("2.5" in issue.message for issue in issues)
    assert any("1.18" in issue.message for issue in issues)


def test_french_technical_terminology_and_false_friend() -> None:
    auditor = CrossLingualAuditor()
    issues = auditor.validate_terminology(
        "The API will eventually retry the request. The library is stable.",
        "L'API réessaiera éventuellement la requête. La librairie est stable.",
    )

    assert any(issue.category == "terminology" for issue in issues)
    assert any("false friend" in issue.message for issue in issues)


def test_rlhf_jsonl_export_schema() -> None:
    prompt = EvaluationPrompt(
        prompt_id="schema-001",
        language="en",
        text="Return 42.",
        expected_entities=["42"],
    )
    pipeline = RLHFRankingPipeline()
    evaluations = pipeline.evaluate_pair(
        prompt,
        ModelCandidate(label="A", model_name="a", response="The answer is 42."),
        ModelCandidate(label="B", model_name="b", response="The answer is 41."),
    )
    record = json.loads(pipeline.export_dpo_jsonl([evaluations]))

    assert set(record) == {"prompt", "chosen", "rejected", "metadata"}
    assert record["metadata"]["prompt_id"] == "schema-001"
    assert record["chosen"] == "The answer is 42."
