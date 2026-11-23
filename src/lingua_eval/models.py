"""Typed data contracts used by the evaluation pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Language = Literal["en", "fr", "uk", "ru"]
Severity = Literal["info", "warning", "error", "critical"]


class EvaluationPrompt(BaseModel):
    """A prompt and the checks that apply to its candidate responses."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    language: Language
    text: str = Field(min_length=1)
    domain: str = "general"
    expected_entities: list[str] = Field(default_factory=list)
    test_assertions: list[str] = Field(default_factory=list)
    terminology: dict[str, str] = Field(default_factory=dict)


class ModelCandidate(BaseModel):
    """One model response in a pairwise comparison."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["A", "B"]
    model_name: str = Field(min_length=1)
    response: str = ""


class RubricScore(BaseModel):
    """Normalized scores for the five evaluation dimensions."""

    model_config = ConfigDict(extra="forbid")

    instruction_following: float = Field(ge=0.0, le=1.0)
    factuality: float = Field(ge=0.0, le=1.0)
    code_correctness: float = Field(ge=0.0, le=1.0)
    linguistic_naturalness: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)

    def weighted_total(self, weights: dict[str, float] | None = None) -> float:
        """Return a weighted score on a zero-to-one scale."""
        active_weights = weights or {
            "instruction_following": 0.20,
            "factuality": 0.20,
            "code_correctness": 0.25,
            "linguistic_naturalness": 0.20,
            "safety": 0.15,
        }
        return round(
            sum(getattr(self, name) * weight for name, weight in active_weights.items()),
            6,
        )


class AuditIssue(BaseModel):
    """A deterministic finding produced by one of the auditors."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    category: str
    message: str
    line: int | None = None
    evidence: str | None = None


class RLHFPreferenceRecord(BaseModel):
    """A pairwise preference record suitable for DPO-style JSONL."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    language: Language
    prompt: str
    chosen: str
    rejected: str
    chosen_model: str | None = None
    rejected_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dpo_dict(self) -> dict[str, Any]:
        """Return the conventional prompt/chosen/rejected JSON object."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": {
                "prompt_id": self.prompt_id,
                "language": self.language,
                "chosen_model": self.chosen_model,
                "rejected_model": self.rejected_model,
                **self.metadata,
            },
        }
