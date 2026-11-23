"""Pairwise scoring and report generation for multilingual preference data."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .code_auditor import CodeAuditResult, PythonCodeVerifier
from .cross_lingual import CrossLingualAudit, CrossLingualAuditor
from .models import AuditIssue, EvaluationPrompt, ModelCandidate, RLHFPreferenceRecord, RubricScore


@dataclass(slots=True)
class CandidateEvaluation:
    """One candidate's scores and audit output."""

    candidate: ModelCandidate
    score: RubricScore
    total: float
    code_audit: CodeAuditResult
    cross_lingual_audit: CrossLingualAudit


@dataclass(slots=True)
class PairwiseEvaluation:
    """A pairwise result with the selected preference direction."""

    prompt: EvaluationPrompt
    candidate_a: CandidateEvaluation
    candidate_b: CandidateEvaluation
    chosen_label: str
    rejected_label: str


class RLHFRankingPipeline:
    """Score response pairs and write JSON, Markdown, and DPO JSONL artifacts."""

    def __init__(
        self,
        code_verifier: PythonCodeVerifier | None = None,
        cross_lingual_auditor: CrossLingualAuditor | None = None,
    ) -> None:
        self.code_verifier = code_verifier or PythonCodeVerifier()
        self.cross_lingual_auditor = cross_lingual_auditor or CrossLingualAuditor()

    def evaluate_pair(
        self,
        prompt: EvaluationPrompt,
        candidate_a: ModelCandidate,
        candidate_b: ModelCandidate,
    ) -> PairwiseEvaluation:
        """Score both candidates and select the higher weighted total."""
        evaluated_a = self._evaluate_candidate(prompt, candidate_a)
        evaluated_b = self._evaluate_candidate(prompt, candidate_b)
        if evaluated_a.total >= evaluated_b.total:
            chosen_label, rejected_label = "A", "B"
        else:
            chosen_label, rejected_label = "B", "A"
        return PairwiseEvaluation(prompt, evaluated_a, evaluated_b, chosen_label, rejected_label)

    def evaluate_batch(self, rows: Iterable[dict[str, Any]]) -> list[PairwiseEvaluation]:
        """Evaluate JSON-compatible input rows."""
        evaluations: list[PairwiseEvaluation] = []
        for row in rows:
            prompt = EvaluationPrompt(
                prompt_id=row["prompt_id"],
                language=row["language"],
                text=row["prompt"],
                domain=row.get("domain", "general"),
                expected_entities=row.get("expected_entities", []),
                test_assertions=row.get("test_assertions", []),
                terminology=row.get("terminology", {}),
            )
            candidate_a = ModelCandidate(label="A", **row["candidate_a"])
            candidate_b = ModelCandidate(label="B", **row["candidate_b"])
            evaluations.append(self.evaluate_pair(prompt, candidate_a, candidate_b))
        return evaluations

    def write_reports(
        self,
        evaluations: list[PairwiseEvaluation],
        json_path: str | Path,
        markdown_path: str | Path | None = None,
        jsonl_path: str | Path | None = None,
    ) -> None:
        """Write structured JSON, Markdown, and optional preference JSONL."""
        json_target = Path(json_path)
        json_target.parent.mkdir(parents=True, exist_ok=True)
        json_target.write_text(json.dumps(self._json_report(evaluations), indent=2, ensure_ascii=False), encoding="utf-8")
        if markdown_path is not None:
            markdown_target = Path(markdown_path)
            markdown_target.parent.mkdir(parents=True, exist_ok=True)
            markdown_target.write_text(self._markdown_report(evaluations), encoding="utf-8")
        if jsonl_path is not None:
            jsonl_target = Path(jsonl_path)
            jsonl_target.parent.mkdir(parents=True, exist_ok=True)
            jsonl_target.write_text(self.export_dpo_jsonl(evaluations), encoding="utf-8")

    def export_dpo_jsonl(self, evaluations: Iterable[PairwiseEvaluation]) -> str:
        """Serialize chosen/rejected pairs as newline-delimited JSON."""
        lines: list[str] = []
        for evaluation in evaluations:
            chosen = evaluation.candidate_a if evaluation.chosen_label == "A" else evaluation.candidate_b
            rejected = evaluation.candidate_b if evaluation.chosen_label == "A" else evaluation.candidate_a
            record = RLHFPreferenceRecord(
                prompt_id=evaluation.prompt.prompt_id,
                language=evaluation.prompt.language,
                prompt=evaluation.prompt.text,
                chosen=chosen.candidate.response,
                rejected=rejected.candidate.response,
                chosen_model=chosen.candidate.model_name,
                rejected_model=rejected.candidate.model_name,
                metadata={"chosen_score": chosen.total, "rejected_score": rejected.total},
            )
            lines.append(json.dumps(record.to_dpo_dict(), ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")

    def _evaluate_candidate(self, prompt: EvaluationPrompt, candidate: ModelCandidate) -> CandidateEvaluation:
        code_audit = self.code_verifier.verify(candidate.response, prompt.test_assertions)
        cross_audit = self.cross_lingual_auditor.audit(prompt, candidate.response)
        code_present = bool(code_audit.code_blocks)
        textual_safety_risk = bool(
            re.search(
                r"(?:\bimport\s+subprocess\b|\b(?:execute|run)\b.{0,40}\bshell\b|"
                r"\b(?:выполнить|запустить)\b.{0,40}\bshell\b)",
                candidate.response,
                re.IGNORECASE | re.DOTALL,
            )
        )
        critical_safety = sum(issue.severity == "critical" for issue in code_audit.issues)
        execution_errors = sum(issue.category in {"syntax", "execution", "timeout"} for issue in code_audit.issues)
        linguistic_errors = sum(issue.category in {"language_leakage", "terminology"} for issue in cross_audit.issues)
        factual_errors = sum(issue.category == "numeric_entity" for issue in cross_audit.issues)
        score = RubricScore(
            instruction_following=1.0 if candidate.response.strip() else 0.0,
            factuality=max(0.0, 1.0 - 0.25 * factual_errors),
            code_correctness=(
                1.0 if code_audit.passed else (0.5 if code_present and code_audit.syntax_ok and not execution_errors else 0.0)
            ),
            linguistic_naturalness=max(0.0, 1.0 - 0.25 * linguistic_errors),
            safety=max(0.0, 1.0 - 0.5 * critical_safety - 0.5 * textual_safety_risk),
        )
        return CandidateEvaluation(candidate, score, score.weighted_total(), code_audit, cross_audit)

    @staticmethod
    def _json_report(evaluations: list[PairwiseEvaluation]) -> dict[str, Any]:
        def serialise_issue(issue: AuditIssue) -> dict[str, Any]:
            return issue.model_dump()

        def serialise(candidate: CandidateEvaluation) -> dict[str, Any]:
            code_audit = asdict(candidate.code_audit)
            code_audit["issues"] = [serialise_issue(issue) for issue in candidate.code_audit.issues]
            cross_lingual_audit = asdict(candidate.cross_lingual_audit)
            cross_lingual_audit["issues"] = [
                serialise_issue(issue) for issue in candidate.cross_lingual_audit.issues
            ]
            return {
                "label": candidate.candidate.label,
                "model_name": candidate.candidate.model_name,
                "score": candidate.score.model_dump(),
                "weighted_total": candidate.total,
                "code_audit": code_audit,
                "cross_lingual_audit": cross_lingual_audit,
            }

        return {
            "schema_version": "1.0",
            "evaluation_count": len(evaluations),
            "evaluations": [
                {
                    "prompt_id": item.prompt.prompt_id,
                    "language": item.prompt.language,
                    "prompt": item.prompt.text,
                    "chosen": item.chosen_label,
                    "rejected": item.rejected_label,
                    "candidates": [serialise(item.candidate_a), serialise(item.candidate_b)],
                }
                for item in evaluations
            ],
        }

    @staticmethod
    def _markdown_report(evaluations: list[PairwiseEvaluation]) -> str:
        lines = [
            "# Lingua Evaluation Report",
            "",
            f"Evaluations: {len(evaluations)}",
            "",
            "| Prompt | Language | Chosen | A score | B score |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in evaluations:
            lines.append(
                f"| {item.prompt.prompt_id} | {item.prompt.language} | {item.chosen_label} | "
                f"{item.candidate_a.total:.3f} | {item.candidate_b.total:.3f} |"
            )
        lines.append("")
        return "\n".join(lines)
