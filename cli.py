"""Command-line entry point for the sample evaluation batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lingua_eval.rlhf_pipeline import PairwiseEvaluation, RLHFRankingPipeline


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object.")
        rows.append(value)
    return rows


def _summary(evaluations: list[PairwiseEvaluation]) -> str:
    lines = [
        "Prompt       Lang  Chosen  A score  B score  Code A  Code B",
        "------------  ----  ------  -------  -------  ------  ------",
    ]
    for item in evaluations:
        lines.append(
            f"{item.prompt.prompt_id:<12}  {item.prompt.language:<4}  {item.chosen_label:^6}  "
            f"{item.candidate_a.total:>7.3f}  {item.candidate_b.total:>7.3f}  "
            f"{str(item.candidate_a.code_audit.passed):<6}  {str(item.candidate_b.code_audit.passed):<6}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multilingual response pairs.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL batch.")
    parser.add_argument("--out", required=True, type=Path, help="JSON report path.")
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--dpo-out", type=Path, help="Optional DPO/RLHF JSONL path.")
    args = parser.parse_args()

    pipeline = RLHFRankingPipeline()
    evaluations = pipeline.evaluate_batch(_load_jsonl(args.input))
    markdown_path = args.markdown_out or args.out.with_suffix(".md")
    dpo_path = args.dpo_out or args.out.with_name(f"{args.out.stem}.dpo.jsonl")
    pipeline.write_reports(evaluations, args.out, markdown_path, dpo_path)
    print(_summary(evaluations))
    print(f"\nWrote JSON report: {args.out}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(f"Wrote DPO JSONL: {dpo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
