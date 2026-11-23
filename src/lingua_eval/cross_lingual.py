"""Deterministic checks for multilingual response fidelity."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .models import AuditIssue, EvaluationPrompt

_TOKEN_RE = re.compile(
    r"(?:[$€£¥]\s?\d[\d\s,]*(?:\.\d+)?)|(?:\b\d+(?:[.,]\d+)?\s?%)|(?:\b\d+(?:[.,]\d+)?\b)|(?:`[^`]+`)|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b"
)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿА-Яа-яІіЇїЄєҐґ']+")

_STOPWORDS = {
    "en": {"the", "and", "is", "are", "this", "that", "with", "from", "for"},
    "fr": {"le", "la", "les", "et", "est", "sont", "ce", "cette", "avec", "pour"},
    "uk": {"це", "і", "та", "є", "для", "з", "у", "на", "що", "як"},
    "ru": {"это", "и", "есть", "для", "с", "на", "что", "как", "в"},
}
_TECHNICAL_ALLOWLIST = {
    "api", "backend", "bug", "callback", "class", "code", "cpu", "debug", "docker",
    "endpoint", "error", "framework", "http", "json", "library", "model", "python",
    "sql", "token", "url", "yaml",
}
_FALSE_FRIENDS = {
    "eventually": "éventuellement",
    "actually": "actuellement",
    "library": "librairie",
    "sensible": "sensible",
}
_EXPECTED_TECHNICAL = {
    "eventually": {"finalement", "à terme"},
    "actually": {"en réalité", "en fait"},
    "library": {"bibliothèque"},
}


@dataclass(slots=True)
class CrossLingualAudit:
    """Findings for one response against one evaluation prompt."""

    issues: list[AuditIssue]
    preserved: bool
    language_consistent: bool
    terminology_consistent: bool


class CrossLingualAuditor:
    """Run token preservation, language leakage, and terminology checks."""

    def audit(self, prompt: EvaluationPrompt, response: str) -> CrossLingualAudit:
        """Return deterministic findings for a prompt/response pair."""
        issues = self.check_numeric_entity_preservation(prompt.text, response, prompt.expected_entities)
        issues.extend(self.check_language_consistency(response, prompt.language))
        issues.extend(self.validate_terminology(prompt.text, response, prompt.language, prompt.terminology))
        return CrossLingualAudit(
            issues=issues,
            preserved=not any(issue.category == "numeric_entity" for issue in issues),
            language_consistent=not any(issue.category == "language_leakage" for issue in issues),
            terminology_consistent=not any(issue.category == "terminology" for issue in issues),
        )

    def check_numeric_entity_preservation(
        self,
        source: str,
        response: str,
        expected_entities: list[str] | None = None,
    ) -> list[AuditIssue]:
        """Check numbers, money, percentages, identifiers, and named entities."""
        issues: list[AuditIssue] = []
        source_tokens = Counter(self._normalise_token(token) for token in _TOKEN_RE.findall(source))
        response_tokens = Counter(self._normalise_token(token) for token in _TOKEN_RE.findall(response))
        for token, count in source_tokens.items():
            if response_tokens[token] < count:
                issues.append(
                    AuditIssue(
                        severity="error",
                        category="numeric_entity",
                        message=f"Source token '{token}' is missing or altered in the response.",
                        evidence=token,
                    )
                )
        for token, count in response_tokens.items():
            if response_tokens[token] - source_tokens[token] > 0:
                issues.append(
                    AuditIssue(
                        severity="warning",
                        category="numeric_entity",
                        message=f"Response introduces token '{token}' not present in the source.",
                        evidence=token,
                    )
                )
        for entity in expected_entities or []:
            if entity.casefold() not in response.casefold():
                issues.append(
                    AuditIssue(
                        severity="error",
                        category="numeric_entity",
                        message=f"Expected entity '{entity}' is missing from the response.",
                        evidence=entity,
                    )
                )
        return issues

    def check_language_consistency(self, response: str, language: str) -> list[AuditIssue]:
        """Detect repeated stopwords from another supported language."""
        words = {word.casefold() for word in _WORD_RE.findall(response)}
        own = _STOPWORDS[language]
        findings: list[AuditIssue] = []
        for other_language, stopwords in _STOPWORDS.items():
            if other_language == language:
                continue
            leaked = sorted(words & stopwords)
            if len(leaked) >= 2:
                findings.append(
                    AuditIssue(
                        severity="warning",
                        category="language_leakage",
                        message=f"Response contains {other_language.upper()} function-word leakage.",
                        evidence=", ".join(leaked),
                    )
                )
        if language == "fr":
            english_words = [word for word in words if word in _STOPWORDS["en"] and word not in _TECHNICAL_ALLOWLIST]
            if len(english_words) >= 2 and not (words & own):
                findings.append(
                    AuditIssue(
                        severity="warning",
                        category="language_leakage",
                        message="French response has insufficient French function words.",
                        evidence=", ".join(sorted(english_words)),
                    )
                )
        return findings

    def validate_terminology(
        self,
        source: str,
        response: str,
        language: str = "fr",
        glossary: dict[str, str] | None = None,
    ) -> list[AuditIssue]:
        """Check English technical terms and common French false friends."""
        if language != "fr":
            return []
        source_lower = source.casefold()
        response_lower = response.casefold()
        issues: list[AuditIssue] = []
        active = {**_EXPECTED_TECHNICAL, **({key: {value} for key, value in (glossary or {}).items()})}
        for english_term, translations in active.items():
            if english_term in source_lower and not any(translation.casefold() in response_lower for translation in translations):
                issues.append(
                    AuditIssue(
                        severity="warning",
                        category="terminology",
                        message=f"Technical term '{english_term}' has no accepted French rendering.",
                        evidence=", ".join(sorted(translations)),
                    )
                )
        for english_term, false_friend in _FALSE_FRIENDS.items():
            if english_term in source_lower and false_friend in response_lower:
                accepted = ", ".join(sorted(_EXPECTED_TECHNICAL.get(english_term, {false_friend})))
                if false_friend not in {item.casefold() for item in _EXPECTED_TECHNICAL.get(english_term, set())}:
                    issues.append(
                        AuditIssue(
                            severity="error",
                            category="terminology",
                            message=f"French false friend '{false_friend}' used for '{english_term}'.",
                            evidence=f"Prefer: {accepted}",
                        )
                    )
        return issues

    @staticmethod
    def _normalise_token(token: str) -> str:
        return re.sub(r"\s+", "", token.casefold()).replace(",", ".")
