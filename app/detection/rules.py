"""Data-driven rule engine for common Indian scam patterns.

Rules live in ``rules.yaml`` next to this module; adding a rule means adding a
YAML entry, not changing code. See the header of that file for the match syntax.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.detection.normalize import canonicalize, normalize_text

Language = Literal["en", "hi"]

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")


class RuleConfigError(ValueError):
    """Raised when a rules file is malformed or internally inconsistent."""


class LocalizedText(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    en: str = Field(min_length=1)
    hi: str = Field(min_length=1)

    def get(self, language: Language) -> str:
        return self.hi if language == "hi" else self.en


class MatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    any: list[str] = Field(default_factory=list)
    all: list[list[str]] = Field(default_factory=list)
    none: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_positive_condition(self) -> MatchSpec:
        if not self.any and not self.all:
            raise ValueError("match needs at least one of 'any' or 'all'")
        if any(not group for group in self.all):
            raise ValueError("groups inside 'all' must not be empty")
        return self


class RuleDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    category: str
    severity: float = Field(gt=0.0, le=1.0)
    reason: LocalizedText
    match: MatchSpec
    enabled: bool = True


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerates the `shared` anchor block

    categories: dict[str, LocalizedText]
    rules: list[RuleDefinition]

    @model_validator(mode="after")
    def _check_references(self) -> RuleSet:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)
            if rule.category not in self.categories:
                raise ValueError(f"rule {rule.id!r} uses unknown category {rule.category!r}")
        return self


@dataclass(frozen=True, slots=True)
class RuleHit:
    rule_id: str
    category: str
    severity: float
    reason_en: str
    reason_hi: str
    matched_text: str

    def reason(self, language: Language = "en") -> str:
        return self.reason_hi if language == "hi" else self.reason_en


def _compile(rule_id: str, pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(canonicalize(pattern), re.IGNORECASE)
    except re.error as exc:
        raise RuleConfigError(f"rule {rule_id}: invalid pattern {pattern!r}: {exc}") from exc


class CompiledRule:
    __slots__ = ("definition", "_any", "_all", "_none")

    def __init__(self, definition: RuleDefinition) -> None:
        self.definition = definition
        rid = definition.id
        self._any = [_compile(rid, p) for p in definition.match.any]
        self._all = [[_compile(rid, p) for p in group] for group in definition.match.all]
        self._none = [_compile(rid, p) for p in definition.match.none]

    @staticmethod
    def _first_match(patterns: Iterable[re.Pattern[str]], text: str) -> re.Match[str] | None:
        for pattern in patterns:
            found = pattern.search(text)
            if found:
                return found
        return None

    def match(self, normalized_text: str) -> str | None:
        """Return the matched snippet if the rule fires on already-normalised text."""
        snippets: list[str] = []

        if self._any:
            found = self._first_match(self._any, normalized_text)
            if found is None:
                return None
            snippets.append(found.group(0))

        for group in self._all:
            found = self._first_match(group, normalized_text)
            if found is None:
                return None
            snippets.append(found.group(0))

        if self._first_match(self._none, normalized_text):
            return None

        return " ... ".join(s.strip() for s in snippets)


class RuleEngine:
    def __init__(
        self,
        rules: Iterable[RuleDefinition],
        categories: Mapping[str, LocalizedText] | None = None,
    ) -> None:
        self._rules = [CompiledRule(rule) for rule in rules if rule.enabled]
        self.categories: dict[str, LocalizedText] = dict(categories or {})

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_RULES_PATH) -> RuleEngine:
        rule_set = load_rule_set(path)
        return cls(rule_set.rules, rule_set.categories)

    @property
    def rule_ids(self) -> list[str]:
        return [rule.definition.id for rule in self._rules]

    def category_label(self, category: str, language: Language = "en") -> str:
        label = self.categories.get(category)
        return label.get(language) if label else category

    def evaluate(self, text: str) -> list[RuleHit]:
        """Run every enabled rule and return hits, most severe first."""
        normalized = normalize_text(text)
        if not normalized:
            return []

        hits: list[RuleHit] = []
        for rule in self._rules:
            snippet = rule.match(normalized)
            if snippet is None:
                continue
            d = rule.definition
            hits.append(
                RuleHit(
                    rule_id=d.id,
                    category=d.category,
                    severity=d.severity,
                    reason_en=d.reason.en,
                    reason_hi=d.reason.hi,
                    matched_text=snippet,
                )
            )

        hits.sort(key=lambda hit: (-hit.severity, hit.rule_id))
        return hits


def load_rule_set(path: Path | str = DEFAULT_RULES_PATH) -> RuleSet:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuleConfigError(f"cannot read rules file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuleConfigError(f"rules file {path} must contain a mapping at the top level")

    try:
        return RuleSet.model_validate(raw)
    except ValidationError as exc:
        raise RuleConfigError(f"invalid rules file {path}:\n{exc}") from exc


@lru_cache(maxsize=4)
def get_rule_engine(path: Path | str = DEFAULT_RULES_PATH) -> RuleEngine:
    """Return a cached engine so patterns are compiled once per process."""
    return RuleEngine.from_file(path)
