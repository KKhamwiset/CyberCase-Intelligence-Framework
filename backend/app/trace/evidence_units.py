from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from app.sources.bundle import CaseSourceItem
from app.trace.quotes import MAX_QUOTE_CHARS
from app.trace.sentences import sentence_spans

MAX_EVIDENCE_UNITS_PER_CITATION = 8
UNIT_REFERENCE = re.compile(r"(?P<source>.+):U-(?P<number>\d+)")

UnitRejection = Literal["malformed", "unknown_source", "wrong_source", "unknown_unit"]


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    source_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class UnitResolution:
    units: tuple[EvidenceUnit, ...]
    rejected: tuple[tuple[str, UnitRejection], ...]


def unit_id_of(source_id: str, number: int) -> str:
    return f"{source_id}:U-{number:04d}"


def bounded_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    while end - start > MAX_QUOTE_CHARS:
        window = text[start : start + MAX_QUOTE_CHARS]
        cut = max(window.rfind(" "), window.rfind("\t"))
        cut = start + cut if cut > 0 else start + MAX_QUOTE_CHARS
        spans.append((start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if start < end:
        spans.append((start, end))
    return spans


@lru_cache(maxsize=128)
def unit_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, end)
        for sentence_start, sentence_end in sentence_spans(text)
        for start, end in bounded_spans(text, sentence_start, sentence_end)
        if text[start:end].strip()
    )


def segment_source(source_id: str, text: str) -> tuple[EvidenceUnit, ...]:
    return tuple(
        EvidenceUnit(
            unit_id=unit_id_of(source_id, number),
            source_id=source_id,
            start=start,
            end=end,
            text=text[start:end],
        )
        for number, (start, end) in enumerate(unit_spans(text), start=1)
    )


class EvidenceUnitRegistry:
    def __init__(self, sources: Mapping[str, CaseSourceItem]) -> None:
        self.sources = sources
        self.segmented: dict[str, tuple[EvidenceUnit, ...]] = {}

    def units(self, source_id: str) -> tuple[EvidenceUnit, ...]:
        if source_id not in self.segmented:
            self.segmented[source_id] = segment_source(source_id, self.sources[source_id].text)
        return self.segmented[source_id]

    def reject_or_find(self, source_id: str, reference: str) -> EvidenceUnit | UnitRejection:
        parsed = UNIT_REFERENCE.fullmatch(reference.strip())
        if parsed is None:
            return "malformed"
        owner = parsed["source"]
        if source_id not in self.sources or owner not in self.sources:
            return "unknown_source"
        if owner != source_id:
            return "wrong_source"
        units = self.units(source_id)
        number = int(parsed["number"])
        if not 1 <= number <= len(units):
            return "unknown_unit"
        return units[number - 1]

    def resolve(self, source_id: str, references: Sequence[str]) -> UnitResolution:
        found: list[EvidenceUnit] = []
        rejected: list[tuple[str, UnitRejection]] = []
        for reference in references:
            outcome = self.reject_or_find(source_id, reference)
            if isinstance(outcome, EvidenceUnit):
                if outcome not in found:
                    found.append(outcome)
            elif (reference, outcome) not in rejected:
                rejected.append((reference, outcome))
        return UnitResolution(units=tuple(found), rejected=tuple(rejected))


__all__ = [
    "MAX_EVIDENCE_UNITS_PER_CITATION",
    "EvidenceUnit",
    "EvidenceUnitRegistry",
    "UnitRejection",
    "UnitResolution",
    "segment_source",
    "unit_id_of",
    "unit_spans",
]
