from __future__ import annotations

from app.sources.bundle import CaseSourceItem
from app.trace.evidence_units import (
    EvidenceUnitRegistry,
    segment_source,
    unit_id_of,
)
from app.trace.quotes import MAX_QUOTE_CHARS

ENGLISH = (
    "The finance server was encrypted overnight by an unknown actor.\n"
    "Backups on the same network were deleted before the note appeared.\n"
    "\n"
    "The ransom note asked for two bitcoin within three days."
)
THAI = (
    "เซิร์ฟเวอร์ของฝ่ายการเงินถูกเข้ารหัสข้อมูลในช่วงกลางคืนโดยผู้ไม่ประสงค์ดี\n"
    "ข้อมูลสำรองบนเครือข่ายเดียวกันถูกลบก่อนที่ข้อความเรียกค่าไถ่จะปรากฏขึ้น"
)


def item(source_id: str, text: str) -> CaseSourceItem:
    return CaseSourceItem(source_id=source_id, source_kind="narrative", text=text)


def test_the_same_text_always_gets_the_same_ids():
    first = segment_source("SRC-01", ENGLISH)
    second = segment_source("SRC-01", ENGLISH)

    assert first == second
    assert [unit.unit_id for unit in first] == [
        "SRC-01:U-0001",
        "SRC-01:U-0002",
        "SRC-01:U-0003",
    ]


def test_offsets_reconstruct_the_exact_source_text_of_each_unit():
    for text in (ENGLISH, THAI):
        for unit in segment_source("SRC-01", text):
            assert unit.text == text[unit.start : unit.end]
            assert unit.text == unit.text.strip()
            assert unit.text


def test_thai_text_is_cut_into_addressable_units():
    units = segment_source("SRC-02", THAI)

    assert len(units) == 2
    assert units[0].text.startswith("เซิร์ฟเวอร์")
    assert units[1].text.endswith("ปรากฏขึ้น")


def test_units_are_in_source_order_and_never_overlap():
    units = segment_source("SRC-01", ENGLISH)

    assert all(a.end <= b.start for a, b in zip(units, units[1:], strict=False))


def test_blank_and_degenerate_text_gives_no_units():
    for text in ("", "   ", "\n\n \t\n"):
        assert segment_source("SRC-01", text) == ()


def test_a_very_short_text_is_one_unit():
    [unit] = segment_source("SRC-01", "  Hi.  ")

    assert unit.text == "Hi."
    assert (unit.start, unit.end) == (2, 5)


def test_a_line_longer_than_a_quote_is_cut_into_units_a_quote_can_hold():
    text = " ".join(["word"] * 1_500)

    units = segment_source("SRC-01", text)

    assert len(units) > 1
    assert all(len(unit.text) <= MAX_QUOTE_CHARS for unit in units)
    assert " ".join(unit.text for unit in units) == text


def test_an_unbroken_run_longer_than_a_quote_is_hard_cut():
    units = segment_source("SRC-01", "x" * (2 * MAX_QUOTE_CHARS + 10))

    assert [len(unit.text) for unit in units] == [MAX_QUOTE_CHARS, MAX_QUOTE_CHARS, 10]


def test_two_sources_never_share_an_id():
    registry = EvidenceUnitRegistry(
        {"SRC-01": item("SRC-01", ENGLISH), "SRC-02": item("SRC-02", ENGLISH)}
    )

    first = {unit.unit_id for unit in registry.units("SRC-01")}
    second = {unit.unit_id for unit in registry.units("SRC-02")}

    assert first.isdisjoint(second)
    assert all(unit.source_id == "SRC-02" for unit in registry.units("SRC-02"))


def test_an_id_is_the_source_id_then_a_four_digit_number():
    assert unit_id_of("SRC-01", 17) == "SRC-01:U-0017"
