from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from app.analysis.prompts import CASE_READING_EVIDENCE_PROMPT, CASE_READING_JSON_PROMPT
from app.analysis.write import reading_request, write_trace
from app.llm.settings import AnalysisPipelineConfig
from app.sources.bundle import CaseSourceBundle, CaseSourceItem
from app.trace.bind import bound_claims, bound_references, resolve_claim
from app.trace.claims import (
    MAX_EVIDENCE_UNITS_PER_CITATION,
    CaseAnalysisClaim,
    CaseFollowupExchange,
    CaseProviderCitation,
    CaseSourceCitation,
)
from app.trace.evidence_units import EvidenceUnitRegistry, segment_source
from app.trace.trace import (
    CaseProviderJudgement,
    CaseProviderReading,
    CaseProviderReadingReply,
)

FIRST = "The finance server was encrypted overnight by an unknown actor."
SECOND = "Backups on the same network were deleted before the note appeared."
THIRD = "The ransom note asked for two bitcoin within three days."
TEXT = f"{FIRST}\n{SECOND}\n\n{THIRD}"
THAI = (
    "เซิร์ฟเวอร์ของฝ่ายการเงินถูกเข้ารหัสข้อมูลในช่วงกลางคืนโดยผู้ไม่ประสงค์ดี\n"
    "ข้อมูลสำรองบนเครือข่ายเดียวกันถูกลบก่อนที่ข้อความเรียกค่าไถ่จะปรากฏขึ้น"
)
OTHER = "A second statement was filed with the police on Monday morning."


def bundle() -> CaseSourceBundle:
    return CaseSourceBundle(
        revision=1,
        sources=(
            CaseSourceItem(source_id="S1", source_kind="narrative", text=TEXT),
            CaseSourceItem(source_id="S2", source_kind="narrative", text=OTHER),
            CaseSourceItem(source_id="S3", source_kind="narrative", text=THAI),
        ),
    )


def claim_citing(*citations: CaseSourceCitation, status="reported") -> CaseAnalysisClaim:
    return CaseAnalysisClaim(
        claim_id="A-01",
        claim_type="reported",
        text="The share was encrypted.",
        epistemic_status=status,
        supporting_source_ids=sorted({c.source_id for c in citations}),
        supporting_citations=list(citations),
    )


def cite(source_id: str, *unit_ids: str, quote: str = "") -> CaseSourceCitation:
    return CaseSourceCitation(
        source_id=source_id, exact_quote=quote, evidence_unit_ids=list(unit_ids)
    )


def reading_of(*claims: CaseAnalysisClaim) -> CaseProviderReading:
    return CaseProviderReading(
        version="case_analysis_trace_v1",
        claims=list(claims),
        involved_parties=[],
        timeline=[],
        impacts=[],
    )


def bound(*claims: CaseAnalysisClaim, mode="evidence_unit", history=()):
    reading, grounding = bound_claims(reading_of(*claims), bundle(), history, citation_mode=mode)
    return reading.claims, grounding


def test_a_valid_unit_id_becomes_the_exact_source_text_with_its_offsets():
    [claim], grounding = bound(claim_citing(cite("S1", "S1:U-0002")))

    [citation] = claim.supporting_citations
    assert citation.exact_quote == TEXT[citation.start : citation.end]
    assert citation.exact_quote == SECOND
    assert citation.evidence_unit_ids == ["S1:U-0002"]
    assert citation.pointer == "direct"
    assert citation.context is not None
    assert claim.epistemic_status == "reported"
    assert claim.unverified_citations == []
    assert grounding.citations_verified == 1


def test_no_model_written_quote_is_needed_or_trusted_when_a_unit_resolves():
    [claim], _ = bound(claim_citing(cite("S1", "S1:U-0003", quote="something the model made up")))

    [citation] = claim.supporting_citations
    assert citation.exact_quote == THIRD
    assert citation.pointer == "direct"
    assert claim.unverified_citations == []


def test_one_claim_can_cite_several_units():
    [claim], grounding = bound(claim_citing(cite("S1", "S1:U-0001", "S1:U-0003")))

    assert [c.exact_quote for c in claim.supporting_citations] == [FIRST, THIRD]
    assert [c.pointer for c in claim.supporting_citations] == ["direct", "direct"]
    assert grounding.evidence_ids_claimed == 2
    assert grounding.evidence_ids_resolved == 2
    assert grounding.evidence_ids_invalid == 0
    assert grounding.claims_with_direct_evidence == 1


def test_thai_text_resolves_by_unit_id():
    [unit_one, unit_two] = segment_source("S3", THAI)

    [claim], _ = bound(claim_citing(cite("S3", unit_two.unit_id)))

    [citation] = claim.supporting_citations
    assert citation.exact_quote == unit_two.text == THAI[unit_two.start : unit_two.end]
    assert unit_one.text not in citation.exact_quote


def test_a_malformed_id_is_unresolved_and_never_becomes_evidence():
    [claim], grounding = bound(claim_citing(cite("S1", "U2", "banana", "S1:U-")))

    assert claim.supporting_citations == []
    assert claim.epistemic_status == "not_confirmed"
    assert {(u.evidence_unit_ids[0], u.reason, u.pointer) for u in claim.unverified_citations} == {
        ("U2", "malformed", "unresolved"),
        ("banana", "malformed", "unresolved"),
        ("S1:U-", "malformed", "unresolved"),
    }
    assert grounding.evidence_ids_claimed == 3
    assert grounding.evidence_ids_invalid == 3
    assert grounding.citations_unfound == 1
    assert grounding.claims_with_direct_evidence == 0


def test_a_unit_that_does_not_exist_is_unresolved():
    [claim], grounding = bound(claim_citing(cite("S1", "S1:U-0004", "S1:U-0000")))

    assert claim.supporting_citations == []
    assert [u.reason for u in claim.unverified_citations] == ["unknown_unit", "unknown_unit"]
    assert grounding.evidence_ids_invalid == 2


def test_a_unit_of_another_source_is_unresolved_not_borrowed():
    [claim], grounding = bound(claim_citing(cite("S1", "S2:U-0001")))

    assert claim.supporting_citations == []
    [record] = claim.unverified_citations
    assert (record.reason, record.evidence_unit_ids, record.source_id) == (
        "wrong_source",
        ["S2:U-0001"],
        "S1",
    )
    assert claim.epistemic_status == "not_confirmed"
    assert grounding.evidence_ids_invalid == 1


def test_an_unknown_source_is_unresolved():
    [claim], _ = bound(claim_citing(cite("S9", "S9:U-0001")))
    [other], _ = bound(claim_citing(cite("S1", "S9:U-0001")))

    assert claim.supporting_citations == []
    assert [u.reason for u in claim.unverified_citations] == ["unknown_source"]
    assert [u.reason for u in other.unverified_citations] == ["unknown_source"]


def test_duplicate_and_zero_padded_ids_name_one_unit():
    [claim], grounding = bound(claim_citing(cite("S1", "S1:U-0001", "S1:U-0001", "S1:U-1")))

    assert [c.exact_quote for c in claim.supporting_citations] == [FIRST]
    assert grounding.evidence_ids_claimed == 1


def test_the_same_unit_cited_twice_by_one_claim_is_kept_once():
    [claim], grounding = bound(claim_citing(cite("S1", "S1:U-0001"), cite("S1", "S1:U-0001")))

    assert len(claim.supporting_citations) == 1
    assert grounding.citations_claimed == 2
    assert grounding.citations_verified == 1
    assert grounding.citations_duplicated == 1


def test_one_bad_id_does_not_discard_the_good_one_beside_it():
    [claim], grounding = bound(claim_citing(cite("S1", "S1:U-0001", "S1:U-0042")))

    assert [c.exact_quote for c in claim.supporting_citations] == [FIRST]
    assert [u.reason for u in claim.unverified_citations] == ["unknown_unit"]
    assert claim.epistemic_status == "reported"
    assert (grounding.evidence_ids_resolved, grounding.evidence_ids_invalid) == (1, 1)


def test_an_invalid_id_beside_a_locatable_quote_is_recovered_not_direct():
    quote = "encrypted overnight"
    [claim], _ = bound(claim_citing(cite("S1", "S1:U-0042", quote=quote)))

    [citation] = claim.supporting_citations
    assert citation.exact_quote == quote
    assert citation.pointer == "recovered"
    assert citation.evidence_unit_ids == []
    assert [u.reason for u in claim.unverified_citations] == ["unknown_unit"]
    assert claim.epistemic_status == "reported"


def test_a_quote_only_citation_is_recovered_in_evidence_mode_and_plain_in_legacy_mode():
    quoted = claim_citing(cite("S1", quote="encrypted overnight"))

    [evidence_claim], _ = bound(quoted)
    [legacy_claim], _ = bound(quoted, mode="legacy_quote")

    assert evidence_claim.supporting_citations[0].pointer == "recovered"
    assert legacy_claim.supporting_citations[0].pointer is None
    assert (
        evidence_claim.supporting_citations[0].exact_quote
        == legacy_claim.supporting_citations[0].exact_quote
    )


def test_a_quote_that_cannot_be_found_still_goes_the_legacy_unverified_way():
    [claim], grounding = bound(claim_citing(cite("S1", quote="never said anywhere in the source")))

    assert claim.supporting_citations == []
    assert claim.epistemic_status == "not_confirmed"
    assert [u.written_quote for u in claim.unverified_citations] == [
        "never said anywhere in the source"
    ]
    assert grounding.citations_unfound == 1


def test_resolving_an_already_resolved_claim_changes_nothing():
    [first], _ = bound(claim_citing(cite("S1", "S1:U-0001", "S1:U-0099")))
    [second], _ = bound(first)

    assert second.supporting_citations == first.supporting_citations
    assert second.unverified_citations == first.unverified_citations


def test_a_followup_answer_is_cited_by_its_own_units():
    history = (
        CaseFollowupExchange(
            qa_id="QA-01", gap_key="how_much", question="How much?", answer="Two bitcoin."
        ),
    )

    [claim], _ = bound(claim_citing(cite("QA-01", "QA-01:U-0001")), history=history)

    [citation] = claim.supporting_citations
    assert (citation.source_id, citation.exact_quote) == ("QA-01", "Two bitcoin.")


def test_a_direct_citation_carries_the_document_page_of_its_unit():
    unit = segment_source("S1", TEXT)[1]
    document = CaseSourceItem(
        source_id="S1",
        source_kind="document",
        text=TEXT,
        document_id="d1",
        filename="report.pdf",
        provenance={
            "pages": [
                {"page_number": 1, "start_offset": 0, "end_offset": unit.start},
                {"page_number": 2, "start_offset": unit.start, "end_offset": len(TEXT)},
            ]
        },
    )
    source_bundle = CaseSourceBundle(revision=1, sources=(document,))

    reading, _ = bound_claims(
        reading_of(claim_citing(cite("S1", unit.unit_id))),
        source_bundle,
        citation_mode="evidence_unit",
    )

    [citation] = reading.claims[0].supporting_citations
    assert (citation.document_id, citation.filename, citation.page_numbers) == (
        "d1",
        "report.pdf",
        [2],
    )


def test_resolve_claim_alone_resolves_ids_without_a_search_or_registry_passed_in():
    registry = {source.source_id: source for source in bundle().sources}

    claim = resolve_claim(claim_citing(cite("S1", "S1:U-0001")), registry, None)

    assert claim.supporting_citations[0].exact_quote == FIRST


def test_the_resolution_rate_counts_every_cited_id():
    [_, _], grounding = bound(
        claim_citing(cite("S1", "S1:U-0001", "S1:U-0099", "banana")),
        claim_citing(cite("S1", "S1:U-0002")).model_copy(update={"claim_id": "A-02"}),
    )

    assert grounding.evidence_ids_claimed == 4
    assert grounding.evidence_ids_resolved == 2
    assert grounding.evidence_ids_resolved / grounding.evidence_ids_claimed == 0.5


def test_the_registry_says_why_it_rejects_an_id():
    registry = EvidenceUnitRegistry({s.source_id: s for s in bundle().sources})

    assert registry.reject_or_find("S1", "nonsense") == "malformed"
    assert registry.reject_or_find("S1", "S2:U-0001") == "wrong_source"
    assert registry.reject_or_find("S1", "S9:U-0001") == "unknown_source"
    assert registry.reject_or_find("S9", "S9:U-0001") == "unknown_source"
    assert registry.reject_or_find("S1", "S1:U-0500") == "unknown_unit"
    assert registry.reject_or_find("S1", " S1:U-0001 ").text == FIRST


def test_the_provider_accepts_ids_only_citations_and_several_ids():
    reply = CaseProviderReadingReply.model_validate(
        {
            "version": "case_analysis_trace_v1",
            "claims": [
                {
                    "claim_id": "A-01",
                    "claim_type": "reported",
                    "text": "The share was encrypted.",
                    "epistemic_status": "reported",
                    "supporting_source_ids": ["S1"],
                    "supporting_citations": [
                        {"source_id": "S1", "evidence_unit_ids": ["S1:U-0001", "S1:U-0002"]}
                    ],
                }
            ],
            "involved_parties": [],
            "timeline": [],
            "impacts": [],
        }
    )

    [citation] = reply.claims[0].supporting_citations
    assert citation.evidence_unit_ids == ["S1:U-0001", "S1:U-0002"]
    assert citation.exact_quote == ""


def test_the_provider_drops_an_empty_citation_and_bounds_the_ids():
    ids = [f"S1:U-{n:04d}" for n in range(1, 30)]
    reply = CaseProviderReadingReply.model_validate(
        {
            "version": "case_analysis_trace_v1",
            "claims": [
                {
                    "claim_id": "A-01",
                    "claim_type": "reported",
                    "text": "x",
                    "epistemic_status": "reported",
                    "supporting_citations": [
                        {"source_id": "S1"},
                        {"source_id": "S1", "evidence_unit_ids": []},
                        {"source_id": "S1", "evidence_unit_ids": ids},
                        {"source_id": "S1", "evidence_unit_ids": [None, 7, "  ", "S1:U-0001"]},
                    ],
                }
            ],
            "involved_parties": [],
            "timeline": [],
            "impacts": [],
        }
    )

    first, second = reply.claims[0].supporting_citations
    assert first.evidence_unit_ids == ids[:MAX_EVIDENCE_UNITS_PER_CITATION]
    assert second.evidence_unit_ids == ["7", "S1:U-0001"]


def test_a_provider_citation_with_neither_ids_nor_quote_is_refused():
    try:
        CaseProviderCitation(source_id="S1")
    except ValueError:
        return
    raise AssertionError("an empty citation was accepted")


def test_a_model_cannot_write_the_pointer_or_the_offsets_itself():
    reply = CaseProviderReadingReply.model_validate(
        {
            "version": "case_analysis_trace_v1",
            "claims": [
                {
                    "claim_id": "A-01",
                    "claim_type": "reported",
                    "text": "x",
                    "epistemic_status": "reported",
                    "supporting_citations": [
                        {
                            "source_id": "S1",
                            "evidence_unit_ids": ["S1:U-0099"],
                            "pointer": "direct",
                            "start": 0,
                            "end": 5,
                        }
                    ],
                }
            ],
            "involved_parties": [],
            "timeline": [],
            "impacts": [],
        }
    )
    claim = CaseAnalysisClaim.model_validate(reply.claims[0].model_dump())

    [resolved], _ = bound(claim)

    assert resolved.supporting_citations == []
    assert [u.reason for u in resolved.unverified_citations] == ["unknown_unit"]


def history() -> tuple[CaseFollowupExchange, ...]:
    return (
        CaseFollowupExchange(
            qa_id="QA-01", gap_key="how_much", question="How much?", answer="Two bitcoin."
        ),
    )


def test_the_reading_request_shows_addressable_units_and_no_whole_text():
    content = reading_request(bundle(), "english", history(), citation_mode="evidence_unit")

    first, _, thai = content["case_sources"]
    assert "text" not in first
    assert first["evidence_units"] == [
        {"unit_id": "S1:U-0001", "text": FIRST},
        {"unit_id": "S1:U-0002", "text": SECOND},
        {"unit_id": "S1:U-0003", "text": THIRD},
    ]
    assert [u["unit_id"] for u in thai["evidence_units"]] == ["S3:U-0001", "S3:U-0002"]
    assert content["followup_history"] == [
        {
            "qa_id": "QA-01",
            "gap_key": "how_much",
            "question": "How much?",
            "evidence_units": [{"unit_id": "QA-01:U-0001", "text": "Two bitcoin."}],
        }
    ]
    assert json.dumps(content).count("Two bitcoin.") == 1


def test_the_reading_request_in_legacy_mode_is_the_whole_text_as_before():
    content = reading_request(bundle(), "english", history())

    assert content["case_sources"][0]["text"] == TEXT
    assert "evidence_units" not in content["case_sources"][0]
    assert content["followup_history"][0]["answer"] == "Two bitcoin."


def test_the_source_text_is_never_changed_by_the_unit_labels():
    source_bundle = bundle()
    before = [source.text for source in source_bundle.sources]

    reading_request(source_bundle, "english", (), citation_mode="evidence_unit")

    assert [source.text for source in source_bundle.sources] == before


def run_write_trace(reply: dict, **options):
    seen: list[dict] = []

    async def request_stage(**kwargs):
        seen.append(kwargs)
        if kwargs["stage"] == "case_reading":
            return CaseProviderReadingReply.model_validate(reply)
        return CaseProviderJudgement(
            version="case_analysis_trace_v1", summary="The share was encrypted [A-01]."
        )

    with patch("app.analysis.write.request_stage", new=request_stage):
        trace = asyncio.run(
            write_trace(
                sources=bundle(), language="english", config=AnalysisPipelineConfig(), **options
            )
        )
    return trace, seen


def evidence_reply(*unit_ids: str) -> dict:
    return {
        "version": "case_analysis_trace_v1",
        "claims": [
            {
                "claim_id": "A-01",
                "claim_type": "reported",
                "text": "The share was encrypted.",
                "epistemic_status": "reported",
                "supporting_source_ids": ["S1"],
                "contradicting_source_ids": [],
                "supporting_citations": [{"source_id": "S1", "evidence_unit_ids": list(unit_ids)}],
                "contradicting_citations": [],
            }
        ],
        "involved_parties": [{"name": "Finance", "role": "victim", "claim_ids": ["A-01"]}],
        "timeline": [],
        "impacts": [],
    }


def test_reading_grounding_and_judgement_join_into_one_trace_by_unit_id():
    trace, (reading_call, judgement_call) = run_write_trace(
        evidence_reply("S1:U-0001"), citation_mode="evidence_unit"
    )

    assert reading_call["system"] == CASE_READING_EVIDENCE_PROMPT
    assert reading_call["grammar"] is False
    assert trace.summary == "The share was encrypted [A-01]."
    [citation] = trace.claims[0].supporting_citations
    assert citation.exact_quote == FIRST
    assert citation.pointer == "direct"
    assert bound_references(trace).involved_parties[0].support == "bound"
    assert trace.grounding.evidence_ids_resolved == 1
    sent = json.dumps(judgement_call["content"], ensure_ascii=False)
    assert FIRST in sent
    for hidden in ("evidence_unit_ids", "S1:U-0001", '"pointer"', '"start"', '"end"'):
        assert hidden not in sent


def test_the_setting_chooses_the_citation_mode_and_legacy_still_works():
    with patch("app.analysis.write.settings") as settings:
        settings.citation_mode = "legacy_quote"
        legacy_reply = evidence_reply()
        legacy_reply["claims"][0]["supporting_citations"] = [
            {"source_id": "S1", "exact_quote": "encrypted overnight"}
        ]
        trace, (reading_call, _) = run_write_trace(legacy_reply)

    assert reading_call["system"] == CASE_READING_JSON_PROMPT
    assert "text" in reading_call["content"]["case_sources"][0]
    [citation] = trace.claims[0].supporting_citations
    assert citation.exact_quote == "encrypted overnight"
    assert citation.pointer is None


def test_an_invalid_id_does_not_stop_the_analysis():
    trace, _ = run_write_trace(evidence_reply("S1:U-0077"), citation_mode="evidence_unit")

    assert trace.claims[0].epistemic_status == "not_confirmed"
    assert trace.grounding.evidence_ids_invalid == 1
    assert trace.summary
