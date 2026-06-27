from genie_voice.asr_eval.manifest import ExpectedEntities
from genie_voice.asr_eval.metrics import score_transcript


def test_entity_scoring_matches_observed_billing_variants():
    score = score_transcript(
        "My autopay failed, I want the fee waived on June thirtieth.",
        "My auto pay failed. I want the fee waived on June 30th.",
        ExpectedEntities(
            account_terms=["autopay"],
            billing_actions=["waive"],
            dates=["June thirtieth"],
        ),
    )

    assert score.entity_scores["account_terms"].missing == []
    assert score.entity_scores["billing_actions"].missing == []
    assert score.entity_scores["dates"].missing == []


def test_invoice_id_scoring_allows_spelled_prefix_but_keeps_digits_strict():
    matched = score_transcript(
        "Please check invoice INV-90022.",
        "Please check invoice I n v 90022.",
        ExpectedEntities(invoice_ids=["INV-90022"]),
    )
    missing_zero = score_transcript(
        "Please check invoice INV-90022.",
        "Please check invoice I NV9022.",
        ExpectedEntities(invoice_ids=["INV-90022"]),
    )

    assert matched.entity_scores["invoice_ids"].missing == []
    assert missing_zero.entity_scores["invoice_ids"].missing == ["INV-90022"]


def test_payment_account_term_allows_plural():
    score = score_transcript(
        "I can split it into two payments.",
        "I can split it into two payments.",
        ExpectedEntities(account_terms=["payment"]),
    )

    assert score.entity_scores["account_terms"].missing == []
