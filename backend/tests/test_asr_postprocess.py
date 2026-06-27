from genie_voice.asr_eval.postprocess import normalize_invoice_ids


def test_normalize_invoice_ids_repairs_known_whisper_variants():
    transcript, corrections = normalize_invoice_ids(
        "Invoice I NV9022 and invoice i-NV10,480.2 are open.",
        ["INV-90022", "INV-10482"],
    )

    assert transcript == "Invoice INV-90022 and invoice INV-10482 are open."
    assert [correction.invoice_id for correction in corrections] == ["INV-90022", "INV-10482"]


def test_normalize_invoice_ids_skips_ambiguous_candidates():
    transcript, corrections = normalize_invoice_ids(
        "Invoice I NV9022 is open.",
        ["INV-90022", "INV-99022"],
    )

    assert transcript == "Invoice I NV9022 is open."
    assert corrections == []
