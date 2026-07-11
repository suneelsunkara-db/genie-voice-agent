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


def test_api_postprocess_returns_canonical_metadata(monkeypatch):
    from api.app import asr_postprocess

    monkeypatch.setattr(asr_postprocess, "_candidate_invoice_ids", lambda call_id: ["INV-90022"])
    settings = type(
        "Settings",
        (),
        {
            "providers": type(
                "Providers",
                (),
                {
                    "stt": type(
                        "Stt",
                        (),
                        {"active_options": lambda self: {"postprocess_invoice_ids": True}},
                    )()
                },
            )()
        },
    )()

    transcript, meta = asr_postprocess.postprocess_transcript_for_call(
        "CALL-1",
        "Please check invoice I NV9022.",
        settings,
        language="id-ID",
    )

    assert transcript == "Please check invoice INV-90022."
    assert meta["language"] == "id-ID"
    assert meta["canonical_transcript"] == "Please check invoice INV-90022."
    assert meta["normalized_entities"]["invoice_ids"] == ["INV-90022"]
