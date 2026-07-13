from types import SimpleNamespace

from genie_voice.assist.genie_facts import genie_account_insight
from genie_voice.enrich.engine import enrich_utterance
from genie_voice.enrich.fm import agent_reply_system_prompt
from genie_voice.genie.client import GenieClient
from genie_voice.i18n import (
    asr_model_language,
    canonical_business_context_instruction,
    content_language,
    generated_text_language_check,
    is_zh_asr_compare_language,
    localized_reply_opener,
    normalize_language,
    prose_for_language_check,
    sanitize_generated_display_text,
    stt_options_for_language,
)


def test_normalize_language_aliases_and_rejects_unknown():
    assert normalize_language(None) == "en-US"
    assert normalize_language("thai") == "th-TH"
    assert normalize_language("id") == "id-ID"
    assert normalize_language("mandarin") == "zh-CN"
    assert normalize_language("sensevoice") == "zh-CN-sensevoice"
    assert normalize_language("paraformer") == "zh-CN-paraformer"

    try:
        normalize_language("fr-FR")
    except ValueError as exc:
        assert "Unsupported language" in str(exc)
    else:
        raise AssertionError("expected unsupported language to raise")


def test_content_language_maps_zh_variants_to_zh_cn():
    assert content_language("zh-CN-sensevoice") == "zh-CN"
    assert content_language("zh-CN-paraformer") == "zh-CN"
    assert content_language("th-TH") == "th-TH"


def test_asr_model_language_maps_zh_variants_to_zh():
    assert asr_model_language("zh-CN-sensevoice") == "zh"
    assert asr_model_language("en-US") == "en-US"


def test_is_zh_asr_compare_language():
    assert is_zh_asr_compare_language("zh-CN-paraformer") is True
    assert is_zh_asr_compare_language("th-TH") is False


def test_stt_options_for_language_overlays_route():
    settings = SimpleNamespace(
        providers=SimpleNamespace(
            stt=SimpleNamespace(
                active_options=lambda: {
                    "endpoint": "voice_asr_en_finetuned_whisper_lora",
                    "postprocess_invoice_ids": True,
                    "routes": {
                        "th-TH": {
                            "endpoint": "voice_asr_th_oss_pathumma_whisper_large_v3",
                        }
                    },
                }
            )
        )
    )

    english = stt_options_for_language(settings, "en-US")
    thai = stt_options_for_language(settings, "th-TH")

    assert english["endpoint"] == "voice_asr_en_finetuned_whisper_lora"
    assert english["language"] == "en-US"
    assert thai["endpoint"] == "voice_asr_th_oss_pathumma_whisper_large_v3"
    assert thai["postprocess_invoice_ids"] is True
    assert thai["language"] == "th-TH"

    sensevoice = stt_options_for_language(
        SimpleNamespace(
            providers=SimpleNamespace(
                stt=SimpleNamespace(
                    active_options=lambda: {
                        "endpoint": "voice_asr_en_finetuned_whisper_lora",
                        "routes": {
                            "zh-CN-sensevoice": {"endpoint": "voice_asr_zh_oss_sensevoice_small"},
                        },
                    }
                )
            )
        ),
        "zh-CN-sensevoice",
    )
    assert sensevoice["endpoint"] == "voice_asr_zh_oss_sensevoice_small"
    assert sensevoice["language"] == "zh-CN-sensevoice"


def test_genie_question_is_wrapped_for_non_english_only():
    original = "How many overdue invoices does CUST-4028 have?"

    assert GenieClient._canonical_question(original, "en-US") == original

    wrapped = GenieClient._canonical_question(original, "th-TH")
    assert original in wrapped
    assert "canonical English/US schema" in wrapped
    assert "USD" in wrapped
    assert "Thai" in wrapped


def test_canonical_business_context_explicitly_blocks_local_currency_inference():
    instruction = canonical_business_context_instruction("zh-CN")

    assert "invoice IDs look like INV-90003" in instruction
    assert "money amounts are USD" in instruction
    assert "baht" in instruction
    assert "rupiah" in instruction
    assert "yuan" in instruction


def test_generated_text_language_check_catches_obvious_script_mismatch():
    thai = generated_text_language_check("ลูกค้าต้องการตรวจสอบ INV-90003 จำนวน $239", "th-TH")
    english_for_thai = generated_text_language_check("The customer has one overdue invoice.", "th-TH")
    chinese = generated_text_language_check("客户有一张逾期发票 INV-90003。", "zh-CN")

    assert thai["checked"] is True
    assert thai["matches"] is True
    assert english_for_thai["checked"] is True
    assert english_for_thai["matches"] is False
    assert chinese["checked"] is True
    assert chinese["matches"] is True


def test_prose_for_language_check_strips_canonical_business_tokens():
    stripped = prose_for_language_check("好的，已为 INV-90114 处理 $239.00。")
    assert "INV" not in stripped
    assert "$" not in stripped
    assert "好的" in stripped


def test_agent_reply_system_prompt_includes_target_language():
    prompt = agent_reply_system_prompt("zh-CN")
    assert "zh-CN" in prompt
    assert "Simplified Chinese" in prompt


def test_localized_reply_openers_are_not_english_for_non_english():
    assert localized_reply_opener("th-TH", genie_insight=True).startswith("จากข้อมูล")
    assert localized_reply_opener("id-ID", genie_insight=False).startswith("Berdasarkan")
    assert localized_reply_opener("zh-CN", genie_insight=True).startswith("根据")
    assert localized_reply_opener("zh-CN-sensevoice", genie_insight=True).startswith("根据")
    assert localized_reply_opener("en-US", genie_insight=False).startswith("Based on")


def test_sanitize_generated_display_text_removes_internal_schema_leaks():
    thai = "ลูกค้า CUST-4028 มีใบแจ้งหนี้ค้างชำระ 1 ใบ (อ้างอิงจากคอลัมน์ overdue_invoice_count ในตาราง invoices)"
    indonesian = "Ada 1 invoice jatuh tempo (berdasarkan kolom invoices.status = 'overdue')."
    chinese = "根据 `invoices.customer_id = 'CUST-4028'` 且 `invoices.status = 'overdue'` 的查询结果，CUST-4028 有 1 张逾期发票。"

    assert "ตาราง" not in sanitize_generated_display_text(thai)
    assert "invoices." not in sanitize_generated_display_text(indonesian)
    assert sanitize_generated_display_text(chinese).startswith("CUST-4028")
    assert "customer_id" not in sanitize_generated_display_text("Untuk customer_id CUST-4028, ada 1 invoice.")


def test_genie_account_insight_passes_language_to_genie_client():
    class FakeGenie:
        def __init__(self):
            self.language = None

        def ask(self, question, conversation_id=None, language=None):
            self.language = language
            return {
                "answer": "ลูกค้ามีใบแจ้งหนี้ค้างชำระ 1 ใบ รวม $239.",
                "rows": [[1]],
            }

    fake = FakeGenie()
    result = genie_account_insight(fake, "CUST-4028", language="th-TH")

    assert result
    assert fake.language == "th-TH"


def test_enrich_unavailable_preserves_language(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fm unavailable")

    import genie_voice.enrich.fm as fm

    monkeypatch.setattr(fm, "fm_enrich_customer_utterance", boom)
    result = enrich_utterance("ช่วยตรวจสอบ invoice INV-90003", speaker=1, language="th-TH")

    assert result["available"] is False
    assert result["language"] == "th-TH"
    assert result["speaker"] == 1
