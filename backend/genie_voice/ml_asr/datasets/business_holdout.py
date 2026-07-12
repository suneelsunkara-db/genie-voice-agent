from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.config import DatasetSpec, EvalConfig
from genie_voice.ml_asr.manifest import empty_entities, write_manifest_jsonl


@dataclass(frozen=True)
class BusinessScenario:
    scenario: str
    reference_transcript: str
    expected_entities: dict[str, list[str]]


SCENARIO_ORDER = (
    "billing_dispute",
    "payment_lookup",
    "payment_confirmation",
    "charge_refusal",
    "account_balance",
    "refund_request",
    "payment_extension",
    "case_close",
)


def bootstrap_business_language(
    config: EvalConfig,
    dataset: DatasetSpec,
    *,
    language: LanguageCode,
    clips_per_scenario: int,
    volume_mode: bool = False,
) -> dict[str, Any]:
    lang_spec = dataset.languages[language]
    scenarios = _scenario_catalog(language)
    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    clip_index = 0

    for scenario_name in SCENARIO_ORDER:
        scenario_templates = scenarios[scenario_name]
        for variant in range(clips_per_scenario):
            template = scenario_templates[variant % len(scenario_templates)]
            materialized = _materialize_variant(template, variant=variant, clip_index=clip_index + 1, language=language)
            clip_index += 1
            clip_id = f"{lang_spec.manifest_language}_business_holdout_{clip_index:04d}"
            remote_audio = f"{lang_spec.remote_audio_dir}/{clip_id}.wav"
            local_audio = (
                f"{config.local_audio_dir}/{lang_spec.manifest_language}/business_holdout/{clip_id}.wav"
            )
            rows.append(
                {
                    "clip_id": clip_id,
                    "call_id": f"HOLDOUT-{lang_spec.manifest_language.upper()}-{clip_index:04d}",
                    "speaker": "customer",
                    "audio_path": remote_audio,
                    "audio_format": "audio/wav",
                    "sample_rate_hz": 16000,
                    "duration_seconds": None,
                    "domain": dataset.domain,
                    "scenario": scenario_name,
                    "split": dataset.split,
                    "dataset_version": dataset.dataset_id,
                    "language": lang_spec.manifest_language,
                    "reference_transcript": materialized.reference_transcript,
                    "expected_entities": materialized.expected_entities,
                    "metadata": {
                        "source": dataset.source,
                        "eval_tier": dataset.eval_tier,
                        "dataset_id": dataset.dataset_id,
                        "audio_mode": dataset.audio_mode,
                        "audio_status": "pending",
                        "human_transcript_approved": True,
                        "recording_channel": dataset.audio_mode,
                        "business_holdout": True,
                    },
                }
            )
            uploads.append(
                {
                    "local": local_audio,
                    "remote": remote_audio,
                    "status": dataset.audio_mode,
                }
            )

    manifest_path = lang_spec.remote_manifest_path if volume_mode else lang_spec.local_manifest_path
    write_manifest_jsonl(manifest_path, rows)
    return {
        "dataset_id": dataset.dataset_id,
        "language": language,
        "rows": len(rows),
        "scenarios": len(SCENARIO_ORDER),
        "clips_per_scenario": clips_per_scenario,
        "manifest": manifest_path,
        "remote_manifest": lang_spec.remote_manifest_path,
        "uploads": uploads if not volume_mode else [],
        "volume_mode": volume_mode,
    }


def _materialize_variant(
    template: BusinessScenario,
    *,
    variant: int,
    clip_index: int,
    language: LanguageCode,
) -> BusinessScenario:
    entities = {key: list(values) for key, values in template.expected_entities.items()}
    transcript = template.reference_transcript
    prefix = _invoice_prefix(language)

    if entities.get("invoice_ids"):
        current = entities["invoice_ids"][0]
        base_num = int("".join(ch for ch in current if ch.isdigit()) or str(90000 + clip_index))
        new_num = base_num + (variant * 17)
        new_invoice = f"{prefix}{new_num}"
        transcript = transcript.replace(current, new_invoice)
        entities["invoice_ids"] = [new_invoice]

    if entities.get("amounts"):
        current_amount = entities["amounts"][0]
        digits = "".join(ch for ch in current_amount if ch.isdigit())
        if digits:
            shifted = max(1, int(digits) + (variant * 7) + (clip_index % 5))
            replacement = current_amount.replace(digits, str(shifted), 1)
            transcript = transcript.replace(current_amount, replacement)
            entities["amounts"] = [replacement]

    return BusinessScenario(template.scenario, transcript, entities)


def _scenario_catalog(language: LanguageCode) -> dict[str, list[BusinessScenario]]:
    builders = {
        "en-US": _english_scenarios,
        "th-TH": _thai_scenarios,
        "id-ID": _indonesian_scenarios,
        "zh-CN": _chinese_scenarios,
    }
    builder = builders.get(language)
    if builder is None:
        raise ValueError(f"No business scenario catalog for {language}")
    return builder()


def _invoice_prefix(language: LanguageCode) -> str:
    return {
        "en-US": "INV-EN",
        "th-TH": "INV-TH",
        "id-ID": "INV-ID",
        "zh-CN": "INV-ZH",
    }[language]


def _english_scenarios() -> dict[str, list[BusinessScenario]]:
    prefix = "INV-EN"
    return {
        "billing_dispute": [
            BusinessScenario(
                "billing_dispute",
                f"Hello, I am calling to dispute invoice {prefix}90001 for 36 dollars because the amount is wrong.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90001"],
                    "amounts": ["36 dollars"],
                    "billing_actions": ["dispute"],
                    "account_terms": ["invoice"],
                },
            ),
            BusinessScenario(
                "billing_dispute",
                f"I need to dispute invoice {prefix}90009 for 124 dollars; the charge is incorrect.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90009"],
                    "amounts": ["124 dollars"],
                    "billing_actions": ["dispute"],
                    "account_terms": ["invoice", "charge"],
                },
            ),
        ],
        "payment_lookup": [
            BusinessScenario(
                "payment_lookup",
                f"Can you check whether invoice {prefix}90002 for 58 dollars was received?",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90002"],
                    "amounts": ["58 dollars"],
                    "billing_actions": ["payment"],
                    "account_terms": ["invoice"],
                },
            ),
        ],
        "payment_confirmation": [
            BusinessScenario(
                "payment_confirmation",
                f"I want to confirm that invoice {prefix}90003 has been paid.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90003"],
                    "billing_actions": ["payment"],
                    "confirmations": ["confirm"],
                    "account_terms": ["invoice"],
                },
            ),
        ],
        "charge_refusal": [
            BusinessScenario(
                "charge_refusal",
                f"No, do not charge invoice {prefix}90004 for 72 dollars to that card.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90004"],
                    "amounts": ["72 dollars"],
                    "billing_actions": ["charge"],
                    "refusals": ["no", "do not"],
                    "account_terms": ["card"],
                },
            ),
        ],
        "account_balance": [
            BusinessScenario(
                "account_balance",
                f"What is the remaining balance on invoice {prefix}90005?",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90005"],
                    "billing_actions": ["balance"],
                    "account_terms": ["balance", "invoice"],
                },
            ),
        ],
        "refund_request": [
            BusinessScenario(
                "refund_request",
                f"Please refund invoice {prefix}90006 for 45 dollars.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90006"],
                    "amounts": ["45 dollars"],
                    "billing_actions": ["refund"],
                    "account_terms": ["invoice"],
                },
            ),
        ],
        "payment_extension": [
            BusinessScenario(
                "payment_extension",
                f"I need an extension on invoice {prefix}90007 for 89 dollars.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90007"],
                    "amounts": ["89 dollars"],
                    "billing_actions": ["extension"],
                    "account_terms": ["invoice"],
                },
            ),
        ],
        "case_close": [
            BusinessScenario(
                "case_close",
                f"Yes, that resolves my issue for invoice {prefix}90008. Thank you.",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90008"],
                    "confirmations": ["yes"],
                    "account_terms": ["invoice"],
                },
            ),
        ],
    }


def _thai_scenarios() -> dict[str, list[BusinessScenario]]:
    prefix = "INV-TH"
    return {
        "billing_dispute": [
            BusinessScenario(
                "billing_dispute",
                f"สวัสดีค่ะ ฉันโทรมาโต้แย้งใบแจ้งหนี้ {prefix}90001 ยอด 36 ดอลลาร์ เพราะยอดไม่ถูกต้อง",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90001"],
                    "amounts": ["36 dollars"],
                    "billing_actions": ["dispute"],
                    "account_terms": ["ใบแจ้งหนี้"],
                },
            ),
        ],
        "payment_lookup": [
            BusinessScenario(
                "payment_lookup",
                f"ช่วยตรวจสอบว่าใบแจ้งหนี้ {prefix}90002 จำนวน 58 ดอลลาร์ ชำระแล้วหรือยัง",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90002"],
                    "amounts": ["58 dollars"],
                    "billing_actions": ["payment"],
                    "account_terms": ["ใบแจ้งหนี้"],
                },
            ),
        ],
        "payment_confirmation": [
            BusinessScenario(
                "payment_confirmation",
                f"ฉันต้องการยืนยันว่าใบแจ้งหนี้ {prefix}90003 ถูกชำระแล้ว",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90003"],
                    "billing_actions": ["payment"],
                    "confirmations": ["ยืนยัน"],
                    "account_terms": ["ใบแจ้งหนี้"],
                },
            ),
        ],
        "charge_refusal": [
            BusinessScenario(
                "charge_refusal",
                f"ไม่ค่ะ อย่าเรียกเก็บใบแจ้งหนี้ {prefix}90004 จำนวน 72 ดอลลาร์ จากบัตรนั้น",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90004"],
                    "amounts": ["72 dollars"],
                    "billing_actions": ["charge"],
                    "refusals": ["ไม่"],
                    "account_terms": ["บัตร"],
                },
            ),
        ],
        "account_balance": [
            BusinessScenario(
                "account_balance",
                f"ยอดคงเหลือของใบแจ้งหนี้ {prefix}90005 เท่าไร",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90005"],
                    "billing_actions": ["balance"],
                    "account_terms": ["ยอดคงเหลือ", "ใบแจ้งหนี้"],
                },
            ),
        ],
        "refund_request": [
            BusinessScenario(
                "refund_request",
                f"ขอคืนเงินสำหรับใบแจ้งหนี้ {prefix}90006 จำนวน 45 ดอลลาร์",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90006"],
                    "amounts": ["45 dollars"],
                    "billing_actions": ["refund"],
                    "account_terms": ["ใบแจ้งหนี้"],
                },
            ),
        ],
        "payment_extension": [
            BusinessScenario(
                "payment_extension",
                f"ขอขยายเวลาชำระใบแจ้งหนี้ {prefix}90007 จำนวน 89 ดอลลาร์",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90007"],
                    "amounts": ["89 dollars"],
                    "billing_actions": ["extension"],
                    "account_terms": ["ใบแจ้งหนี้"],
                },
            ),
        ],
        "case_close": [
            BusinessScenario(
                "case_close",
                f"ใช่ค่ะ เรื่องใบแจ้งหนี้ {prefix}90008 แก้ไขเรียบร้อยแล้ว ขอบคุณค่ะ",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90008"],
                    "confirmations": ["ใช่"],
                    "account_terms": ["ใบแจ้งหนี้"],
                },
            ),
        ],
    }


def _indonesian_scenarios() -> dict[str, list[BusinessScenario]]:
    prefix = "INV-ID"
    return {
        "billing_dispute": [
            BusinessScenario(
                "billing_dispute",
                f"Halo saya ingin mengajukan keberatan untuk tagihan {prefix}90001 sebesar 36 dolar karena jumlahnya salah",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90001"],
                    "amounts": ["36 dollars"],
                    "billing_actions": ["dispute"],
                    "account_terms": ["tagihan"],
                },
            ),
        ],
        "payment_lookup": [
            BusinessScenario(
                "payment_lookup",
                f"Bisakah Anda memeriksa apakah tagihan {prefix}90002 sebesar 58 dolar sudah dibayar",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90002"],
                    "amounts": ["58 dollars"],
                    "billing_actions": ["payment"],
                    "account_terms": ["tagihan"],
                },
            ),
        ],
        "payment_confirmation": [
            BusinessScenario(
                "payment_confirmation",
                f"Saya ingin mengonfirmasi bahwa tagihan {prefix}90003 sudah dibayar",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90003"],
                    "billing_actions": ["payment"],
                    "confirmations": ["konfirmasi"],
                    "account_terms": ["tagihan"],
                },
            ),
        ],
        "charge_refusal": [
            BusinessScenario(
                "charge_refusal",
                f"Tidak, jangan kenakan tagihan {prefix}90004 sebesar 72 dolar ke kartu itu",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90004"],
                    "amounts": ["72 dollars"],
                    "billing_actions": ["charge"],
                    "refusals": ["tidak"],
                    "account_terms": ["kartu"],
                },
            ),
        ],
        "account_balance": [
            BusinessScenario(
                "account_balance",
                f"Berapa sisa saldo tagihan {prefix}90005",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90005"],
                    "billing_actions": ["balance"],
                    "account_terms": ["saldo", "tagihan"],
                },
            ),
        ],
        "refund_request": [
            BusinessScenario(
                "refund_request",
                f"Mohon refund tagihan {prefix}90006 sebesar 45 dolar",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90006"],
                    "amounts": ["45 dollars"],
                    "billing_actions": ["refund"],
                    "account_terms": ["tagihan"],
                },
            ),
        ],
        "payment_extension": [
            BusinessScenario(
                "payment_extension",
                f"Saya perlu perpanjangan pembayaran untuk tagihan {prefix}90007 sebesar 89 dolar",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90007"],
                    "amounts": ["89 dollars"],
                    "billing_actions": ["extension"],
                    "account_terms": ["tagihan"],
                },
            ),
        ],
        "case_close": [
            BusinessScenario(
                "case_close",
                f"Ya, masalah tagihan {prefix}90008 sudah selesai. Terima kasih",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90008"],
                    "confirmations": ["ya"],
                    "account_terms": ["tagihan"],
                },
            ),
        ],
    }


def _chinese_scenarios() -> dict[str, list[BusinessScenario]]:
    prefix = "INV-ZH"
    return {
        "billing_dispute": [
            BusinessScenario(
                "billing_dispute",
                f"你好，我要对发票 {prefix}90001 提出异议，金额 36 美元，因为数额不对",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90001"],
                    "amounts": ["36 dollars"],
                    "billing_actions": ["dispute"],
                    "account_terms": ["发票"],
                },
            ),
        ],
        "payment_lookup": [
            BusinessScenario(
                "payment_lookup",
                f"请帮我查一下发票 {prefix}90002 的 58 美元是否已经付款",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90002"],
                    "amounts": ["58 dollars"],
                    "billing_actions": ["payment"],
                    "account_terms": ["发票"],
                },
            ),
        ],
        "payment_confirmation": [
            BusinessScenario(
                "payment_confirmation",
                f"我想确认发票 {prefix}90003 已经付款",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90003"],
                    "billing_actions": ["payment"],
                    "confirmations": ["确认"],
                    "account_terms": ["发票"],
                },
            ),
        ],
        "charge_refusal": [
            BusinessScenario(
                "charge_refusal",
                f"不，不要用那张卡收取发票 {prefix}90004 的 72 美元",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90004"],
                    "amounts": ["72 dollars"],
                    "billing_actions": ["charge"],
                    "refusals": ["不"],
                    "account_terms": ["卡"],
                },
            ),
        ],
        "account_balance": [
            BusinessScenario(
                "account_balance",
                f"发票 {prefix}90005 还有多少余额",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90005"],
                    "billing_actions": ["balance"],
                    "account_terms": ["余额", "发票"],
                },
            ),
        ],
        "refund_request": [
            BusinessScenario(
                "refund_request",
                f"请退款发票 {prefix}90006 的 45 美元",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90006"],
                    "amounts": ["45 dollars"],
                    "billing_actions": ["refund"],
                    "account_terms": ["发票"],
                },
            ),
        ],
        "payment_extension": [
            BusinessScenario(
                "payment_extension",
                f"我需要延长发票 {prefix}90007 的 89 美元付款期限",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90007"],
                    "amounts": ["89 dollars"],
                    "billing_actions": ["extension"],
                    "account_terms": ["发票"],
                },
            ),
        ],
        "case_close": [
            BusinessScenario(
                "case_close",
                f"好的，发票 {prefix}90008 的问题已经解决了，谢谢",
                {
                    **empty_entities(),
                    "invoice_ids": [f"{prefix}90008"],
                    "confirmations": ["好的"],
                    "account_terms": ["发票"],
                },
            ),
        ],
    }
