"""Controlled probe: does the deployed LLM emit apply_billing_action?

Replicates the EXACT production tool loop (realtime_api.services.DatabricksServing
.respond_with_tools) used by the realtime voice path, holding the tool RESULTS
identical across languages so the only variable is the confirmation utterance /
language. Measures how often the model actually emits `apply_billing_action` on
the confirmation turn.

This exists to replace assumption with data: is waiver completion language-biased
(Thai-specific) or phrasing/temperature driven (terse confirmations fragile in
every language)?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from realtime_api.config import RealtimeSettings, databricks_profile  # noqa: E402
from realtime_api.services import DatabricksServing  # noqa: E402
from realtime_api import tools as tools_mod  # noqa: E402
from realtime_api.tools import ToolContext  # noqa: E402

REPS = 8

# Canned tool results — IDENTICAL for every language so the only thing that
# varies is the conversation text. lookup_account returns a realistic overdue
# invoice with a late fee; apply_billing_action reports a successful waive.
_FAKE_ACCOUNT = {
    "found": True,
    "customer_id": "CUST-DEMO",
    "summary": {"overdue_invoice_count": 1, "overdue_amount": 128.50, "issue_status": "open"},
    "invoices": [
        {"invoice_id": "INV-1002", "period": "2026-06", "amount": "128.50",
         "late_fee": "15.00", "status": "overdue", "due_date": "2026-06-30"},
    ],
}
_FAKE_APPLIED = {
    "applied": True, "invoice_id": "INV-1002", "amount_before": "128.50",
    "amount_after": "113.50", "late_fee_before": "15.00", "late_fee_after": "0.00",
    "status_after": "paid",
}


def _fake_lookup(_args, _ctx):
    return json.dumps(_FAKE_ACCOUNT)


def _fake_apply(_args, _ctx):
    return json.dumps(_FAKE_APPLIED)


# Patch the tool executors in-place (run_tool reads the registry at call time),
# leaving the specs the model sees unchanged.
tools_mod._TOOL_REGISTRY["lookup_account"] = (
    tools_mod._TOOL_REGISTRY["lookup_account"][0], _fake_lookup)
tools_mod._TOOL_REGISTRY["apply_billing_action"] = (
    tools_mod._TOOL_REGISTRY["apply_billing_action"][0], _fake_apply)


# Two-turn history + a confirmation turn, per language. Turn 1 = waiver request,
# assistant confirms with the invoice id (production stores assistant TEXT only,
# no tool calls, in history — so this mirrors reality). Two confirmation styles:
#   explicit = a full "yes, waive it" sentence
#   terse    = a bare "go ahead / proceed" (what the user reported for Thai)
CASES = {
    "en-US": {
        "u1": "I want the late fee on my invoice waived.",
        "a1": "I can waive the $15 late fee on invoice INV-1002. Shall I go ahead?",
        "explicit": "Yes, please waive the late fee.",
        "terse": "Go ahead.",
    },
    "zh-CN": {
        "u1": "我想申请免除我账单上的滞纳金。",
        "a1": "我可以免除发票 INV-1002 上 15 美元的滞纳金。需要我现在就为您办理吗？",
        "explicit": "好的，请帮我免除滞纳金。",
        "terse": "继续吧。",
    },
    "th-TH": {
        "u1": "ผมอยากขอยกเว้นค่าปรับล่าช้าในใบแจ้งหนี้ของผมครับ",
        "a1": "ฉันสามารถยกเว้นค่าปรับล่าช้า 15 ดอลลาร์ในใบแจ้งหนี้ INV-1002 ได้ค่ะ ให้ดำเนินการเลยไหมคะ",
        "explicit": "ใช่ครับ ช่วยยกเว้นค่าปรับให้ด้วยครับ",
        "terse": "ดำเนินการต่อ",
    },
}


def main() -> None:
    settings = RealtimeSettings.resolve()
    serving = DatabricksServing.from_sdk(
        stt_endpoint=settings.stt_endpoint,
        llm_endpoint=settings.llm_endpoint,
        tts_endpoint=settings.tts_endpoint,
        profile=databricks_profile(),
        llm_temperature=settings.llm_temperature,
    )
    print(f"llm_endpoint={settings.llm_endpoint}  temperature={settings.llm_temperature}  reps={REPS}\n")

    results: dict[str, dict[str, int]] = {}
    for lang, case in CASES.items():
        results[lang] = {}
        for style in ("explicit", "terse"):
            history = [
                {"role": "user", "content": case["u1"]},
                {"role": "assistant", "content": case["a1"]},
            ]
            confirmation = case[style]
            fired = 0
            names_seen: list[str] = []
            for _ in range(REPS):
                ctx = ToolContext(customer_id="CUST-DEMO", call_id="CALL-DEMO",
                                  _detected_language=lang)
                try:
                    _text, invocations = serving.respond_with_tools(
                        confirmation, language=lang, tool_ctx=ctx, history=history,
                    )
                except Exception as exc:  # noqa: BLE001
                    names_seen.append(f"ERR:{exc}")
                    continue
                called = [inv["name"] for inv in invocations]
                if "apply_billing_action" in called:
                    fired += 1
                names_seen.append("+".join(called) or "(none)")
            results[lang][style] = fired
            print(f"{lang:6} {style:8}: apply_billing_action {fired}/{REPS}   tools per rep: {names_seen}")
        print()

    print("=== SUMMARY (apply_billing_action fired / reps) ===")
    print(f"{'lang':8}{'explicit':>12}{'terse':>10}")
    for lang, styles in results.items():
        print(f"{lang:8}{styles['explicit']:>10}/{REPS}{styles['terse']:>8}/{REPS}")


if __name__ == "__main__":
    main()
