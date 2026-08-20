"""Live shadow evaluation for the shared multilingual capability navigator."""
from __future__ import annotations

import json

from realtime_api.runtime.capabilities import CapabilityId
from realtime_api.runtime.navigation import classifier_capabilities
from realtime_api.serving_factory import shared_serving


CASES = [
    ("concierge", "Take me to telecom billing", "en-US", CapabilityId.NAVIGATE_TELCO),
    ("concierge", "Llévame a servicios financieros", "es-ES", CapabilityId.NAVIGATE_FSI),
    ("concierge", "मुझे नॉलेज एजेंट पर ले जाएँ", "hi-IN", CapabilityId.NAVIGATE_KNOWLEDGE),
    ("card", "I'd like the statement insights option", "en-US", CapabilityId.CARD_STATEMENT),
    ("card", "ทำไมค่าใช้จ่ายของฉันเพิ่มขึ้นในเดือนนี้", "th-TH", CapabilityId.INVESTIGATE_AGENT_MODE),
    ("card", "我现在有多少奖励积分？", "zh-CN", CapabilityId.PACK_FACTS),
    ("billing", "What is my overdue balance?", "en-US", CapabilityId.PACK_FACTS),
    ("billing", "Muéstrame la tendencia mensual de facturas vencidas", "es-ES", CapabilityId.BILLING_ANALYSIS),
    ("billing", "हाँ, विलंब शुल्क माफ कर दें", "hi-IN", CapabilityId.BILLING_ACTION),
    ("billing", "กรุงเทพตอนนี้กี่โมง", "th-TH", CapabilityId.CURRENT_TIME),
]


def main() -> None:
    serving = shared_serving()
    failures = 0
    for profile, utterance, language, expected in CASES:
        catalog = classifier_capabilities(profile)
        context = (
            "I can waive the late fee. Shall I go ahead?"
            if expected == CapabilityId.BILLING_ACTION
            else ""
        )
        raw = serving.classify_navigation(
            utterance,
            language=language,
            capabilities=[item.classifier_view() for item in catalog],
            context=context,
        )
        actual = raw.get("capability_id")
        confirmation_ok = (
            raw.get("confirmed") is True
            if expected == CapabilityId.BILLING_ACTION
            else True
        )
        ok = (
            actual == expected.value
            and float(raw.get("confidence") or 0) >= 0.8
            and confirmation_ok
        )
        failures += not ok
        print(
            json.dumps(
                {
                    "ok": ok,
                    "profile": profile,
                    "language": language,
                    "expected": expected.value,
                    "actual": actual,
                    "confidence": raw.get("confidence"),
                    "confirmed": raw.get("confirmed"),
                },
                ensure_ascii=False,
            )
        )
    if failures:
        raise SystemExit(f"{failures}/{len(CASES)} navigation cases failed")


if __name__ == "__main__":
    main()
