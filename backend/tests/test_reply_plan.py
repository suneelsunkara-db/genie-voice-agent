from genie_voice.assist.reply_plan import (
    build_reply_action_plan,
    compose_action_instructions,
    render_deterministic_reply,
)

OMAR_ACCOUNT = {
    "found": True,
    "customer": {"full_name": "Omar Patel", "status": "at_risk"},
    "summary": {"overdue_amount": 239.0, "overdue_invoice_count": 1},
    "invoices": [
        {
            "invoice_id": "INV-90114",
            "status": "overdue",
            "amount": "239.00",
            "late_fee": "40.00",
            "period": "2026-04",
            "due_date": "2026-04-23",
        }
    ],
}


def test_render_deterministic_reply_waiver_and_plan_zh():
    nudge = {"waiver_requested": True, "payment_plan_requested": True, "next_best_action": "set_up_payment_plan"}
    plan = build_reply_action_plan(
        nudge,
        OMAR_ACCOUNT,
        customer_message="可以帮我免除滞纳金并设置一个付款计划吗?",
    )
    reply = render_deterministic_reply(
        plan,
        language="zh-CN",
        opener="根据 Genie 洞察，",
        customer_name="Omar Patel",
    )
    assert reply
    assert "免除" in reply
    assert "付款计划" in reply
    assert "自动扣款" not in reply
    assert "INV-90114" in reply


def test_build_reply_action_plan_waiver_and_plan_from_customer_text_when_fm_neutral():
    nudge = {"next_best_action": "continue", "waiver_requested": False, "payment_plan_requested": False}
    plan = build_reply_action_plan(
        nudge,
        OMAR_ACCOUNT,
        customer_message="可以帮我免除滞纳金并设置一个付款计划吗?",
    )
    assert plan.offer_waiver is True
    assert plan.offer_payment_plan is True


def test_build_reply_action_plan_waiver_and_plan_from_fm_flags():
    nudge = {
        "next_best_action": "continue",
        "waiver_requested": True,
        "payment_plan_requested": True,
        "all_intents": ["payment_arrangement", "late_fee"],
    }
    plan = build_reply_action_plan(nudge, OMAR_ACCOUNT)
    assert plan.offer_waiver is True
    assert plan.offer_payment_plan is True
    assert plan.invoice_id == "INV-90114"
    assert "waiver" in plan.title.lower() or "waive" in plan.title.lower()


def test_build_reply_action_plan_from_nba_offer_fee_waiver():
    nudge = {"next_best_action": "offer_fee_waiver", "waiver_requested": False, "payment_plan_requested": False}
    plan = build_reply_action_plan(nudge, OMAR_ACCOUNT)
    assert plan.offer_waiver is True


def test_compose_action_instructions_requires_proceed_question_for_dual_offer():
    nudge = {
        "next_best_action": "continue",
        "waiver_requested": True,
        "payment_plan_requested": True,
    }
    plan = build_reply_action_plan(nudge, OMAR_ACCOUNT)
    instructions = compose_action_instructions(plan)
    assert "payment plan" in instructions.lower()
    assert "waive" in instructions.lower()
    assert "proceed" in instructions.lower()


def test_continue_nba_without_flags_is_informational():
    nudge = {"next_best_action": "continue", "waiver_requested": False, "payment_plan_requested": False}
    plan = build_reply_action_plan(nudge, OMAR_ACCOUNT)
    assert plan.offer_waiver is False
    assert plan.offer_payment_plan is False
    assert "unrelated" in compose_action_instructions(plan).lower()
