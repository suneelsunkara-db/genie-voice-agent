"""Long governed work stays conversational without becoming noisy."""
from __future__ import annotations

from realtime_api.runtime.engagement import (
    HEARTBEAT_EVERY_S,
    MAX_PROGRESS_MOMENTS,
    EngagementKind,
    LongWorkEngagement,
)
from realtime_api.runtime.speech_scheduler import (
    SpeechKind,
    SpeechRequest,
    SpeechScheduler,
)


def test_cadence_has_no_message_before_grace_window():
    policy = LongWorkEngagement()
    assert policy.pop_due(0.79) == []
    assert policy.next_due_s == 0.8


def test_each_stage_emits_once_in_order():
    policy = LongWorkEngagement()

    assert [stage.key for stage in policy.pop_due(0.8)] == ["ack"]
    assert policy.pop_due(11.99) == []
    assert [stage.key for stage in policy.pop_due(12.0)] == ["progress_1"]
    assert [stage.key for stage in policy.pop_due(30.0)] == ["progress_2"]
    assert [stage.key for stage in policy.pop_due(55.0)] == ["progress_3"]


def test_a_backlog_speaks_only_the_current_moment():
    """After a stall the caller wants one sentence, not a queue read back."""
    policy = LongWorkEngagement()
    assert [stage.key for stage in policy.pop_due(55.0)] == ["progress_3"]
    assert policy.next_due_s == 55.0 + HEARTBEAT_EVERY_S


def test_only_the_first_stage_is_an_ack():
    policy = LongWorkEngagement()
    stages = [policy.pop_due(t) for t in (0.8, 12.0, 30.0, 55.0)]
    flat = [stage for group in stages for stage in group]
    assert flat[0].kind == EngagementKind.ACK
    assert all(stage.kind == EngagementKind.PROGRESS for stage in flat[1:])


def test_work_running_past_the_opening_cadence_keeps_speaking():
    """A two-minute governed read used to go silent at 55s, which reads as hung."""
    policy = LongWorkEngagement()
    for due in (0.8, 12.0, 30.0, 55.0):
        assert policy.pop_due(due)
    assert not policy.exhausted

    spoken = 0
    elapsed = 55.0
    while elapsed < 200.0:
        elapsed += HEARTBEAT_EVERY_S
        spoken += len(policy.pop_due(elapsed))
    assert spoken >= 4


def test_a_real_upstream_step_pushes_the_generic_heartbeat_back():
    policy = LongWorkEngagement()
    for due in (0.8, 12.0, 30.0, 55.0):
        policy.pop_due(due)
    next_before = policy.next_due_s
    assert next_before is not None

    policy.defer(next_before - 1.0)
    assert policy.next_due_s is not None
    assert policy.next_due_s > next_before


def test_scheduler_budget_matches_the_cadence_it_has_to_carry():
    """A cadence the policy allows must not be silently dropped by the budget."""
    scheduler = SpeechScheduler()

    assert scheduler.accept(SpeechRequest(SpeechKind.ACK, "Looking now."))
    assert not scheduler.accept(SpeechRequest(SpeechKind.ACK, "Another ack."))
    for index in range(MAX_PROGRESS_MOMENTS):
        assert scheduler.accept(
            SpeechRequest(SpeechKind.PROGRESS, f"Progress {index}.")
        )
    assert not scheduler.accept(SpeechRequest(SpeechKind.PROGRESS, "Too many."))
    assert scheduler.skipped == ["ack:budget", "progress:budget"]


def test_policy_is_bounded_even_for_a_very_long_lookup():
    policy = LongWorkEngagement()
    total = 0
    for elapsed in range(0, 7200, 5):
        total += len(policy.pop_due(float(elapsed)))
    assert policy.exhausted
    assert total <= MAX_PROGRESS_MOMENTS + 1  # + the single ack
