"""The English cue router is retired; typed semantic navigation owns Home."""
from __future__ import annotations

from realtime_api.profiles import get_profile


def test_profile_withdraws_the_keyword_resolver_hook():
    profile = get_profile("concierge")
    assert profile is not None
    assert profile.resolve_intents is None
