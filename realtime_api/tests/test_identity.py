from types import SimpleNamespace

from realtime_api.runtime import identity
from realtime_api.runtime.identity import SessionPrincipal, enrich_principal_identity


def test_identity_enrichment_uses_current_user_without_caching_token(monkeypatch):
    identity._IDENTITY_CACHE.clear()
    calls = []

    class CurrentUser:
        def me(self):
            calls.append("me")
            return SimpleNamespace(
                display_name="Example Person",
                user_name="person@example.com",
            )

    monkeypatch.setattr(
        identity,
        "workspace_client_for_principal",
        lambda _principal: SimpleNamespace(current_user=CurrentUser()),
    )
    principal = SessionPrincipal(
        access_token="secret-token",
        email="person@example.com",
        username="person@example.com",
        from_forwarded_header=True,
    )
    enriched = enrich_principal_identity(principal)
    assert enriched.name == "Example Person"
    assert enriched.email == "person@example.com"
    assert calls == ["me"]
    assert "secret-token" not in repr(enriched)
    assert all("secret-token" not in repr(item) for item in identity._IDENTITY_CACHE.items())

    # A second session for the same immutable actor uses only cached public fields.
    assert enrich_principal_identity(principal).name == "Example Person"
    assert calls == ["me"]
