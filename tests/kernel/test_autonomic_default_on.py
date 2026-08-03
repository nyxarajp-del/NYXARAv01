"""L-AGENCY: she lives between prompts — but never underneath a hermetic test."""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile


def test_the_autonomic_loop_is_on_by_default():
    """Autonomy is the point of L-AGENCY: without a prompt she should still be finding gaps,
    acquiring, and training rather than idling until spoken to."""
    for profile in (Profile.DEV, Profile.PROD):
        assert NyxaraSettings.for_profile(profile).server.autonomic is True


def test_the_test_profile_forces_it_off():
    """A hermetic suite must never sprout a background thread that reaches the network, trains,
    or writes state underneath the test that is running."""
    assert NyxaraSettings.for_profile(Profile.TEST).server.autonomic is False


def test_it_can_be_turned_off_explicitly(monkeypatch):
    monkeypatch.setenv("NYXARA_SERVER__AUTONOMIC", "false")
    assert NyxaraSettings().server.autonomic is False


def test_autonomy_does_not_widen_what_she_is_allowed_to_do():
    """Autonomy changes what she *initiates*, never what she is *permitted*. In particular an
    AUTONOMOUS actor still cannot authorise self-modification for itself."""
    from nyxara.agency.permissions import Authority, Capability, PermissionRequest, RiskTier
    from nyxara.agency.permissions import build_default_policy

    policy = build_default_policy()
    request = PermissionRequest(capability=Capability.SELF_MODIFY, authority=Authority.AUTONOMOUS,
                                risk=RiskTier.CRITICAL, reversible=False,
                                reason="rewrite my own source")
    assert not policy.check(request).allowed
