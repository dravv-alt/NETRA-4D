from drishti.engine.adaptive_escalation import (
    AdaptiveSpecialistEscalationController,
    EscalationState,
    SpecialistEscalationConfig,
    SpecialistPolicyMode,
)


def _controller():
    return AdaptiveSpecialistEscalationController(SpecialistEscalationConfig(
        mode=SpecialistPolicyMode.CONDITIONAL,
        trigger_threshold=.60,
        recovery_threshold=.30,
        min_consecutive_low_quality_frames=3,
        specialist_window_frames=4,
        cooldown_frames=3,
        min_specialist_interval_frames=2,
        min_recovery_frames=2,
    ))


def test_one_noisy_frame_cannot_escalate():
    controller = _controller()
    first = controller.step(0, 0., .9)
    second = controller.step(1, .04, .1)
    assert first.invoke_specialist is False
    assert second.state == EscalationState.BASELINE
    assert second.consecutive_low_quality_frames == 0


def test_sustained_trigger_interval_and_cooldown_hysteresis():
    controller = _controller()
    assert not controller.step(0, 0., .8).invoke_specialist
    assert not controller.step(1, .04, .8).invoke_specialist
    start = controller.step(2, .08, .8)
    assert start.state == EscalationState.SPECIALIST
    assert start.invoke_specialist is True
    assert start.transition == "BASELINE_TO_SPECIALIST"
    assert controller.step(3, .12, .8).invoke_specialist is False
    assert controller.step(4, .16, .8).invoke_specialist is True
    # The specialist window expires; the controller must cool down instead of
    # immediately re-triggering on the same poor-quality signal.
    cooldown = controller.step(6, .24, .8)
    assert cooldown.state == EscalationState.COOLDOWN
    assert cooldown.invoke_specialist is False
    assert cooldown.transition == "SPECIALIST_TO_COOLDOWN"
    assert controller.step(7, .28, .8).invoke_specialist is False
    assert controller.step(8, .32, .8).invoke_specialist is False
    resumed = controller.step(9, .36, .1)
    assert resumed.state == EscalationState.BASELINE
    assert resumed.transition == "COOLDOWN_TO_BASELINE"


def test_baseline_and_every_frame_policy_are_explicit():
    baseline = AdaptiveSpecialistEscalationController(SpecialistEscalationConfig(mode=SpecialistPolicyMode.BASELINE_ONLY))
    every = AdaptiveSpecialistEscalationController(SpecialistEscalationConfig(mode=SpecialistPolicyMode.EVERY_FRAME))
    assert baseline.step(0, 0., 1.).invoke_specialist is False
    assert every.step(0, 0., 0.).invoke_specialist is True
    assert every.step(1, .04, 0.).invoke_specialist is True
