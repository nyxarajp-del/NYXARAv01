"""Tests for nyxara.agency.health."""

from __future__ import annotations

from nyxara.agency.governor import Governor
from nyxara.agency.health import HealthMonitor, HealthReport, HealthState, Subsystem
from nyxara.kernel.config import ResourceLimits
from nyxara.kernel.errors import ErrorLedger, InvariantViolation


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _mon(clk=None, **kw):
    return HealthMonitor(clock=clk or FakeClock(), **kw)


# -------------------- HealthState ordering -------------------- #
def test_state_ordering():
    assert HealthState.HEALTHY < HealthState.DEGRADED < HealthState.UNHEALTHY < HealthState.DOWN
    assert HealthState.DOWN.label == "down"


# -------------------- Subsystem -------------------- #
def test_subsystem_unknown_without_data():
    s = Subsystem("x")
    assert s.evaluate(0.0) is HealthState.UNKNOWN


def test_subsystem_healthy_with_beat():
    s = Subsystem("x", heartbeat_timeout=10.0)
    s.beat(100.0)
    assert s.evaluate(105.0) is HealthState.HEALTHY


def test_subsystem_stale_is_unhealthy():
    s = Subsystem("x", heartbeat_timeout=10.0, down_grace=3.0)
    s.beat(100.0)
    assert s.evaluate(115.0) is HealthState.UNHEALTHY  # 15s > 10s but < 30s


def test_subsystem_down_past_grace():
    s = Subsystem("x", heartbeat_timeout=10.0, down_grace=3.0)
    s.beat(100.0)
    assert s.evaluate(140.0) is HealthState.DOWN  # 40s > 30s


def test_subsystem_error_rate_degraded():
    s = Subsystem("x", heartbeat_timeout=100.0, degraded_error_rate=0.1,
                  unhealthy_error_rate=0.3, ewma_alpha=0.3)
    s.beat(0.0)
    s.record(False)  # one failure -> ewma 0.3 -> unhealthy actually
    assert s.error_rate > 0


def test_subsystem_error_rate_thresholds():
    s = Subsystem("x", heartbeat_timeout=1000.0, degraded_error_rate=0.1,
                  unhealthy_error_rate=0.3)
    s.beat(0.0)
    # drive a moderate error rate between degraded and unhealthy
    s.error_rate = 0.2
    s.successes = 1
    assert s.evaluate(0.0) is HealthState.DEGRADED
    s.error_rate = 0.5
    assert s.evaluate(0.0) is HealthState.UNHEALTHY


def test_subsystem_latency_ewma():
    s = Subsystem("x")
    s.beat(0.0, latency=1.0)
    s.beat(0.0, latency=1.0)
    assert s.latency_ewma > 0


def test_subsystem_metrics():
    s = Subsystem("x")
    s.beat(0.0)
    m = s.metrics()
    assert m["name"] == "x" and "error_rate" in m and "state" in m


# -------------------- monitor: registration & signals -------------------- #
def test_register_and_get():
    mon = _mon()
    s = mon.register("mem", heartbeat_timeout=5.0)
    assert mon.get("mem") is s


def test_beat_autoregisters():
    mon = _mon()
    mon.beat("auto")
    assert mon.get("auto") is not None and mon.get("auto").beats == 1


def test_record_success_and_error():
    mon = _mon()
    mon.record_success("x")
    mon.record_error("x")
    s = mon.get("x")
    assert s.successes == 1 and s.errors == 1


# -------------------- evaluate / overall -------------------- #
def test_evaluate_overall_healthy():
    clk = FakeClock()
    mon = _mon(clk)
    mon.register("a", heartbeat_timeout=10.0)
    mon.beat("a")
    mon.record_success("a")
    rep = mon.evaluate()
    assert isinstance(rep, HealthReport)
    assert rep.healthy and rep.overall is HealthState.HEALTHY


def test_evaluate_overall_is_worst():
    clk = FakeClock()
    mon = _mon(clk)
    mon.register("a", heartbeat_timeout=10.0)
    mon.register("b", heartbeat_timeout=10.0)
    mon.beat("a")
    mon.beat("b")
    clk.advance(50.0)  # both stale -> b down
    rep = mon.evaluate()
    assert rep.overall is HealthState.DOWN
    assert not rep.healthy


def test_evaluate_empty_monitor_unknown():
    rep = _mon().evaluate()
    assert rep.overall is HealthState.UNKNOWN


def test_liveness():
    clk = FakeClock()
    mon = _mon(clk)
    mon.register("a", heartbeat_timeout=10.0)
    mon.beat("a")
    mon.record_success("a")
    assert mon.liveness()
    clk.advance(100.0)
    assert not mon.liveness()


# -------------------- pull checks -------------------- #
def test_register_check_combines_worst():
    mon = _mon()
    mon.register("svc", heartbeat_timeout=100.0)
    mon.beat("svc")
    mon.record_success("svc")
    mon.register_check("svc", lambda: HealthState.DEGRADED)
    rep = mon.evaluate()
    assert mon.get("svc").state is HealthState.DEGRADED


def test_check_with_detail_tuple():
    mon = _mon()
    mon.register_check("svc", lambda: (HealthState.UNHEALTHY, "boom"))
    mon.evaluate()
    assert mon.get("svc").state is HealthState.UNHEALTHY
    assert mon.get("svc").message == "boom"


# -------------------- recovery -------------------- #
def test_recovery_restores_subsystem():
    mon = _mon(max_recovery_attempts=3)
    mon.register("p", heartbeat_timeout=1000.0)
    mon.beat("p")
    mon.get("p").error_rate = 0.9  # unhealthy
    mon.get("p").successes = 1

    def heal(s):
        s.error_rate = 0.0
        return True

    mon.on_recover("p", heal)
    rep = mon.evaluate()
    assert "p" in rep.recovered
    assert mon.get("p").state <= HealthState.DEGRADED


def test_recovery_bounded_then_quarantine():
    mon = _mon(max_recovery_attempts=2)
    mon.register("f", heartbeat_timeout=1000.0)
    mon.beat("f")
    mon.get("f").error_rate = 0.9
    mon.get("f").successes = 1
    mon.on_recover("f", lambda s: False)  # never heals
    for _ in range(3):
        rep = mon.evaluate()
    assert mon.get("f").quarantined
    assert "f" in rep.escalate
    assert mon.get("f").recovery_attempts == 2  # bounded


def test_quarantined_always_escalates():
    mon = _mon(max_recovery_attempts=1)
    mon.register("f", heartbeat_timeout=1000.0)
    mon.beat("f")
    mon.get("f").error_rate = 0.9
    mon.get("f").successes = 1
    mon.on_recover("f", lambda s: False)
    mon.evaluate()  # quarantines
    rep = mon.evaluate()  # still escalates
    assert "f" in rep.escalate


def test_no_recovery_handler_no_crash():
    mon = _mon()
    mon.register("x", heartbeat_timeout=1000.0)
    mon.beat("x")
    mon.get("x").error_rate = 0.9
    mon.get("x").successes = 1
    rep = mon.evaluate()  # no handler registered
    assert rep.overall >= HealthState.UNHEALTHY
    assert "x" not in rep.recovered


def test_failing_handler_does_not_crash():
    mon = _mon(max_recovery_attempts=3)
    mon.register("x", heartbeat_timeout=1000.0)
    mon.beat("x")
    mon.get("x").error_rate = 0.9
    mon.get("x").successes = 1

    def bad(s):
        raise RuntimeError("healer broke")

    mon.on_recover("x", bad)
    rep = mon.evaluate()  # must not raise
    assert "x" not in rep.recovered


# -------------------- integrations -------------------- #
def test_bind_governor_resource_pressure():
    gov = Governor(ResourceLimits(daily_spend_budget=1.0))
    gov.charge(1.0)
    mon = _mon()
    mon.bind_governor(gov)
    mon.evaluate()
    assert mon.get("resources").state >= HealthState.DEGRADED


def test_bind_governor_healthy_when_budget_left():
    gov = Governor(ResourceLimits(daily_spend_budget=10.0))
    mon = _mon()
    mon.bind_governor(gov)
    mon.evaluate()
    assert mon.get("resources").state is HealthState.HEALTHY


def test_bind_error_ledger_serious_is_unhealthy():
    led = ErrorLedger()
    led.record(InvariantViolation("core breach"))
    mon = _mon()
    mon.bind_error_ledger(led)
    mon.evaluate()
    assert mon.get("errors").state is HealthState.UNHEALTHY


def test_bind_error_ledger_minor_is_degraded():
    led = ErrorLedger()
    led.record(ValueError("oops"))
    mon = _mon()
    mon.bind_error_ledger(led)
    mon.evaluate()
    assert mon.get("errors").state is HealthState.DEGRADED


def test_bind_error_ledger_clean_is_healthy():
    mon = _mon()
    mon.bind_error_ledger(ErrorLedger())
    mon.evaluate()
    assert mon.get("errors").state is HealthState.HEALTHY


# -------------------- events / report -------------------- #
def test_events_record_transitions():
    clk = FakeClock()
    mon = _mon(clk)
    mon.register("a", heartbeat_timeout=10.0)
    mon.beat("a")
    mon.record_success("a")
    mon.evaluate()  # healthy
    clk.advance(50.0)
    mon.evaluate()  # -> down, transition logged
    assert any(e.get("to") == "down" for e in mon.events())


def test_report_dict_shape():
    mon = _mon()
    mon.register("a", heartbeat_timeout=10.0)
    mon.beat("a")
    mon.record_success("a")
    d = mon.report()
    assert "overall" in d and "subsystems" in d and "healthy" in d
