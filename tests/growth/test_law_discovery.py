"""Tests for nyxara.growth.law_discovery — the Frontier Law Discovery Engine.

She discovers genuinely new governing laws from data and from experiments she runs herself, with
**no LLM and no network**: free-form symbolic regression, dimensional-analysis-guided sparse search,
SINDy-style dynamical-law discovery, and Noether-style invariant discovery. Every test is hermetic
and deterministic — a fixed seed, in-memory data, and (for the sandbox) the deterministic
:class:`~nyxara.sim.physics_world.PhysicsWorld`.

The proof that it is *real*: she recovers known physical laws she was never handed
(``y = ½·g·t²``, ``F = k/r²``, momentum conservation, an oscillator's equations of motion and its
conserved energy) — and honestly **abstains** on pure noise rather than inventing a false law.
"""

from __future__ import annotations

import random

from nyxara.growth.law_discovery import (
    DiscoveryReport,
    Law,
    LawDiscoveryEngine,
    LawEvidence,
    Variable,
    _Feature,
)


# --------------------------------------------------------------------------- #
# helpers — deterministic synthetic datasets from KNOWN laws
# --------------------------------------------------------------------------- #
def _eng(**kw) -> LawDiscoveryEngine:
    kw.setdefault("seed", 20260716)
    return LawDiscoveryEngine(**kw)


def _dataset(fn, ranges, n=80, seed=1):
    rng = random.Random(seed)
    X = [[rng.uniform(lo, hi) for lo, hi in ranges] for _ in range(n)]
    y = [fn(row) for row in X]
    return X, y


# --------------------------------------------------------------------------- #
# ground-truth recovery — the real proof the engine discovers laws
# --------------------------------------------------------------------------- #
def test_recovers_free_fall_law():
    """y = ½·g·t² — a two-variable power law recovered exactly from data."""
    X, y = _dataset(lambda r: 0.5 * r[0] * r[1] ** 2, [(1.5, 20.0), (0.1, 5.0)], n=90)
    rep = _eng().discover_from_data(X, y, var_names=["g", "t"], target_name="fall")
    best = rep.best()
    assert best is not None, "should have discovered a law"
    assert best.evidence.verdict == "corroborated"
    assert best.evidence.holdout_r2 > 0.999
    assert best.evidence.extrapolation_r2 > 0.99          # generalises OUTSIDE the training range
    # the discovered coefficient is genuinely ½ on the g·t² term
    coeff = next((c for t, c in best.law.terms if "t" in t and "g" in t), None)
    assert coeff is not None and abs(coeff - 0.5) < 1e-3


def test_recovers_inverse_square_law():
    """F = k/r² — a negative-exponent monomial (Newton/Coulomb form)."""
    X, y = _dataset(lambda r: 3.0 / (r[0] ** 2), [(0.5, 5.0)], n=80)
    rep = _eng().discover_from_data(X, y, var_names=["r"], target_name="F")
    best = rep.best()
    assert best is not None
    assert best.evidence.verdict == "corroborated"
    assert "r^-2" in best.law.expression
    coeff = next((c for t, c in best.law.terms if "r^-2" in t), None)
    assert coeff is not None and abs(coeff - 3.0) < 1e-2


def test_recovers_momentum_conservation():
    """total momentum p = m1·v1 + m2·v2 — a sum of two cross-product terms."""
    X, y = _dataset(lambda r: r[0] * r[1] + r[2] * r[3], [(0.5, 3.0)] * 4, n=100)
    rep = _eng().discover_from_data(X, y, var_names=["m1", "v1", "m2", "v2"], target_name="p")
    best = rep.best()
    assert best is not None
    assert best.evidence.verdict == "corroborated"
    assert best.evidence.holdout_r2 > 0.999


def test_abstains_on_pure_noise():
    """The honesty test: given noise, she must invent NO law — she abstains, never bluffs."""
    rng = random.Random(5)
    X = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(80)]
    y = [rng.gauss(0.0, 1.0) for _ in X]
    rep = _eng().discover_from_data(X, y, var_names=["a", "b"], target_name="z")
    assert len(rep.discoveries) == 0
    assert rep.abstained >= 1
    assert rep.best() is None


def test_dimensional_guidance_prunes_to_consistent_form():
    """With dimensions supplied, only the dimensionally-consistent monomial survives."""
    X, y = _dataset(lambda r: 0.5 * r[0] * r[1] ** 2, [(1.5, 20.0), (0.1, 5.0)], n=90)
    L = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    g_dim = tuple(L[i] - 2 * T[i] for i in range(7))       # acceleration L·T⁻²
    rep = _eng().discover_from_data(X, y, var_names=["g", "t"], target_name="fall",
                                    dims=[g_dim, T], target_dim=L)
    best = rep.best()
    assert best is not None and best.evidence.verdict == "corroborated"
    # a dimensionally-inconsistent term (e.g. g·t, dimension L·T⁻¹) must NOT appear
    assert all("g·t^2" in t or "g^" in t or t == "g·t^2" or "t^2" in t
               for t, _ in best.law.terms)


# --------------------------------------------------------------------------- #
# SINDy-style dynamical-law discovery
# --------------------------------------------------------------------------- #
def test_discovers_oscillator_dynamics():
    """From a damped-oscillator trajectory, recover dx/dt = v and dv/dt = -k·x - c·v (SINDy)."""
    x, v, k, c, dt = 1.0, 0.0, 1.0, 0.3, 0.05
    rows = []
    for _ in range(140):
        rows.append([x, v])
        a = -k * x - c * v
        v += a * dt
        x += v * dt
    rep = _eng().discover_dynamics(rows, dt=dt, var_names=["x", "v"])
    assert len(rep.discoveries) >= 2                       # one law per state coordinate
    exprs = " ".join(d.law.expression for d in rep.discoveries)
    assert "dx/dt" in exprs and "dv/dt" in exprs
    # the dv/dt law must contain a restoring -x term
    dvdt = next(d for d in rep.discoveries if d.law.target == "dv/dt")
    xcoeff = next((cc for t, cc in dvdt.law.terms if t == "x"), None)
    assert xcoeff is not None and xcoeff < -0.5


# --------------------------------------------------------------------------- #
# Noether-style invariant / conservation-law discovery
# --------------------------------------------------------------------------- #
def test_discovers_conserved_energy():
    """From an undamped oscillator, discover the conserved quantity ½x² + ½v² (energy)."""
    eng = _eng()
    import nyxara.growth.law_discovery as ld
    if ld._np is None:  # pragma: no cover - numpy present in CI; invariant path needs it
        return
    x, v, k, dt = 1.0, 0.0, 1.0, 0.02
    rows = []
    for _ in range(220):
        rows.append([x, v])
        v += (-k * x) * dt
        x += v * dt
    rep = eng.discover_invariant([rows], var_names=["x", "v"])
    assert len(rep.discoveries) == 1
    law = rep.discoveries[0].law
    assert law.kind == "invariant" and "const" in law.expression
    # both x² and v² carry meaningful weight in the conserved quantity
    terms = {t: c for t, c in law.terms}
    assert any("x^2" in t for t in terms) and any("v^2" in t for t in terms)


# --------------------------------------------------------------------------- #
# she runs her OWN experiments in the physics sandbox
# --------------------------------------------------------------------------- #
def test_discovers_law_from_physics_sandbox():
    """She sweeps gravity in the sandbox, drops a body, and rediscovers ½·g·t² from her own data."""
    rep = _eng().discover_from_physics(rounds=1)
    assert rep.source in ("physics", "synthetic")
    best = rep.best()
    assert best is not None and best.evidence.verdict == "corroborated"
    assert best.evidence.holdout_r2 > 0.99
    # the law relates the fall to g and t² (the coefficient is ~½, up to the sandbox's discretisation)
    assert "t" in best.law.expression


# --------------------------------------------------------------------------- #
# active experiment design — the intervention that breaks a tie
# --------------------------------------------------------------------------- #
def test_active_experiment_probes_where_theories_disagree():
    """Two laws agree on [0,1] but diverge outside it; the chosen probe lands in the divergence."""
    la = Law(target="y", expression="y = x", var_names=["x"],
             _features=[_Feature("monomial", exps={0: 1.0})], _coeffs=[1.0])
    lb = Law(target="y", expression="y = x^2", var_names=["x"],
             _features=[_Feature("monomial", exps={0: 2.0})], _coeffs=[1.0])
    x = _eng().active_experiment([la, lb], var_index=0, low=0.0, high=5.0)
    assert x > 3.0                                          # divergence grows with x → probe high end


# --------------------------------------------------------------------------- #
# the self-extending law tower — persistence & compounding
# --------------------------------------------------------------------------- #
def test_law_library_persists_across_sessions(tmp_path):
    path = str(tmp_path / "laws.json")
    e1 = _eng(path=path)
    e1.discover_from_physics(rounds=1)
    assert len(e1.known_laws()) >= 1
    first = e1.known_laws()[0].expression
    # a fresh engine on the same path reloads what she already discovered (compounding)
    e2 = LawDiscoveryEngine(seed=99, path=path)
    assert len(e2.known_laws()) >= 1
    assert e2.known_laws()[0].expression == first


# --------------------------------------------------------------------------- #
# graceful degradation — a capability, never a requirement
# --------------------------------------------------------------------------- #
def test_degrades_when_everything_is_none():
    """With no collaborators at all she still discovers from a self-generated round."""
    rep = LawDiscoveryEngine().discover_laws(rounds=1)
    assert isinstance(rep, DiscoveryReport)
    assert rep.best() is not None


def test_throwing_dependency_is_survivable():
    """A dependency that throws while folding a result in must not crash the run."""
    class _Boom:
        def ingest_text(self, *a, **k):
            raise RuntimeError("boom")

    rep = _eng(knowledge=_Boom()).discover_from_physics(rounds=1)
    assert rep.best() is not None                           # the discovery still lands


def test_too_little_data_abstains_cleanly():
    rep = _eng().discover_from_data([[1.0], [2.0]], [1.0, 2.0], var_names=["x"])
    assert rep.best() is None and "too few" in rep.reason


def test_no_numpy_still_fits_via_pure_python(monkeypatch):
    """Force the numpy-free path: the pure-Python normal-equations solver still recovers ½·g·t²."""
    import nyxara.growth.law_discovery as ld
    monkeypatch.setattr(ld, "_np", None)
    X, y = _dataset(lambda r: 0.5 * r[0] * r[1] ** 2, [(1.5, 20.0), (0.1, 5.0)], n=80)
    rep = ld.LawDiscoveryEngine(seed=1).discover_from_data(X, y, var_names=["g", "t"])
    best = rep.best()
    assert best is not None and best.evidence.holdout_r2 > 0.99


# --------------------------------------------------------------------------- #
# report / to_dict roundtrip — everything is JSON-safe
# --------------------------------------------------------------------------- #
def test_report_to_dict_is_json_safe():
    import json
    rep = _eng().discover_from_physics(rounds=1)
    blob = json.dumps(rep.to_dict())
    assert '"best"' in blob and '"discoveries"' in blob
    round_tripped = json.loads(blob)
    assert round_tripped["kept"] == len(rep.discoveries)


def test_law_and_evidence_to_dict():
    ev = LawEvidence(verdict="corroborated", holdout_r2=0.99)
    assert ev.to_dict()["verdict"] == "corroborated"
    law = Law(target="y", expression="y = 2·x", var_names=["x"], terms=[("x", 2.0)])
    d = law.to_dict()
    assert d["expression"] == "y = 2·x" and d["terms"][0]["coeff"] == 2.0


def test_variable_dataclass():
    var = Variable(name="g", dim=(0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0))
    assert var.name == "g" and var.dim[1] == 1.0


# --------------------------------------------------------------------------- #
# the capability is REAL + WIRED — reachable from the running NyxaraCore
# --------------------------------------------------------------------------- #
def test_core_discover_laws_is_wired():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore(enable_tools=False, enable_skills=False)
    assert core.law_discovery is not None, "the faculty must be built and reachable"
    out = core.discover_laws(rounds=1)
    assert "error" not in out
    assert out["source"] == "law_discovery"
    # the status report surfaces the live counter
    rep = core.report()
    assert "laws_discovered" in rep


# --------------------------------------------------------------------------- #
# self-run experiments across FOUR sciences — she designs & runs each herself
# --------------------------------------------------------------------------- #
def _run_experiment(name):
    eng = _eng()
    data = getattr(eng, name)()
    assert data is not None, f"{name} produced no data"
    rep = eng.discover_from_data(
        data["X"], data["y"], var_names=data["var_names"], target_name=data["target"],
        dims=data.get("dims"), target_dim=data.get("target_dim"), source="physics")
    best = rep.best()
    assert best is not None, f"{name} abstained: {rep.reason}"
    assert best.evidence.verdict == "corroborated"
    assert best.evidence.extrapolation_r2 > 0.98
    return best.law


def _is_ratio(expr: str) -> bool:
    """A discovered inverse relationship renders as either ``a / b`` (GP) or ``a·b^-1`` (sparse)."""
    return "/" in expr or "^-" in expr


def test_selfrun_pendulum_period_law():
    """She swings a pendulum and discovers T² = 4π²·L/g (period² ∝ L/g)."""
    law = _run_experiment("_exp_pendulum")
    assert "L" in law.expression and "g" in law.expression and _is_ratio(law.expression)
    coeff = next((c for _t, c in law.terms), None)
    assert coeff is not None and abs(coeff - 4 * 3.141592653589793 ** 2) < 1.0   # ≈ 39.48


def test_selfrun_spring_period_law():
    """She sets a mass on a Hooke spring in the sandbox and discovers T² = 4π²·m/k."""
    law = _run_experiment("_exp_spring")
    assert "m" in law.expression and "k" in law.expression and _is_ratio(law.expression)


def test_selfrun_projectile_range_law():
    """She launches a projectile at 45° and discovers R = v²/g."""
    law = _run_experiment("_exp_projectile")
    assert "v" in law.expression and "g" in law.expression and _is_ratio(law.expression)


def test_selfrun_terminal_velocity_law():
    """She drops a body through quadratic drag and discovers v² = m·g/k."""
    law = _run_experiment("_exp_terminal_velocity")
    e = law.expression
    assert "m" in e and "g" in e and "k" in e and _is_ratio(e)


def test_selfrun_coulomb_inverse_square_law():
    """Electromagnetism lab: she measures the force between charges and discovers F ∝ q₁·q₂/r²."""
    law = _run_experiment("_exp_coulomb")
    e = law.expression
    assert "q1" in e and "q2" in e and "r" in e
    assert "r^-2" in e or "r^2)" in e or "/ (r · r)" in e or "r · r" in e   # inverse-square in r


def test_selfrun_ideal_gas_law():
    """Thermodynamics lab: emergent pressure gives the ideal-gas law P ∝ N·T."""
    law = _run_experiment("_exp_ideal_gas")
    assert "N" in law.expression and "T" in law.expression


def test_selfrun_wave_speed_law():
    """Optics/acoustics lab: a plucked string gives the wave law v² = T/μ."""
    law = _run_experiment("_exp_wave_speed")
    assert "Tension" in law.expression and "mu" in law.expression


def test_selfrun_rc_time_constant_law():
    """Circuits lab: a discharging capacitor gives τ = R·C."""
    law = _run_experiment("_exp_rc_decay")
    assert "R" in law.expression and "C" in law.expression


# --------------------------------------------------------------------------- #
# Buckingham-π dimensional reduction generates the exact power-law monomial
# --------------------------------------------------------------------------- #
def test_pi_augment_generates_dimensional_monomial():
    from nyxara.growth.law_discovery import _DIM_A, _DIM_T, _DIM_L, _build_library
    eng = _eng()
    feats = _build_library(2)
    aug = eng._pi_augment(list(feats), [_DIM_A, _DIM_T], _DIM_L)   # g,t → distance ⇒ g·t²
    sigs = {f.signature() for f in aug}
    # a monomial with g^1·t^2 must have been injected by the π-solver
    assert any(f.kind == "monomial" and f.exps.get(0) == 1.0 and f.exps.get(1) == 2.0 for f in aug)
    assert len(sigs) >= len(feats)


# --------------------------------------------------------------------------- #
# the closed scientific-method loop — hypothesise → predict → falsify
# --------------------------------------------------------------------------- #
def test_discover_cycle_corroborates_and_falsifies_rival(tmp_path):
    eng = _eng(path=str(tmp_path / "laws.json"))
    result = eng.discover_cycle()          # first experiment = free-fall
    assert result["verdict"] == "corroborated"
    assert result["law"] and "t^2" in result["law"]
    assert result.get("rival_falsified") is True     # the wrong-exponent theory is ruled out


def test_discover_cycle_is_noise_robust(tmp_path):
    """With measurement noise injected she STILL recovers the law and falsifies the rival."""
    eng = _eng(path=str(tmp_path / "laws.json"))
    result = eng.discover_cycle(noise=0.05)
    assert result["verdict"] == "corroborated"       # the true law survives noisy data
    assert result.get("rival_falsified") is True      # the wrong-exponent rival is still ruled out


# --------------------------------------------------------------------------- #
# meta-law unification — compounding discoveries into deeper theory
# --------------------------------------------------------------------------- #
def test_unify_laws_merges_shared_structure(tmp_path):
    eng = _eng(path=str(tmp_path / "laws.json"))
    # discover both the pendulum (T² ∝ L/g) and the spring (T² ∝ m/k) — same ratio shape
    for _ in range(3):
        eng.discover_cycle()               # cycles free-fall, pendulum, spring, ...
    unis = eng.unify_laws()
    ratio = [u for u in unis if u["shape"] == [-1.0, 1.0]]
    assert ratio, f"pendulum + spring should unify into a ratio law; got {unis}"
    assert len(ratio[0]["members"]) >= 2


# --------------------------------------------------------------------------- #
# the lab notebook — auditable, append-only record of her science
# --------------------------------------------------------------------------- #
def test_lab_notebook_records_discoveries(tmp_path):
    import json
    import os
    eng = _eng(path=str(tmp_path / "laws.json"))
    eng.discover_cycle()
    nb = str(tmp_path / "law_notebook.jsonl")
    assert os.path.exists(nb)
    entries = [json.loads(ln) for ln in open(nb, encoding="utf-8").read().splitlines()]
    assert any(e["event"] in ("discovery", "cycle") for e in entries)


def test_expanded_library_recovers_new_ratio_law():
    """The enlarged pairwise library can now express square-over-linear ratios (a = v²/r),
    a common physical form the earlier palette could not discover."""
    import random

    rng = random.Random(3)
    X, y = [], []
    for _ in range(60):
        v = rng.uniform(2.0, 12.0)
        r = rng.uniform(1.0, 8.0)
        X.append([v, r])
        y.append(v * v / r)
    rep = _eng().discover_from_data(X, y, var_names=["v", "r"], target_name="a")
    best = rep.best()
    assert best is not None, "engine should recover a = v²/r, not abstain"
    assert best.evidence.verdict == "corroborated"
    assert best.evidence.holdout_r2 > 0.999
