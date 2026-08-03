"""Tests for nyxara.sim.epi_world — an epidemic whose dynamics genuinely emerge.

The simulator must *integrate* a compartmental model (SIR / SEIR) so that the epidemiological
signals a scientist reads off a real outbreak — the early exponential growth rate, the peak,
the final size — emerge from the dynamics rather than being written down. All hermetic, no LLM.
"""

from __future__ import annotations


from nyxara.sim.epi_world import EpidemicWorld, Outbreak


def test_sir_outbreak_emerges_and_burns_out():
    w = EpidemicWorld(beta=0.6, gamma=0.2)
    ob = w.simulate()
    assert isinstance(ob, Outbreak)
    assert 0.0 < ob.peak_prevalence() < 1.0
    assert ob.peak_time() > 0.0
    # with R0 = 3 a large fraction is eventually infected, and the epidemic ends (I → ~0)
    assert ob.final_size() > 0.5
    assert ob.I[-1] < 0.05


def test_early_growth_rate_matches_beta_minus_gamma():
    # the governing law of the early phase is r = β − γ; the simulator must reproduce it closely
    for beta, gamma in ((0.6, 0.2), (0.9, 0.3), (0.45, 0.15)):
        r = EpidemicWorld(beta=beta, gamma=gamma).early_growth_rate()
        assert r is not None
        assert abs(r - (beta - gamma)) < 0.05


def test_subthreshold_epidemic_does_not_take_off():
    # R0 < 1 (β < γ): the outbreak decays instead of growing — the epidemic threshold, emergent
    w = EpidemicWorld(beta=0.1, gamma=0.3)
    assert w.r0 < 1.0
    ob = w.simulate()
    assert ob.peak_prevalence() <= ob.I[0] + 1e-6      # never rises above the seed


def test_seir_adds_a_latent_compartment():
    seir = EpidemicWorld(beta=0.8, gamma=0.2, sigma=0.4).simulate()
    assert seir.E                                       # the exposed compartment is populated
    assert max(seir.E) > 0.0
    assert seir.peak_prevalence() > 0.0


def test_determinism():
    a = EpidemicWorld(beta=0.6, gamma=0.2).simulate()
    b = EpidemicWorld(beta=0.6, gamma=0.2).simulate()
    assert a.I == b.I
