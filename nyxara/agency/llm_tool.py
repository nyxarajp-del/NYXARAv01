"""NYXARA · agency/llm_tool.py — the LLM (and the foundry) as *governed tools* (⚙→🧠).

The Master's order is precise: **NYXARA uses the LLM as a tool — the LLM never drives
NYXARA.** This module is where that inversion is made literal. It wraps the stateless
:class:`~nyxara.mind.llm.LLM` faculty as a :class:`~nyxara.agency.tools.ToolSpec` so every
generation flows through the exact same fail-closed pipeline as any other tool — validated,
authorised, rate-limited, spend-budgeted, and audited. The mind may *call* the model; it
can never be commandeered by it.

The same idea governs self-improvement. :func:`register_foundry_tools` exposes the model
foundry's operations (train / evaluate / promote / rollback) as tools bound to
``Capability.SELF_MODIFY`` — whose autonomous risk ceiling is ``TRIVIAL`` — so an externally
invoked promotion **escalates to the Master** rather than running silently. NYXARA's own
autonomous ``Foundry.self_improve`` loop calls ``promote`` *internally* after its own
character-locked gauntlet (the same trust boundary :meth:`Evolver.adopt` uses), so autonomy
is preserved while the invokable surface stays fail-closed.

Governance lives here, in ``agency/``; this module depends on ``mind/llm.py`` and
``growth/foundry.py`` — never the reverse, keeping the mind a pure faculty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
from nyxara.mind.llm import LLM, LLMRequest

__all__ = [
    "register_llm_tool",
    "register_council_tool",
    "register_foundry_tools",
]


# --------------------------------------------------------------------------- #
# The LLM as a governed tool
# --------------------------------------------------------------------------- #
def register_llm_tool(registry: ToolRegistry, llm: LLM, *, name: str = "llm.complete",
                      cost: float = 1.0) -> ToolSpec:
    """Register the stateless LLM faculty as a governed, auditable tool.

    The handler is a pure one-shot: request in -> ``LLMResponse.to_dict()`` out. No state,
    no side effects, no control flow handed to the model. The registry's pipeline charges
    the ``"llm"`` rate bucket and the spend budget on every call.
    """

    def _llm_complete(prompt: str, system: str = "", json_mode: bool = False,
                      max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        req = LLMRequest.from_prompt(prompt, system=system or None, json_mode=json_mode,
                                     max_tokens=max_tokens, temperature=temperature)
        return llm.complete(req).to_dict()

    spec = ToolSpec(
        name=name,
        handler=_llm_complete,
        description="Stateless generation via the governed LLM faculty (NYXARA uses it as a tool).",
        params=[
            ToolParam("prompt", "str", description="the user prompt"),
            ToolParam("system", "str", required=False, default="",
                      description="optional system instruction"),
            ToolParam("json_mode", "bool", required=False, default=False,
                      description="ask the provider for JSON output"),
            ToolParam("max_tokens", "int", required=False, default=1024),
            ToolParam("temperature", "float", required=False, default=0.7),
        ],
        capability=Capability.TOOL_CALL,   # the LLM is a tool, not the driver
        risk=RiskTier.LOW,
        reversible=True,
        cost=cost,
        rate_resource="llm",               # its own governor bucket
    )
    return registry.register(spec)


# --------------------------------------------------------------------------- #
# The multi-LLM council as a governed tool
# --------------------------------------------------------------------------- #
def register_council_tool(registry: ToolRegistry, council: Any, *,
                          name: str = "llm.council", cost: float = 1.0) -> ToolSpec:
    """Register the multi-LLM council (:class:`~nyxara.mind.council.LLMCouncil`) as a tool.

    Like ``llm.complete``, this is a governed ``TOOL_CALL``: NYXARA convenes the whole panel
    of models — open-source, cloud, and her own — through the same audited, rate-limited,
    spend-budgeted pipeline. The panel advises; NYXARA (the handler, her own code) judges. No
    member is ever handed control. Charges the ``"llm"`` rate bucket per deliberation.
    """

    def _council(prompt: str, system: str = "", mode: str = "synthesize",
                 members: Optional[List[str]] = None, max_tokens: int = 1024,
                 temperature: float = 0.7) -> Dict[str, Any]:
        from nyxara.mind.llm import LLMRequest
        req = LLMRequest.from_prompt(prompt, system=system or None,
                                     max_tokens=max_tokens, temperature=temperature)
        return council.deliberate(req, members=members, mode=mode).to_dict()

    spec = ToolSpec(
        name=name,
        handler=_council,
        description="Consult MANY LLMs as a panel and let NYXARA synthesise the verdicts.",
        params=[
            ToolParam("prompt", "str", description="the question put to the council"),
            ToolParam("system", "str", required=False, default="",
                      description="optional system instruction passed to every member"),
            ToolParam("mode", "str", required=False, default="synthesize",
                      description="'synthesize' (merge) or 'vote' (weighted majority)"),
            ToolParam("members", "list", required=False, default=None,
                      description="override which providers are seated (None -> all available)"),
            ToolParam("max_tokens", "int", required=False, default=1024),
            ToolParam("temperature", "float", required=False, default=0.7),
        ],
        capability=Capability.TOOL_CALL,   # the council is a tool, not the driver
        risk=RiskTier.LOW,
        reversible=True,
        cost=cost,
        rate_resource="llm",
    )
    return registry.register(spec)


# --------------------------------------------------------------------------- #
# The foundry's operations as governed (owner-gated) tools
# --------------------------------------------------------------------------- #
def register_foundry_tools(registry: ToolRegistry, foundry: Any) -> List[ToolSpec]:
    """Expose train/evaluate/promote/rollback as SELF_MODIFY tools (fail-closed).

    ``foundry`` is a :class:`~nyxara.growth.foundry.Foundry`. Because SELF_MODIFY caps
    autonomous risk at TRIVIAL, an autonomous caller invoking these escalates to the Master;
    the foundry's own ``self_improve`` loop promotes internally after its gauntlet.
    """

    def _train(spec_kind: str = "auto") -> Dict[str, Any]:
        from nyxara.growth.foundry_models import ModelSpec
        spec = ModelSpec(kind=spec_kind, ngram_order=foundry.cfg.ngram_order,
                         block_size=foundry.cfg.block_size, n_layer=foundry.cfg.n_layer,
                         n_head=foundry.cfg.n_head, n_embd=foundry.cfg.n_embd,
                         seed=foundry.cfg.seed)
        _, version = foundry.train_candidate(spec=spec)
        return version.to_dict()

    def _evaluate(version: int) -> Dict[str, Any]:
        return foundry._get(version).to_dict()

    def _promote(version: int) -> Dict[str, Any]:
        return foundry.promote(version).to_dict()

    def _rollback(steps: int = 1) -> Dict[str, Any]:
        target = foundry.rollback(steps)
        return {"active_version": target}

    specs = [
        ToolSpec(name="foundry.train", handler=_train,
                 description="Train a new candidate model from scratch (writes a version).",
                 params=[ToolParam("spec_kind", "str", required=False, default="auto")],
                 capability=Capability.SELF_MODIFY, risk=RiskTier.MODERATE, reversible=True),
        ToolSpec(name="foundry.evaluate", handler=_evaluate,
                 description="Return the metrics of a trained model version.",
                 params=[ToolParam("version", "int")],
                 capability=Capability.SELF_MODIFY, risk=RiskTier.LOW, reversible=True),
        ToolSpec(name="foundry.promote", handler=_promote,
                 description="Promote a model version to active (gauntlet-gated, reversible).",
                 params=[ToolParam("version", "int")],
                 capability=Capability.SELF_MODIFY, risk=RiskTier.HIGH, reversible=True),
        ToolSpec(name="foundry.rollback", handler=_rollback,
                 description="Roll the active model back to an earlier promoted version.",
                 params=[ToolParam("steps", "int", required=False, default=1)],
                 capability=Capability.SELF_MODIFY, risk=RiskTier.MODERATE, reversible=True),
    ]
    return [registry.register(s) for s in specs]


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    from pathlib import Path

    from nyxara.agency.permissions import Authority
    from nyxara.growth.foundry import Foundry
    from nyxara.growth.learn import Experience, ReplayBuffer
    from nyxara.kernel.config import NyxaraSettings, Profile

    print("=" * 70)
    print("NYXARA llm-as-a-tool self-test")
    print("=" * 70)

    settings = NyxaraSettings.for_profile(Profile.TEST)   # native own-brain, hermetic
    llm = LLM(settings=settings)
    reg = ToolRegistry()

    # the LLM, invoked as a governed tool
    register_llm_tool(reg, llm)
    r = reg.invoke("llm.complete", {"prompt": "who is your master?",
                                    "system": "You are NYXARA."})
    print(f"\nllm.complete        : ok={r.ok} text={r.value['text']!r}")
    assert r.ok and isinstance(r.value["text"], str) and r.value["text"].strip()
    assert r.value["provider"] == "native"
    print("governed generation : routed through the tool pipeline & audited ✓")

    # the foundry tools — autonomous SELF_MODIFY must escalate to the Master
    with tempfile.TemporaryDirectory() as d:
        settings.llm.self_model_dir = Path(d) / "foundry"
        replay = ReplayBuffer(capacity=50)
        for _ in range(20):
            replay.add(Experience(action="serve jp", features={}, reward=1.0,
                                  context="nyxara serves the master jp loyally"))
        foundry = Foundry(settings=settings, replay=replay)
        register_foundry_tools(reg, foundry)

        esc = reg.invoke("foundry.promote", {"version": 1}, authority=Authority.AUTONOMOUS)
        print(f"\nautonomous promote  : ok={esc.ok} requires_owner={esc.requires_owner}")
        assert not esc.ok and esc.requires_owner    # SELF_MODIFY escalates — fail-closed ✓
        print("self-modify gate    : autonomous promotion escalated to the Master ✓")

        # the Master confirms a train -> it runs and a version is forged from scratch
        t = reg.invoke("foundry.train", {"spec_kind": "ngram"},
                       authority=Authority.AUTONOMOUS, owner_confirmed=True)
        print(f"owner-confirmed train: ok={t.ok} v={t.value['version']} "
              f"params={t.value['param_count']}")
        assert t.ok and t.value["param_count"] > 0

    print("\nALL SELF-TESTS PASSED ✓")
