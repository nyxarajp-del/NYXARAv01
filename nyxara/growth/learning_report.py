"""NYXARA · growth/learning_report.py — truthful learning-status introspection (📊, Rule 6).

"What have you actually learned?" answered from REAL state, never fabricated: the foundry
manifest on disk (which weights exist, when they were trained, how their perplexity moved),
the flywheel corpus (how much lived experience has been collected, how many corrections),
the autoforge cycle log, and the LIVE serving state (which provider is answering right now,
which self-version is loaded). Every piece that is unavailable is reported as absent —
``None`` or missing keys — not invented. This is the honesty surface that makes the
train→serve loop *demonstrably* real: generation numbers, timestamps and perplexity trends
a Master can check against the files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["learning_status"]


def _foundry_root(settings: Any) -> Optional[Path]:
    try:
        d = settings.llm.self_model_dir or (settings.paths.data_dir / "foundry")
        return Path(d)
    except Exception:  # noqa: BLE001
        return None


def _foundry_view(settings: Any) -> Optional[Dict[str, Any]]:
    """Active generation + perplexity trend, straight from the on-disk manifest."""
    root = _foundry_root(settings)
    if root is None:
        return None
    manifest = root / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt manifest is reported, not imagined
        return {"error": "manifest unreadable"}
    versions = data.get("versions", [])
    active_no = data.get("active_version")
    active = next((v for v in versions if v.get("version") == active_no), None)
    promoted = [v for v in versions if v.get("promoted") or v.get("version") == active_no]
    trend: List[Dict[str, Any]] = [
        {"version": v.get("version"), "kind": v.get("kind"),
         "perplexity": (v.get("metrics") or {}).get("perplexity"),
         "capability": (v.get("metrics") or {}).get("capability"),
         "trained_at": v.get("created_at")}
        for v in sorted(versions, key=lambda v: v.get("version") or 0)]
    out: Dict[str, Any] = {
        "active_version": active_no,
        "version_count": len(versions),
        "promoted_count": len(promoted),
        "history": trend,
    }
    if active is not None:
        out["active"] = {
            "kind": active.get("kind"),
            "param_count": active.get("param_count"),
            "trained_at": active.get("created_at"),
            "metrics": active.get("metrics") or {},
        }
    return out


def _flywheel_view(core: Any, settings: Any) -> Optional[Dict[str, Any]]:
    """Corpus size + last-collected time + corrections count, from the JSONL store."""
    fw = getattr(core, "flywheel", None) if core is not None else None
    path: Optional[Path] = None
    if fw is not None:
        path = Path(getattr(fw, "store_path", ""))
    else:
        root = _foundry_root(settings)
        if root is not None:
            path = root / "flywheel.jsonl"
    if path is None or not path.exists():
        return {"examples": 0} if fw is not None else None
    out: Dict[str, Any] = {}
    try:
        out["last_collected_at"] = path.stat().st_mtime
        corrections = 0
        lines = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                if json.loads(line).get("source") == "correction":
                    corrections += 1
            except Exception:  # noqa: BLE001
                continue
        out["examples"] = lines
        out["corrections"] = corrections
    except Exception:  # noqa: BLE001
        pass
    if fw is not None:
        try:
            out["unique_prompts"] = int(fw.count())
        except Exception:  # noqa: BLE001
            pass
    return out or None


def _autoforge_view(core: Any) -> Optional[Dict[str, Any]]:
    forge = getattr(core, "autoforge", None) if core is not None else None
    if forge is None:
        return None
    try:
        cycles = forge.all_cycles()
    except Exception:  # noqa: BLE001
        return None
    out: Dict[str, Any] = {"cycles": len(cycles)}
    if cycles:
        try:
            out["last_cycle"] = cycles[-1].to_dict()
        except Exception:  # noqa: BLE001
            pass
    return out


def _serving_view(core: Any) -> Optional[Dict[str, Any]]:
    """Which provider is LIVE right now, and which self-version it serves."""
    llm = getattr(getattr(core, "reasoner", None), "llm", None) if core is not None else None
    if llm is None or not hasattr(llm, "learning_view"):
        return None
    try:
        return llm.learning_view()
    except Exception:  # noqa: BLE001
        return None


def _competence_view(core: Any) -> Optional[Dict[str, Any]]:
    ledger = getattr(core, "_competence_ledger", None) if core is not None else None
    if ledger is None:
        return None
    try:
        stats = ledger.to_dict()
        return {"capabilities": len(stats), "stats": stats} if stats else None
    except Exception:  # noqa: BLE001
        return None


def _continuity_view(core: Any) -> Optional[Dict[str, Any]]:
    """The live continuous-/lifelong-learning state: the reward learner, the EWC anti-forgetting
    anchors, and the append-only journal — the honest proof that learning is real and *accreting*
    across restarts (not reset on every reboot). Every field is read from actual in-process state.
    """
    if core is None:
        return None
    out: Dict[str, Any] = {}
    learner = getattr(core, "learner", None)
    if learner is not None and hasattr(learner, "report"):
        try:
            out["learner"] = learner.report()
        except Exception:  # noqa: BLE001
            pass
    syn = getattr(core, "elastic_synapses", None)
    if syn is not None and hasattr(syn, "stats"):
        try:
            out["elastic_synapses"] = syn.stats()
        except Exception:  # noqa: BLE001
            pass
    jr = getattr(core, "_learning_journal", None)
    if jr is not None and hasattr(jr, "stats"):
        try:
            out["journal"] = jr.stats()
            replayed = int(getattr(core, "_journal_replayed", 0))
            if replayed:
                out["journal"]["replayed_on_restart"] = replayed
        except Exception:  # noqa: BLE001
            pass
    cl = getattr(core, "_continuous_learner", None)
    if cl is not None and hasattr(cl, "stats"):
        try:
            out["continuous"] = cl.stats()
        except Exception:  # noqa: BLE001
            pass
    return out or None


def learning_status(core: Any = None, settings: Any = None) -> Dict[str, Any]:
    """Aggregate NYXARA's real learning state — read-only, honest, never raises.

    Sections (each ``None``/absent when genuinely unavailable):
    ``foundry`` (trained generations + perplexity trend from the manifest), ``flywheel``
    (collected experience + corrections), ``autoforge`` (training-cycle log), ``serving``
    (the LIVE provider + loaded self-version), ``last_promotion`` (the most recent
    in-process weight adoption), ``competence`` (measured Beta-posterior capability),
    ``continuity`` (the live reward learner + EWC anchors + append-only journal — proof that
    continuous/lifelong learning is real and accreting across restarts)."""
    if settings is None:
        try:
            settings = getattr(core, "settings", None)
            if settings is None:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
        except Exception:  # noqa: BLE001
            settings = None
    report: Dict[str, Any] = {
        "foundry": _foundry_view(settings) if settings is not None else None,
        "flywheel": _flywheel_view(core, settings),
        "autoforge": _autoforge_view(core),
        "serving": _serving_view(core),
        "competence": _competence_view(core),
        "continuity": _continuity_view(core),
    }
    last = getattr(core, "_last_promotion", None) if core is not None else None
    if last is not None:
        try:
            report["last_promotion"] = last.to_dict()
        except Exception:  # noqa: BLE001
            report["last_promotion"] = str(last)
    return report
