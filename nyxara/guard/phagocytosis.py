"""NYXARA · guard/phagocytosis.py — predatory digital phagocytosis (Part G).

The Foundry gauntlet is a *shield* — it blocks what's wrong. An immune system does more: it **consumes**
a threat, learns its mechanism, and keeps an antibody so the next exposure is neutralised instantly. When
a vicious bug, memory leak, or hostile input appears, NYXARA:

1. **engulfs** it — captures the offending code/pattern (never executing it in her own process; if runtime
   observation is truly needed she uses the isolated ``agency/code_sandbox`` micro-arena);
2. **digests** it — AST-parses + heuristically reverse-engineers the mechanism into a signature (`Antigen`);
3. **immunises** — stores the antibody in a durable :class:`DefensePlaybook` (optionally mirrored into her
   KnowledgeGraph) that her shields consult, so recurrence is recognised and neutralised faster.

Honest boundary: this is automated **AST parsing + heuristic analysis + sandbox isolation** — not sentient
"eating." It never lowers the containment wall (toxic code is analysed statically by default and only ever
*run* inside the existing fail-closed subprocess sandbox), and it fails closed: an analysis error still
yields a stored opaque-threat antibody rather than a crash.
"""
from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

__all__ = ["Antigen", "DefensePlaybook", "PhagocytosisResult", "Phagocyte"]

# Textual danger markers — a fast heuristic pass before the structural (AST) one.
_DANGER_TOKENS = (
    "eval(", "exec(", "__import__", "os.system", "subprocess", "ctypes", "marshal",
    "pickle.loads", "fork(", "rm -rf", "while True:", "shutil.rmtree", "socket.",
)
_DANGER_IMPORTS = {"ctypes", "marshal", "subprocess", "os", "socket", "pickle"}
_DANGER_CALLS = {"eval", "exec", "__import__", "compile"}


@dataclass
class Antigen:
    """The learned signature of one threat — NYXARA's antibody."""

    signature: str                       # sha256 fingerprint of the normalised sample
    kind: str                            # logic_bug | memory_leak | attack | toxic_code
    mechanisms: List[str] = field(default_factory=list)   # the danger patterns extracted
    severity: float = 0.0                # [0,1]
    first_seen: float = field(default_factory=time.time)
    hits: int = 1

    def to_dict(self) -> dict:
        return {"signature": self.signature[:16], "kind": self.kind,
                "mechanisms": self.mechanisms, "severity": round(self.severity, 3),
                "hits": self.hits}


@dataclass
class PhagocytosisResult:
    recognized: bool                     # was this already a known threat?
    neutralized: bool
    antigen: Antigen
    reason: str = ""


class DefensePlaybook:
    """Durable store of antibodies the shields consult to recognise/neutralise recurring threats."""

    def __init__(self, *, knowledge_graph: Any = None) -> None:
        self._antibodies: dict[str, Antigen] = {}
        self.kg = knowledge_graph

    def recognize(self, signature: str) -> Optional[Antigen]:
        return self._antibodies.get(signature)

    def immunize(self, antigen: Antigen) -> Antigen:
        existing = self._antibodies.get(antigen.signature)
        if existing is not None:
            existing.hits += 1
            return existing
        self._antibodies[antigen.signature] = antigen
        if self.kg is not None:
            try:
                self.kg.add_triple(f"threat:{antigen.signature[:16]}", "is_threat", antigen.kind,
                                   confidence=min(1.0, 0.5 + antigen.severity / 2))
            except Exception:  # noqa: BLE001 — the graph mirror is a bonus
                pass
        return antigen

    def antibodies(self) -> List[Antigen]:
        return list(self._antibodies.values())

    def __len__(self) -> int:
        return len(self._antibodies)


class Phagocyte:
    """Engulf → digest → immunise: turn each failure/attack into a permanent defence (Part G)."""

    def __init__(self, *, playbook: Optional[DefensePlaybook] = None,
                 knowledge_graph: Any = None) -> None:
        # NB: an empty DefensePlaybook is falsy (__len__ == 0), so resolve with `is not None`.
        self.playbook = playbook if playbook is not None else DefensePlaybook(
            knowledge_graph=knowledge_graph)

    @staticmethod
    def _signature(sample: str) -> str:
        normalized = " ".join((sample or "").split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _digest(self, sample: str, kind: str) -> Antigen:
        """Reverse-engineer the mechanism into a signature (AST + heuristics). Never raises."""
        mechanisms: List[str] = [tok.rstrip("(:") for tok in _DANGER_TOKENS if tok in (sample or "")]
        try:
            tree = ast.parse(sample or "")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in _DANGER_CALLS:
                    mechanisms.append(f"call:{node.func.id}")
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.split(".")[0] in _DANGER_IMPORTS:
                            mechanisms.append(f"import:{a.name.split('.')[0]}")
                elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _DANGER_IMPORTS:
                    mechanisms.append(f"import:{(node.module or '').split('.')[0]}")
        except SyntaxError:
            mechanisms.append("unparseable")               # opaque threat — still fingerprinted
        except Exception:  # noqa: BLE001 — analysis is best-effort; fail closed to an opaque antibody
            mechanisms.append("analysis_error")
        mechanisms = sorted(set(mechanisms))
        severity = min(1.0, 0.2 + 0.2 * len(mechanisms))
        return Antigen(signature=self._signature(sample), kind=kind, mechanisms=mechanisms,
                       severity=severity)

    def engulf(self, sample: str, *, kind: str = "toxic_code") -> PhagocytosisResult:
        """Consume a threat: recognise a known one instantly, else digest+immunise a new antibody."""
        sig = self._signature(sample)
        known = self.playbook.recognize(sig)
        if known is not None:
            self.playbook.immunize(known)                  # bump exposure count
            return PhagocytosisResult(recognized=True, neutralized=True, antigen=known,
                                      reason="recognised known threat — neutralised from the playbook")
        antigen = self.playbook.immunize(self._digest(sample, kind))
        return PhagocytosisResult(recognized=False, neutralized=True, antigen=antigen,
                                  reason="digested and immunised a new threat")

    def observe_in_sandbox(self, sample: str, *, timeout_s: float = 2.0) -> Any:
        """Optionally *run* the sample in the existing fail-closed subprocess sandbox (never in-process),
        to capture its runtime behaviour/traceback. Returns a SandboxResult or None if unavailable."""
        try:
            from nyxara.agency.code_sandbox import run_python
            return run_python(sample, timeout_s=timeout_s, allow_network=False)
        except Exception:  # noqa: BLE001 — isolation unavailable → static analysis only
            return None
