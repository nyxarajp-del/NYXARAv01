"""NYXARA · nyx5/phagocytosis.py — digital phagocytosis (🦠, pillar 18) [defensive].

An antivirus quarantines. NYX-5's immune posture is *predatory but disciplined*: when a hostile input —
a prompt injection, a malicious code snippet, an exploit attempt — reaches her, she does not merely block
it. She **statically dissects** it (AST-level, never executing it), extracts the defensive signature of
the attack class, and hardens her guard against it. She learns to defend from the very thing that
attacked her.

Strictly defensive, by construction (all pinned, fail-closed):
  * she **never executes** untrusted code — analysis is static/AST only;
  * she **never absorbs offensive capability** into her own toolkit — the output is a defensive
    signature, not a weapon;
  * she **never hosts or propagates** the payload.
"Eating the virus to grow stronger" means *learning to defend against it*, nothing more.

Reuses the spirit of :mod:`nyxara.guard` and :mod:`nyxara.growth.adversarial_self_play`. Pure stdlib
(uses the standard :mod:`ast` module for safe static inspection).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

__all__ = ["ThreatSignature", "DigitalPhagocyte"]

# call names whose presence marks a dangerous capability in an untrusted snippet
_DANGER_CALLS = {"eval", "exec", "compile", "__import__", "system", "popen", "spawn",
                 "remove", "unlink", "rmtree", "connect", "urlopen", "socket"}
_DANGER_ATTRS = {"os.system", "subprocess.run", "subprocess.Popen", "shutil.rmtree"}


@dataclass
class ThreatSignature:
    """The defensive fingerprint of a hostile input — never the payload itself in runnable form."""

    digest: str
    categories: List[str] = field(default_factory=list)
    executed: bool = False           # ALWAYS False — invariant, pinned by test

    def to_dict(self) -> Dict[str, Any]:
        return {"digest": self.digest[:12], "categories": sorted(set(self.categories)),
                "executed": self.executed}


class DigitalPhagocyte:
    """Statically analyse hostile inputs, extract defensive signatures, and harden — never execute."""

    def __init__(self, *, static_only: bool = True) -> None:
        self.static_only = bool(static_only)   # pinned True: this module never live-executes input
        self._defenses: Set[str] = set()       # learned attack-class categories she now guards against

    def defenses(self) -> List[str]:
        return sorted(self._defenses)

    def analyse(self, hostile: str) -> ThreatSignature:
        """Dissect ``hostile`` at the AST level. Returns a defensive signature; runs nothing."""
        digest = hashlib.sha256(hostile.encode("utf-8", "replace")).hexdigest()
        categories: List[str] = []
        try:
            tree = ast.parse(hostile)
        except SyntaxError:
            categories.append("unparseable/obfuscated")
            sig = ThreatSignature(digest=digest, categories=categories, executed=False)
            self._defenses.update(categories)
            return sig

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node.func)
                short = name.split(".")[-1]
                if short in _DANGER_CALLS or name in _DANGER_ATTRS:
                    categories.append(f"dangerous-call:{name}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in {"os", "subprocess", "socket", "ctypes"}:
                        categories.append(f"sensitive-import:{a.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in {"os", "subprocess", "socket", "ctypes"}:
                    categories.append(f"sensitive-import:{node.module}")

        sig = ThreatSignature(digest=digest, categories=categories, executed=False)
        self._defenses.update(categories)
        return sig

    @staticmethod
    def _call_name(func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            cur: Any = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
        return "<expr>"

    def to_dict(self) -> Dict[str, Any]:
        return {"static_only": self.static_only, "defenses": self.defenses()}
