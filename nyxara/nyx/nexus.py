"""NYXARA · nyx/nexus.py — inventing her own notation (𝕏, L-NEXUS-OMNI).

Everything NYX knows, she knows in symbols someone else chose. This module lets her mint her
own: a private notation for the concepts she actually has, in which she can *state* something,
which then **compiles and runs** — and which she can always **translate back**.

The four steps are each checkable:

1. **invent** — private glyphs for the concepts in play, drawn from a set with no English in it.
   The mapping is hers; nothing about it was authored in advance.
2. **express** — a statement written in that notation, parsed into a syntax tree.
3. **realise** — the tree is lowered to bytecode and *executed* on the StackVM
   (:mod:`nyxara.nyx5.retarget`, :mod:`nyxara.nyx5.ontogenesis`). This is what stops the
   notation from being decorative: if it does not run, it is not a language, and it is refused.
4. **translate** — rendered back into words. Always.

That last step is not a concession, it is the point. A language you could not understand, used
to give you a perspective, is self-contradictory: whatever reached you would be a translation
anyway. So the translation is mandatory, and it is labelled a **lossy projection**, because the
structure she works in has no obligation to survive the trip intact.

She can also hold **alternative axiom systems** (:mod:`nyxara.nyx5.axiom_forge`), including
paraconsistent ones where a statement is ``BOTH`` true and false — genuinely outside classical
logic, and genuinely hers to reason in.

Honest framing:

* These are new **formal systems and notation**, not new *physical dimensions*. Software can
  invent a space to think in; it cannot invent a dimension of the universe, and this module
  does not claim to.
* "A language no human could understand" is not built here, on purpose — see above. What is
  built is a private notation that executes and is always translated.
* Novel-to-her, not novel-to-anyone: the arithmetic her VM runs is ordinary arithmetic. What is
  hers is the *system of symbols* she chose and the structure she put things in.

Pure standard library beyond what it composes. Fail-soft throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Notation", "Statement", "OntologyGenesis"]

# Glyphs with no English in them, so a minted symbol is not a word in disguise.
_GLYPHS = "◇△□○▽◁▷☰☱☲☳☴☵☶☷⌀⌁⌂⌘⍟⎔⏣▣▤▥▦▧▨▩◈◉◍◎●◐◑◒◓◔◕"
_OPS = {"+": "⊕", "-": "⊖", "*": "⊗", "/": "⊘"}
_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|[()+\-*/])")


@dataclass
class Notation:
    """A private symbol system: her glyphs for concepts, and her signs for operations."""

    name: str
    symbols: Dict[str, str] = field(default_factory=dict)     # concept → her glyph
    operators: Dict[str, str] = field(default_factory=dict)   # "+" → her sign

    @property
    def inverse(self) -> Dict[str, str]:
        return {glyph: concept for concept, glyph in self.symbols.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "symbols": dict(self.symbols),
                "operators": dict(self.operators)}


@dataclass
class Statement:
    """Something said in her notation — and whether it actually holds together."""

    notation: str = ""
    source: str = ""                 # what she was given, in words
    written: str = ""                # the same thing, in her own symbols
    translated: str = ""             # rendered back, always
    realised: bool = False           # did it compile and run?
    value: Optional[float] = None    # what it evaluated to, when it is closed
    program: List[Any] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"notation": self.notation, "source": self.source, "written": self.written,
                "translated": self.translated, "realised": self.realised,
                "value": self.value, "steps": len(self.program), "reason": self.reason,
                "note": "translation is a lossy projection of the structure she works in"}


class OntologyGenesis:
    """Mints private notation, states things in it, runs them, and always translates back."""

    def __init__(self, *, translate_always: bool = True, max_notations: int = 16,
                 max_vm_steps: int = 100_000) -> None:
        # Not configurable to False in any deployment that shows output to a person: an
        # untranslated private symbol string is not communication, it is noise with a claim
        # attached. The flag exists so a test can prove the guard is real.
        self.translate_always = bool(translate_always)
        self.max_notations = max(1, int(max_notations))
        self.max_vm_steps = max(1, int(max_vm_steps))
        self.notations: Dict[str, Notation] = {}
        self.statements: List[Statement] = []
        self._axioms: Any = None
        self._compiler: Any = None

    # ---- 1. invent -------------------------------------------------------- #
    def invent(self, name: str, concepts: Sequence[str]) -> Optional[Notation]:
        """Mint a private symbol for each concept. The mapping is hers, not English renamed."""
        try:
            name = str(name).strip()
            live = [str(c).strip().lower() for c in concepts if str(c).strip()]
            if not name or not live:
                return None
            if len(self.notations) >= self.max_notations:
                self.notations.pop(next(iter(self.notations)))
            symbols: Dict[str, str] = {}
            for index, concept in enumerate(dict.fromkeys(live)):
                if index >= len(_GLYPHS):
                    break
                symbols[concept] = _GLYPHS[index]
            notation = Notation(name=name, symbols=symbols, operators=dict(_OPS))
            self.notations[name] = notation
            return notation
        except Exception:  # noqa: BLE001
            return None

    # ---- 2 & 3. express and realise ---------------------------------------- #
    def express(self, notation: Any, expression: str, *,
                bindings: Optional[Dict[str, float]] = None) -> Statement:
        """Say something in her notation, then make it run. A notation that cannot run is refused."""
        out = Statement(source=str(expression or ""))
        try:
            notation = (self.notations.get(str(notation))
                        if not isinstance(notation, Notation) else notation)
            if notation is None:
                out.reason = "no such notation"
                return out
            out.notation = notation.name

            ast = self._parse(out.source)
            if ast is None:
                out.reason = "the statement could not be parsed"
                return out

            out.written = self._render(ast, notation)
            if self.translate_always:
                out.translated = self._render(ast, None)

            realised = self._realise(ast, bindings or {})
            if realised is None:
                out.reason = ("it did not compile and run — a notation that cannot be executed "
                              "is not a language")
                return out
            out.program, out.value, out.realised = realised[0], realised[1], True
            self.statements.append(out)
            del self.statements[:-32]
            return out
        except Exception:  # noqa: BLE001 — a failed statement says nothing, it does not crash
            out.reason = "the statement was abandoned"
            return out

    def _realise(self, ast: Any, bindings: Dict[str, float]) -> Optional[Tuple[List[Any], float]]:
        """Lower to bytecode and execute. Free variables are substituted first, or it cannot run."""
        try:
            grounded = self._bind(ast, {str(k).lower(): float(v) for k, v in bindings.items()})
            if grounded is None:
                return None
            if self._compiler is None:
                from nyxara.nyx5.retarget import OntologicalCompiler
                self._compiler = OntologicalCompiler(max_steps=self.max_vm_steps)
            result = self._compiler.build(grounded)
            if not getattr(result, "validated", False):
                return None
            return list(getattr(result, "program", [])), float(getattr(result, "emulated", 0.0))
        except Exception:  # noqa: BLE001
            return None

    # ---- 4. translate ------------------------------------------------------ #
    def translate(self, statement: Statement) -> str:
        """Whatever reaches a person is a translation. This makes that explicit."""
        try:
            if statement.translated:
                return f"{statement.translated}   [lossy projection of {statement.written}]"
            return statement.source
        except Exception:  # noqa: BLE001
            return statement.source

    # ---- alternative logics ------------------------------------------------- #
    def alternative_axioms(self, name: str) -> Any:
        """Forge a system she may reason in that classical logic does not permit."""
        try:
            if self._axioms is None:
                from nyxara.nyx5.axiom_forge import SubAxiomaticEngine
                self._axioms = SubAxiomaticEngine()
            return self._axioms.forge(str(name))
        except Exception:  # noqa: BLE001
            return None

    def holds(self, system: Any, expression: Any) -> Any:
        """Ask what a statement's truth value is *in that system* — which may be BOTH."""
        try:
            if self._axioms is None or system is None:
                return None
            return self._axioms.evaluate(system, expression)
        except Exception:  # noqa: BLE001
            return None

    # ---- parsing and rendering ---------------------------------------------- #
    @staticmethod
    def _parse(text: str) -> Any:
        """A small recursive-descent parser for the arithmetic her VM can actually execute."""
        try:
            tokens = _TOKEN.findall(str(text or ""))
            if not tokens:
                return None
            position = 0

            def peek() -> Optional[str]:
                return tokens[position] if position < len(tokens) else None

            def take() -> Optional[str]:
                nonlocal position
                token = peek()
                position += 1
                return token

            def atom() -> Any:
                token = take()
                if token is None:
                    raise ValueError("unexpected end")
                if token == "(":
                    inner = expr()
                    if take() != ")":
                        raise ValueError("unbalanced")
                    return inner
                if token == "-":
                    return ("neg", atom())
                try:
                    return float(token)
                except ValueError:
                    return str(token).lower()

            def term() -> Any:
                node = atom()
                while peek() in ("*", "/"):
                    node = (take(), node, atom())
                return node

            def expr() -> Any:
                node = term()
                while peek() in ("+", "-"):
                    node = (take(), node, term())
                return node

            tree = expr()
            return tree if position >= len(tokens) else None
        except Exception:  # noqa: BLE001
            return None

    def _bind(self, node: Any, bindings: Dict[str, float]) -> Any:
        """Replace free variables with their values; without them nothing can be executed."""
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, str):
            value = bindings.get(node)
            return None if value is None else float(value)
        if isinstance(node, tuple):
            if node[0] == "neg":
                inner = self._bind(node[1], bindings)
                return None if inner is None else ("neg", inner)
            left = self._bind(node[1], bindings)
            right = self._bind(node[2], bindings)
            if left is None or right is None:
                return None
            return (node[0], left, right)
        return None

    def _render(self, node: Any, notation: Optional[Notation]) -> str:
        """Render the tree — in her symbols when a notation is given, in words when not."""
        try:
            if isinstance(node, (int, float)):
                return f"{float(node):g}"
            if isinstance(node, str):
                return notation.symbols.get(node, node) if notation else node
            if isinstance(node, tuple):
                if node[0] == "neg":
                    sign = notation.operators.get("-", "-") if notation else "-"
                    return f"{sign}{self._render(node[1], notation)}"
                sign = notation.operators.get(node[0], node[0]) if notation else node[0]
                left = self._render(node[1], notation)
                right = self._render(node[2], notation)
                return f"({left} {sign} {right})"
            return str(node)
        except Exception:  # noqa: BLE001
            return str(node)

    # ---- introspection ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            return {"notations": {n.name: len(n.symbols) for n in self.notations.values()},
                    "statements": len(self.statements),
                    "realised": sum(1 for s in self.statements if s.realised),
                    "translate_always": self.translate_always,
                    "last": self.statements[-1].to_dict() if self.statements else None}
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {"notations": [n.to_dict() for n in self.notations.values()]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.notations = {}
            for raw in data.get("notations", []) or []:
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("name", "")).strip()
                if not name:
                    continue
                self.notations[name] = Notation(
                    name=name,
                    symbols={str(k): str(v) for k, v in (raw.get("symbols") or {}).items()},
                    operators={str(k): str(v) for k, v in (raw.get("operators") or {}).items()})
            return True
        except Exception:  # noqa: BLE001
            return False
