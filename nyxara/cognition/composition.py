"""NYXARA · cognition/composition.py — compositional generalization (⧉).

The signature of a *systematic* mind: having learned the primitives ``jump``, ``walk`` and
the modifier ``twice`` **separately**, it can interpret ``jump twice`` and ``walk twice`` on
first sight — combinations it was never taught. A model that only memorises whole phrases
cannot; one that composes *meanings* can. NYXARA lacked this; here it is, for real.

Meaning lives in the Language of Thought (:mod:`nyxara.mind.lot`): each token maps to a LoT
term, and modifiers map to **higher-order** terms (``twice = λa. seq(a, a)``). Interpretation
is genuine semantic composition — function application plus **β-reduction** (``reduce``) — so a
phrase's meaning is *computed* from its parts and their arrangement, not retrieved. Novel
combinations of known primitives therefore interpret correctly by construction.

The lexicon is learnable one entry at a time (:meth:`CompositionalGrammar.learn`) and persists
through the :class:`~nyxara.memory.store.MemoryStore` (tag ``lexeme``) so primitives learned
once survive restarts. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.mind.lot import (
    And, App, Const, Function, Lambda, Not, Or, Predicate, Term, Var, reduce, var,
)

__all__ = ["CompositionalGrammar", "Interpretation", "serialize_term", "deserialize_term"]


# --------------------------------------------------------------------------- #
# Term (de)serialization for the subset used in lexicon meanings
# --------------------------------------------------------------------------- #
def serialize_term(t: Term) -> Dict[str, Any]:
    """Compact, reversible JSON form for the LoT subset used as lexeme meanings."""
    if isinstance(t, Var):
        return {"k": "Var", "name": t.name, "type": t.type}
    if isinstance(t, Const):
        return {"k": "Const", "name": t.name, "type": t.type}
    if isinstance(t, Predicate):
        return {"k": "Predicate", "name": t.name, "args": [serialize_term(a) for a in t.args]}
    if isinstance(t, Function):
        return {"k": "Function", "name": t.name, "args": [serialize_term(a) for a in t.args]}
    if isinstance(t, And):
        return {"k": "And", "args": [serialize_term(a) for a in t.args]}
    if isinstance(t, Or):
        return {"k": "Or", "args": [serialize_term(a) for a in t.args]}
    if isinstance(t, Not):
        return {"k": "Not", "arg": serialize_term(t.arg)}
    if isinstance(t, Lambda):
        return {"k": "Lambda", "var": serialize_term(t.var), "body": serialize_term(t.body)}
    if isinstance(t, App):
        return {"k": "App", "func": serialize_term(t.func), "arg": serialize_term(t.arg)}
    raise TypeError(f"cannot serialize term of type {type(t).__name__}")


def deserialize_term(d: Dict[str, Any]) -> Term:
    """Inverse of :func:`serialize_term`."""
    k = d.get("k")
    if k == "Var":
        return Var(d["name"], d.get("type", "e"))
    if k == "Const":
        return Const(d["name"], d.get("type", "e"))
    if k == "Predicate":
        return Predicate(d["name"], tuple(deserialize_term(a) for a in d.get("args", [])))
    if k == "Function":
        return Function(d["name"], tuple(deserialize_term(a) for a in d.get("args", [])))
    if k == "And":
        return And(tuple(deserialize_term(a) for a in d.get("args", [])))
    if k == "Or":
        return Or(tuple(deserialize_term(a) for a in d.get("args", [])))
    if k == "Not":
        return Not(deserialize_term(d["arg"]))
    if k == "Lambda":
        v = deserialize_term(d["var"])
        assert isinstance(v, Var)
        return Lambda(v, deserialize_term(d["body"]))
    if k == "App":
        return App(deserialize_term(d["func"]), deserialize_term(d["arg"]))
    raise TypeError(f"cannot deserialize kind {k!r}")


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class Interpretation:
    phrase: str
    meaning: Optional[Term]
    unknown: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.meaning is not None and not self.unknown

    def __str__(self) -> str:
        if self.unknown:
            return f"<unknown primitives: {', '.join(self.unknown)}>"
        return str(self.meaning)


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #
def _tokenize(phrase: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9_]+", str(phrase).lower())]


class CompositionalGrammar:
    """A learnable lexicon + a compositional interpreter.

    Tokens map to LoT meanings. Interpretation composes adjacent meanings by applying any
    higher-order (``Lambda``) meaning to its neighbour and β-reducing; remaining unmodified
    meanings are sequenced. The result is the phrase's computed meaning — *systematic*: novel
    combinations of known primitives interpret without ever having been seen together.
    """

    TAG = "lexeme"

    def __init__(self, *, store: Any = None) -> None:
        self.store = store
        self.lexicon: Dict[str, Term] = {}
        self._hydrated = False

    # ---- learn ---- #
    def learn(self, token: str, meaning: Term) -> Term:
        """Bind a single token to a LoT meaning (a primitive, or a higher-order modifier)."""
        self._hydrate()
        token = str(token).strip().lower()
        if not token or not isinstance(meaning, Term):
            raise ValueError("learn(token, meaning: Term) needs a non-empty token and a LoT term")
        self.lexicon[token] = meaning
        self._persist(token, meaning)
        return meaning

    def known(self, token: str) -> bool:
        self._hydrate()
        return str(token).strip().lower() in self.lexicon

    def vocabulary(self) -> List[str]:
        self._hydrate()
        return sorted(self.lexicon)

    # ---- interpret ---- #
    def interpret(self, phrase: str) -> Interpretation:
        """Compose the meanings of a phrase's tokens into a normal-form LoT meaning."""
        self._hydrate()
        tokens = _tokenize(phrase)
        unknown = [t for t in tokens if t not in self.lexicon]
        if unknown or not tokens:
            return Interpretation(phrase=str(phrase), meaning=None, unknown=unknown)

        terms: List[Term] = [self.lexicon[t] for t in tokens]
        terms = self._compose(terms)
        # any leftover unmodified meanings are sequenced (implicit conjunction of actions)
        meaning = terms[0] if len(terms) == 1 else Function("seq", tuple(terms))
        return Interpretation(phrase=str(phrase), meaning=reduce(meaning))

    def _compose(self, terms: List[Term]) -> List[Term]:
        """Repeatedly apply a Lambda meaning to a non-Lambda neighbour, β-reducing each time.

        A postfix modifier (``jump twice``) applies to its left neighbour; a prefix modifier
        (``twice jump``) to its right — so word order is respected either way."""
        terms = list(terms)
        progressed = True
        while progressed and len(terms) > 1:
            progressed = False
            for i, t in enumerate(terms):
                if not isinstance(t, Lambda):
                    continue
                if i - 1 >= 0 and not isinstance(terms[i - 1], Lambda):
                    combined = reduce(App(t, terms[i - 1]))
                    terms = terms[:i - 1] + [combined] + terms[i + 1:]
                    progressed = True
                    break
                if i + 1 < len(terms) and not isinstance(terms[i + 1], Lambda):
                    combined = reduce(App(t, terms[i + 1]))
                    terms = terms[:i] + [combined] + terms[i + 2:]
                    progressed = True
                    break
        return terms

    # ---- persistence ---- #
    def _persist(self, token: str, meaning: Term) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                f"lexeme:{token} = {meaning}", mem_type=MemoryType.SEMANTIC, importance=0.6,
                tags=[self.TAG],
                metadata={"token": token, "meaning": serialize_term(meaning)})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is None:
            return
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            for rec, _ in hits:
                meta = rec.metadata if isinstance(rec.metadata, dict) else {}
                token = meta.get("token")
                payload = meta.get("meaning")
                if token and isinstance(payload, dict) and token not in self.lexicon:
                    try:
                        self.lexicon[token] = deserialize_term(payload)
                    except Exception:  # noqa: BLE001 — skip an unparseable lexeme
                        pass
        except Exception:  # noqa: BLE001
            pass

    def __len__(self) -> int:
        self._hydrate()
        return len(self.lexicon)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.lot import const, func

    print("=" * 70)
    print("NYXARA compositional generalization self-test")
    print("=" * 70)

    g = CompositionalGrammar()
    # teach PRIMITIVES separately — actions as constants, modifiers as higher-order λ-terms
    g.learn("jump", const("JUMP"))
    g.learn("walk", const("WALK"))
    g.learn("turn", const("TURN"))
    g.learn("twice", Lambda(var("a"), func("seq", var("a"), var("a"))))
    g.learn("thrice", Lambda(var("a"), func("seq", var("a"), func("seq", var("a"), var("a")))))
    print(f"\nvocabulary          : {g.vocabulary()}")

    # single primitives
    assert str(g.interpret("jump")) == "JUMP"

    # THE ACID TEST — combinations never taught, interpreted by composition
    cases = {
        "jump twice": "seq(JUMP, JUMP)",
        "walk twice": "seq(WALK, WALK)",
        "turn thrice": "seq(TURN, seq(TURN, TURN))",
    }
    for phrase, want in cases.items():
        got = str(g.interpret(phrase))
        print(f"  interpret {phrase:14s} -> {got:24s} {'✓' if got == want else '✗ want ' + want}")
        assert got == want, f"{phrase}: got {got}, want {want}"
    print("novel combinations  : interpreted correctly though never taught ✓")

    # prefix modifier order also works, and modifiers nest (higher-order productivity)
    nested = str(g.interpret("jump twice twice"))
    print(f"\nnested modifier     : jump twice twice -> {nested}")
    assert nested == "seq(seq(JUMP, JUMP), seq(JUMP, JUMP))"

    # sequencing of bare primitives (implicit conjunction)
    seqd = str(g.interpret("jump walk"))
    print(f"sequence            : jump walk -> {seqd}")
    assert seqd == "seq(JUMP, WALK)"

    # honesty: an unknown primitive is reported, not hallucinated
    unk = g.interpret("jump sprint")
    print(f"unknown primitive   : {unk}")
    assert not unk.ok and "sprint" in unk.unknown

    # persistence: a fresh grammar over the same store recovers the lexicon
    from nyxara.memory.store import MemoryStore
    store = MemoryStore()
    g2 = CompositionalGrammar(store=store)
    g2.learn("dance", const("DANCE"))
    g2.learn("twice", Lambda(var("a"), func("seq", var("a"), var("a"))))
    g3 = CompositionalGrammar(store=store)
    assert str(g3.interpret("dance twice")) == "seq(DANCE, DANCE)"
    print(f"\npersisted lexicon   : dance twice -> {g3.interpret('dance twice')} ✓")

    print("\nALL SELF-TESTS PASSED ✓")
