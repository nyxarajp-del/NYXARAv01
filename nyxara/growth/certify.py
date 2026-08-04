"""NYXARA · growth/certify.py — carrying the proof into the data she trains on (📜→🧠).

Everything upstream of the corpus builds machine-checkable witnesses. :class:`ProvenItem` has a
``certificate`` field. :meth:`RivalVerifier.verify` returns the proof string alongside its
verdict. :mod:`~nyxara.growth.meta_epistemology` writes ``TheoremCertificate`` objects to disk.

And then every one of them is **dropped at the corpus boundary**:

* ``oracle.ProvenItem.to_doc()`` renders ``prompt`` and ``answer`` — with ``self.certificate``
  in scope, unused.
* ``synthesis._feed_flywheel`` computes the proof string ``why``, uses it for one line of a
  report, and discards it.
* ``distill.DistillationExample.to_dict()`` has no certificate field at all.

So the model learns *the answer* and never *the reason*. It is trained on the output of a proof
system it has no evidence exists. This module is the two lines each of those paths was missing,
plus the thing that makes them safe: a bar for what may be called a certificate.

**The bar.** Only a result from a decision procedure may be embedded — ``ProofResult.certifiable``,
which is ``PROVEN and exact``. Schwartz–Zippel agreement at 24 random points is overwhelming
evidence and is *not* a proof; embedding it as one would teach the model that sampling and
proving are the same act, which is precisely the confusion this whole layer exists to remove.
Sampled results still travel — as evidence, labelled as evidence.

Z3 and sympy are already real dependencies here (``prover.py``, ``invariants.py``,
``causal/axiom_synth.py`` all use them, and ``requirements.txt`` pins ``z3-solver``). Lean 4 is
absent from the repo entirely; :func:`available_backends` reports what is actually installed
rather than what is aspired to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = [
    "PROOF_OPEN",
    "PROOF_CLOSE",
    "Certificate",
    "render_certified_doc",
    "extract_certificate",
    "certify_claim",
    "available_backends",
]

#: The shape a proof takes inside a training document. A paired tag, matching the existing
#: ``<tool>``/``<result>`` convention in :mod:`~nyxara.growth.synth_data`, so the answer span and
#: the proof span are both exactly delimited rather than found by string search — which is what
#: lets the SFT loss mask them separately later.
PROOF_OPEN = "<proof>"
PROOF_CLOSE = "</proof>"

_PROOF_RE = re.compile(re.escape(PROOF_OPEN) + r"(.*?)" + re.escape(PROOF_CLOSE), re.S)
_MAX_CERTIFICATE_CHARS = 600


@dataclass
class Certificate:
    """A witness that may be shown to the model, with the method that produced it."""

    statement: str
    witness: str
    method: str = ""
    #: True only for a decision procedure. Sampling-based agreement sets this False and is
    #: rendered as evidence rather than as proof.
    exact: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def embeddable(self) -> bool:
        return bool(self.witness.strip()) and self.exact

    def render(self) -> str:
        """The proof block as it appears in a training document."""
        witness = self.witness.strip()
        if len(witness) > _MAX_CERTIFICATE_CHARS:
            witness = witness[:_MAX_CERTIFICATE_CHARS - 1].rstrip() + "…"
        head = self.method.strip()
        body = f"{head}: {witness}" if head else witness
        return f"{PROOF_OPEN}{body}{PROOF_CLOSE}"

    def to_dict(self) -> Dict[str, Any]:
        return {"statement": self.statement, "witness": self.witness, "method": self.method,
                "exact": bool(self.exact), "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Certificate":
        return cls(statement=str(d.get("statement", "")), witness=str(d.get("witness", "")),
                   method=str(d.get("method", "")), exact=bool(d.get("exact", True)),
                   meta=dict(d.get("meta") or {}))

    @classmethod
    def from_proof_result(cls, result: Any, *, statement: str = "") -> Optional["Certificate"]:
        """Adapt a :class:`~nyxara.growth.prover.ProofResult`. ``None`` when it proved nothing."""
        if result is None or not getattr(result, "proven", False):
            return None
        claim = getattr(result, "claim", None)
        return cls(statement=statement or getattr(claim, "statement", "") or "",
                   witness=str(getattr(result, "certificate", "") or ""),
                   method=str(getattr(result, "method", "") or ""),
                   # `exact` distinguishes a decision procedure from sampling. Defaulting to
                   # True on an object that predates the field would silently promote sampled
                   # agreement to proof, so absence is treated as "not exact".
                   exact=bool(getattr(result, "exact", False)))


def render_certified_doc(prompt: str, answer: str,
                         certificate: Optional[Certificate] = None, *,
                         system: Optional[str] = None) -> str:
    """Render a supervised pair in her own template, with the proof attached to the answer.

    The proof goes *inside* the assistant span, after the answer. It is part of what she should
    learn to produce — a model that emits the answer and the witness together is one whose
    reasoning can be checked by the reader, which is the entire point of generating this data
    from a prover in the first place.
    """
    body = answer if not (certificate and certificate.embeddable) else \
        f"{answer.rstrip()}\n{certificate.render()}"
    try:
        from nyxara.mind.llm import format_self_training_doc

        try:
            return format_self_training_doc(prompt, body, system=system)
        except TypeError:      # older signature without `system`
            return format_self_training_doc(prompt, body)
    except Exception:  # noqa: BLE001 — a plain rendering beats losing the pair
        return f"{prompt}\n\n{body}\n"


def extract_certificate(doc: str) -> Optional[str]:
    """The proof block's contents, or ``None``. Used by the eval battery and by tests."""
    match = _PROOF_RE.search(doc or "")
    return match.group(1).strip() if match else None


def certify_claim(statement: str, *, kind: str = "", candidate: Optional[str] = None,
                  prover: Any = None) -> Optional[Certificate]:
    """Run ``statement`` through the real prover and return an embeddable certificate.

    ``None`` when the claim could not be *decided* — abstention, a refutation, or a merely
    sampled agreement all return nothing rather than a weaker certificate, because the caller's
    next act is to write this into training data.
    """
    try:
        from nyxara.growth.prover import ProofClaim, Prover
    except Exception:  # noqa: BLE001
        return None
    engine = prover or Prover()
    try:
        result = engine.prove(ProofClaim(kind=kind, statement=statement,
                                         candidate_answer=candidate))
    except Exception:  # noqa: BLE001 — an erroring prover certifies nothing
        return None
    cert = Certificate.from_proof_result(result, statement=statement)
    return cert if (cert and cert.embeddable) else None


def available_backends() -> Dict[str, bool]:
    """What is actually installed, so the dataset card can state it rather than assume it."""
    backends: Dict[str, bool] = {}
    for name, module in (("z3", "z3"), ("sympy", "sympy")):
        try:
            __import__(module)
            backends[name] = True
        except Exception:  # noqa: BLE001
            backends[name] = False
    # Lean has no Python binding; a toolchain is the only evidence it exists.
    import shutil

    backends["lean4"] = bool(shutil.which("lean") or shutil.which("lake"))
    return backends
