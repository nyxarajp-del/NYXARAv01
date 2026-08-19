"""NYXARA · njp/cortex.py — the teacher organ, inside NJP rather than beside it (🎓, NJP V.07).

The Qwythos weights are served here, in ``njp/``, as one of her organs — not in ``mind/`` as an
external service she calls. The distinction is not filing: an organ is built by
:class:`~nyxara.njp.brain.NJPBrain`, reports through :meth:`stats` beside the fabric and the
grounder, and is *absent* rather than zero when it is off. That is the contract the rest of this
package keeps, and weights that sit outside it would be a second system to reason about.

**What it is for, and what it is emphatically not.** It is a **teacher**.
:mod:`nyxara.njp.shadow` runs it beside her own cognition and measures where they differ;
:mod:`nyxara.njp.adversary` challenges anything it asserts; and only independent evidence
establishes anything. Its authority is zero and its convenience is high, which is exactly the
right shape for a proposer. The measurement that decides whether any of it worked is
:mod:`nyxara.eval.teacher_removal`, which takes it away again and asks whether she still performs.

**Three things it inherits from the rung it is modelled on** (``mind/llm.py::LiteRTLMProvider``),
each of which was learned the hard way there:

* **One process, one set of weights.** The engine cache is process-wide and keyed by
  ``(path, gpu_layers, context)``. Per-instance caching means a second organ — a warm-up, a second
  facade, a tool that builds its own — loads a second full copy of a 5.6 GB file. The key includes
  the backend because a CPU engine handed to a caller that asked for GPU answers *silently*.
* **Availability must be cheap.** :meth:`available` is called on every turn, so it checks a flag,
  a file and an import, and never loads an engine or touches the network.
* **A dead runtime takes itself off the ladder.** Once ``llama_cpp`` has proved it cannot start on
  this host, ``_dead`` latches and the organ reports absent instead of failing — and logging —
  once per turn forever.

**On ``<think>``.** Every Qwythos answer opens with a reasoning block. It is stripped from what the
Master reads and **kept on the response**, because :mod:`nyxara.njp.genome` mines exactly those
traces for recurring reasoning shapes. Stripping is a rendering decision; discarding the trace
would throw away the most useful thing the teacher produces.

Fail-soft throughout: every entry point degrades to a null result rather than breaking a turn.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["TeacherReply", "Cortex"]

#: A complete reasoning block, non-greedy and DOTALL.
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
#: A reply the token budget cut off *inside* the reasoning: an opening tag and no closing one.
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
#: Everything up to a closing tag that never had an opening one — **the shape this model actually
#: produces**. Qwen3.5's chat template opens the reasoning block in the PROMPT, so generation
#: starts already inside it and the model only ever emits ``</think>``. Measured against the real
#: weights: ``contains <think>: False | contains </think>: True``. A splitter that requires the
#: opening tag matches nothing here, hands the Master the entire raw monologue as the answer, and
#: leaves njp/genome.py with no trace to mine — both halves of the design failing silently while
#: every field looks populated.
_THINK_CLOSE_ONLY = re.compile(r"\A(?P<reasoning>.*?)</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass
class TeacherReply:
    """One answer from the teacher: what to show, and the reasoning behind it, kept apart."""

    text: str = ""                 # the answer, <think> removed
    raw: str = ""                  # everything the model emitted, reasoning included
    reasoning: str = ""            # just the <think> content — what njp/genome.py mines
    truncated: bool = False        # the token budget cut it off mid-reasoning
    model: str = ""
    latency_s: float = 0.0
    tokens: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text[:400], "chars": len(self.text),
                "reasoning_chars": len(self.reasoning), "truncated": self.truncated,
                "model": self.model, "latency_s": round(self.latency_s, 3),
                "tokens": self.tokens}


def split_think(raw: str, *, preopened: bool = False) -> Tuple[str, str, bool]:
    """``(answer, reasoning, truncated)`` for one raw generation.

    The truncated case is the one worth being careful about. A reply cut off by the token budget
    part-way through its reasoning has ``<think>`` and no ``</think>``, and a naive strip leaves
    it untouched — so the Master receives the raw monologue as though it were the answer. Here
    that returns an empty answer and says ``truncated``, which is honest: there is no answer yet.
    """
    text = str(raw or "")
    reasoning_parts: List[str] = []
    truncated = False

    # Case 1: the template left the opening tag to the model — a complete <think>...</think>.
    reasoning_parts.extend(m.group(0) for m in _THINK.finditer(text))
    body = _THINK.sub("", text)

    # Case 2: the template opened the block itself, so only the closing tag is generated. This is
    # what the real weights do, and it has to be checked BEFORE the truncation case: an unmatched
    # </think> is a *complete* reasoning block whose opener was consumed, not a broken one.
    close_only = _THINK_CLOSE_ONLY.match(body)
    if close_only:
        reasoning_parts.append(close_only.group("reasoning"))
        body = body[close_only.end():]

    # Case 3: cut off by the token budget mid-reasoning — an opener with no closer. There is no
    # answer yet, and returning the monologue as one would be a lie about what happened.
    open_tail = _THINK_OPEN.search(body)
    if open_tail:
        reasoning_parts.append(open_tail.group(0))
        body = body[:open_tail.start()]
        truncated = True

    # Case 4: no tags at all. What that means depends entirely on the template, which is why
    # `preopened` is a declared fact rather than something inferred from the text.
    #
    # Under a pre-opened template — this model's — generation started inside the reasoning block,
    # so a reply carrying no closing tag never left it: the whole thing is reasoning and there is
    # NO ANSWER YET. Reading it as an answer is the original bug in a new disguise, and a worse
    # one, because there is no syntactic marker to notice. Measured on the first live shadow run:
    # three replies, three monologues returned as answers, `reasoning_chars = 0` on all of them.
    #
    # Under an ordinary template the same bytes are simply an answer with no reasoning.
    if preopened and not reasoning_parts and body.strip():
        reasoning_parts.append(body)
        body = ""
        truncated = True

    reasoning = "\n".join(
        re.sub(r"</?think>", "", p, flags=re.IGNORECASE).strip() for p in reasoning_parts)
    return body.strip(), reasoning.strip(), truncated


class Cortex:
    """The Qwythos weights, served in-process, as one of NJP's organs.

    Absent is a first-class state: with no ``llama_cpp`` and no weights on disk this constructs
    fine, reports ``present=False`` with the reason, and every call returns an empty reply. That
    is the normal condition on a machine where the 5.6 GB file was never fetched, and nothing
    downstream should have to special-case it.
    """

    def __init__(self, *, settings: Any = None, model_path: Any = None) -> None:
        self.settings = settings
        self._explicit_path = Path(model_path) if model_path else None
        self._lock = threading.Lock()
        #: Latched load failure. Once the runtime has proved it cannot start here, the organ
        #: reports absent rather than raising once per turn for the rest of the process.
        self._dead: Optional[str] = None
        self.calls = 0
        self.failures = 0
        self.tokens_out = 0

    # ---- config, with honest defaults when there are no settings ---------- #
    def _cfg(self, name: str, default: Any) -> Any:
        try:
            return getattr(self.settings.llm, name, default)
        except Exception:  # noqa: BLE001
            return default

    def model_path(self) -> Optional[Path]:
        if self._explicit_path is not None:
            return self._explicit_path
        try:
            from nyxara.mind.gguf_assets import model_path
            return model_path(self.settings)
        except Exception:  # noqa: BLE001
            return None

    # ---- availability: cheap, and never loads anything -------------------- #
    def binding(self) -> Any:
        """The ``llama_cpp`` module, or ``None`` when the optional wheel is not installed."""
        try:
            from nyxara.mind.gguf_assets import binding
            return binding()
        except Exception:  # noqa: BLE001
            return None

    def available(self) -> bool:
        return self.why_unavailable() is None

    def why_unavailable(self) -> Optional[str]:
        """``None`` when the organ can serve, else the reason — which is what gets reported."""
        try:
            from nyxara.mind.gguf_assets import engine_dead
            shared_dead = engine_dead()
        except Exception:  # noqa: BLE001
            shared_dead = None
        if self._dead or shared_dead:
            return self._dead or shared_dead
        if not bool(self._cfg("gguf_enabled", True)):
            return "disabled by config (llm.gguf_enabled=false)"
        path = self.model_path()
        if path is None:
            return "no model path could be resolved"
        if not path.is_file():
            return (f"weights missing at {path} "
                    f"(python scripts/fetch_qwythos_model.py)")
        if self.binding() is None:
            return "llama_cpp is not installed (pip install llama-cpp-python)"
        return None

    # ---- the engine ------------------------------------------------------- #
    def _engine(self) -> Any:
        """The shared llama.cpp engine from ``mind/gguf_assets``.

        Not a private one. These same weights are also the ladder's ``gguf`` rung
        (``mind/llm.py``) and the source ``njp/graft.py`` converts from; a cache per consumer
        would hold a 5.6-17.9 GB file in memory three times over for one copy on disk. The dead
        latch is shared for the same reason — a runtime that cannot start here cannot start there.
        """
        try:
            from nyxara.mind.gguf_assets import load_engine
            engine = load_engine(self.settings, path=self._explicit_path)
            if engine is None:
                from nyxara.mind.gguf_assets import engine_dead
                self._dead = self._dead or engine_dead()
            return engine
        except Exception as exc:  # noqa: BLE001
            self._dead = f"gguf engine unavailable: {exc}"
            return None

    # ---- asking the teacher ---------------------------------------------- #
    def ask(self, prompt: str, *, system: str = "", max_tokens: Optional[int] = None,
            temperature: Optional[float] = None) -> TeacherReply:
        """One question, one reply. Stateless: no conversation is kept anywhere here."""
        out = TeacherReply(model=str(self._cfg("gguf_model", "qwythos")))
        if not self.available():
            return out
        t0 = time.perf_counter()
        try:
            engine = self._engine()
            if engine is None:
                return out
            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": str(system)})
            messages.append({"role": "user", "content": str(prompt)})
            with self._lock:
                resp = engine.create_chat_completion(
                    messages=messages,
                    max_tokens=int(max_tokens if max_tokens is not None
                                   else self._cfg("gguf_max_output_tokens", 16384)),
                    temperature=float(temperature if temperature is not None
                                      else self._cfg("gguf_temperature", 0.6)),
                    top_p=float(self._cfg("gguf_top_p", 0.95)),
                    top_k=int(self._cfg("gguf_top_k", 20)),
                    repeat_penalty=float(self._cfg("gguf_repeat_penalty", 1.05)),
                )
            raw = str(resp["choices"][0]["message"]["content"] or "")
            out.raw = raw
            if bool(self._cfg("gguf_strip_think", True)):
                out.text, out.reasoning, out.truncated = split_think(
                    raw, preopened=bool(self._cfg("gguf_reasoning_preopened", False)))
            else:
                out.text = raw
            usage = resp.get("usage") or {}
            out.tokens = int(usage.get("completion_tokens", 0) or 0)
            self.calls += 1
            self.tokens_out += out.tokens
        except Exception:  # noqa: BLE001 — a failed ask is an empty reply, never a broken turn
            self.failures += 1
        out.latency_s = time.perf_counter() - t0
        return out

    # ---- reporting -------------------------------------------------------- #
    def stats(self) -> Optional[Dict[str, Any]]:
        """``None`` when the organ is off — *absent*, not a row of zeros.

        ``njp/__init__.py`` states the rule: "one that is off is absent from the reports rather
        than reporting zeros, which is the distinction this whole package keeps". A zeroed teacher
        row reads as a teacher that was consulted and said nothing.
        """
        reason = self.why_unavailable()
        if reason is not None and self.calls == 0:
            return {"present": False, "reason": reason,
                    "model": str(self._cfg("gguf_model", "qwythos")),
                    "path": str(self.model_path() or "")}
        return {
            "present": reason is None,
            "reason": reason,
            "model": str(self._cfg("gguf_model", "qwythos")),
            "path": str(self.model_path() or ""),
            "calls": self.calls, "failures": self.failures, "tokens_out": self.tokens_out,
            "context_tokens": int(self._cfg("gguf_context_tokens", 16384)),
            # Reported so the difference between what the weights CAN do and what this host is
            # actually configured for stays visible. The 1M window is real and it is not what is
            # allocated here.
            "context_ceiling": int(self._cfg("gguf_max_context_tokens", 1_048_576)),
            "on_device": True,
            # Stated on every report: this organ proposes. It is never the thing that decides.
            "authority": "proposal_only",
        }
