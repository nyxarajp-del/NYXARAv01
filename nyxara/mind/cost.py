"""NYXARA · mind/cost.py — the LLM token & spend ledger (💸).

Every call to the stateless LLM faculty (mind/llm.py) burns tokens, and tokens cost
money. This module is the **ledger** that turns each :class:`~nyxara.mind.llm.Usage`
into a costed :class:`LineItem`, aggregates spend by model and provider, and answers
the one question the governor cares about: *are we over today's budget?*

Pricing lives in :data:`PRICES` — ``model (or prefix) -> (usd_per_1k_prompt,
usd_per_1k_completion)`` — matched by **longest prefix**, so ``claude-opus-4-8`` is
priced by the most specific ``claude-opus`` entry and unknown models fall back to a
conservative default rather than silently costing zero.

The ledger is **thread-safe** (a single lock) because the event loop and its
background worker tasks (kernel/jobqueue.py) may record concurrently. It carries no
LLM logic of its own — it merely consumes the dataclasses mind/llm.py already emits.

Depends on :mod:`nyxara.kernel.config`; consumes :mod:`nyxara.mind.llm` types
structurally (duck-typed), never importing heavy provider SDKs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from nyxara.kernel.config import get_settings

__all__ = [
    "PRICES",
    "DEFAULT_PRICE",
    "LineItem",
    "UsageLedger",
]

# --------------------------------------------------------------------------- #
# Pricing table — USD per 1K tokens, (prompt, completion).
# Matched by LONGEST PREFIX against the model name. Order is irrelevant.
# Figures are representative list prices for budgeting, not billing-of-record.
# --------------------------------------------------------------------------- #
# Values are USD per 1K tokens (prompt, completion); estimate_cost divides token
# counts by 1000, so these stay as the headline per-1K rates.
PRICES: Dict[str, Tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude": (3.0, 15.0),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5": (0.50, 1.50),
    # Groq open-weight (very cheap)
    "openai/gpt-oss-120b": (0.15, 0.75),
    "llama": (0.10, 0.10),
    # In-process / local / mock models cost nothing to run on our own hardware.
    "qwen": (0.0, 0.0),
    "local": (0.0, 0.0),
    "nyxara-self": (0.0, 0.0),
    "mock": (0.0, 0.0),
}

# Conservative fallback for an unrecognised model (USD per 1K tokens).
DEFAULT_PRICE: Tuple[float, float] = (1.0, 2.0)


# --------------------------------------------------------------------------- #
# Line item
# --------------------------------------------------------------------------- #
@dataclass
class LineItem:
    """One recorded LLM call and its computed cost."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    ts: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def day(self) -> date:
        return datetime.fromtimestamp(self.ts).date()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "ts": self.ts,
        }


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class UsageLedger:
    """Thread-safe ledger of LLM token usage and spend, with a daily budget."""

    def __init__(self, *, daily_budget: Optional[float] = None) -> None:
        if daily_budget is None:
            try:
                daily_budget = get_settings().resources.daily_spend_budget
            except Exception:  # noqa: BLE001 - config must never break accounting
                daily_budget = 0.0
        self.daily_budget = float(daily_budget)
        self._items: List[LineItem] = []
        self._lock = threading.Lock()

    # ---- pricing helpers ---- #
    @staticmethod
    def price_for(model: str) -> Tuple[float, float]:
        """Return (usd_per_1k_prompt, usd_per_1k_completion) by longest-prefix match."""
        best: Optional[str] = None
        for key in PRICES:
            if model.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        return PRICES[best] if best is not None else DEFAULT_PRICE

    @staticmethod
    def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost in USD for a hypothetical call without recording it."""
        p_rate, c_rate = UsageLedger.price_for(model)
        return (prompt_tokens / 1000.0) * p_rate + (completion_tokens / 1000.0) * c_rate

    # ---- recording ---- #
    def record(
        self,
        resp_or_usage: Any,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> LineItem:
        """Record a call.

        Accepts either an ``LLMResponse`` (provider/model/usage are pulled from it)
        or a raw ``Usage`` (in which case ``provider`` and ``model`` should be given).
        Explicit ``provider``/``model`` kwargs always override what's on the object.
        """
        usage = getattr(resp_or_usage, "usage", None)
        if usage is None:
            usage = resp_or_usage  # treat the object itself as a Usage-like

        prov = provider or getattr(resp_or_usage, "provider", None) or "unknown"
        mdl = model or getattr(resp_or_usage, "model", None) or "unknown"

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        cost = self.estimate_cost(mdl, prompt_tokens, completion_tokens)
        item = LineItem(
            provider=prov,
            model=mdl,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self._items.append(item)
        return item

    # ---- aggregation ---- #
    @property
    def items(self) -> List[LineItem]:
        with self._lock:
            return list(self._items)

    def total_cost(self) -> float:
        with self._lock:
            return sum(i.cost_usd for i in self._items)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(i.total_tokens for i in self._items)

    def by_model(self) -> Dict[str, Dict[str, float]]:
        return self._aggregate(lambda i: i.model)

    def by_provider(self) -> Dict[str, Dict[str, float]]:
        return self._aggregate(lambda i: i.provider)

    def _aggregate(self, key) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        with self._lock:
            items = list(self._items)
        for i in items:
            k = key(i)
            bucket = out.setdefault(k, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            bucket["calls"] += 1
            bucket["tokens"] += i.total_tokens
            bucket["cost_usd"] += i.cost_usd
        for bucket in out.values():
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        return out

    # ---- budget ---- #
    def today_cost(self, *, on: Optional[date] = None) -> float:
        on = on or date.today()
        with self._lock:
            return sum(i.cost_usd for i in self._items if i.day == on)

    def remaining(self, *, on: Optional[date] = None) -> float:
        """Budget left for today (may go negative if overspent)."""
        return self.daily_budget - self.today_cost(on=on)

    def over_budget(self, *, on: Optional[date] = None) -> bool:
        if self.daily_budget <= 0:
            return False
        return self.today_cost(on=on) > self.daily_budget

    # ---- reporting ---- #
    def summary(self) -> str:
        with self._lock:
            n = len(self._items)
        lines = [
            f"UsageLedger: {n} call(s), {self.total_tokens()} tokens, "
            f"${self.total_cost():.4f} total  (budget ${self.daily_budget:.2f}/day, "
            f"${self.remaining():.4f} left today)",
        ]
        for model, b in sorted(self.by_model().items()):
            lines.append(
                f"  {model:<28} {int(b['calls']):>4} calls  "
                f"{int(b['tokens']):>8} tok  ${b['cost_usd']:.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost(), 6),
            "total_tokens": self.total_tokens(),
            "daily_budget": self.daily_budget,
            "today_cost_usd": round(self.today_cost(), 6),
            "remaining_usd": round(self.remaining(), 6),
            "over_budget": self.over_budget(),
            "by_model": self.by_model(),
            "by_provider": self.by_provider(),
            "items": [i.to_dict() for i in self.items],
        }


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA cost ledger self-test")
    print("=" * 70)

    # Minimal Usage-like / LLMResponse-like stand-ins (no heavy imports needed).
    @dataclass
    class _Usage:
        prompt_tokens: int
        completion_tokens: int

    @dataclass
    class _Resp:
        provider: str
        model: str
        usage: _Usage

    ledger = UsageLedger(daily_budget=10.0)

    # record from an LLMResponse-like object
    r = _Resp("openai", "gpt-4o", _Usage(1000, 1000))
    item = ledger.record(r)
    print(f"gpt-4o 1k/1k cost    : ${item.cost_usd:.4f}")
    # gpt-4o: 2.50/1k prompt + 10.0/1k completion -> 2.50 + 10.0 = 12.50
    assert abs(item.cost_usd - 12.50) < 1e-9

    # record from a raw Usage (+ explicit provider/model)
    item2 = ledger.record(_Usage(2000, 500), provider="anthropic", model="claude-opus-4-8")
    # longest-prefix "claude-opus": 15/1k prompt + 75/1k completion -> 30 + 37.5 = 67.5
    print(f"claude-opus cost     : ${item2.cost_usd:.4f}")
    assert abs(item2.cost_usd - 67.5) < 1e-9
    assert item2.model == "claude-opus-4-8"

    # longest-prefix specificity: gpt-4o-mini must NOT use the gpt-4o rate
    mini = UsageLedger.estimate_cost("gpt-4o-mini", 1000, 1000)
    assert abs(mini - (0.15 + 0.60)) < 1e-9
    print(f"longest-prefix match : gpt-4o-mini 1k/1k = ${mini:.4f} (not gpt-4o) ✓")

    # unknown model -> fallback price (not zero)
    unknown = UsageLedger.estimate_cost("some-mystery-model", 1000, 0)
    assert unknown > 0
    print(f"fallback price       : unknown model 1k prompt = ${unknown:.4f}")

    # aggregation
    bm = ledger.by_model()
    bp = ledger.by_provider()
    assert "gpt-4o" in bm and "claude-opus-4-8" in bm
    assert set(bp) == {"openai", "anthropic"}
    print(f"by_provider          : { {k: round(v['cost_usd'], 2) for k, v in bp.items()} }")

    # totals
    assert ledger.total_tokens() == 1000 + 1000 + 2000 + 500
    assert abs(ledger.total_cost() - (12.50 + 67.5)) < 1e-9

    # budget: $10/day, we've spent ~$80 -> over budget, negative remaining
    assert ledger.over_budget() is True
    assert ledger.remaining() < 0
    print(f"budget               : over={ledger.over_budget()}  remaining=${ledger.remaining():.2f}")

    # a free local model adds tokens but no cost
    ledger.record(_Usage(5000, 5000), provider="local", model="qwen")
    assert UsageLedger.estimate_cost("qwen", 5000, 5000) == 0.0
    print("free local model     : tokens counted, $0 cost ✓")

    print("\n" + ledger.summary())
    print("\nALL SELF-TESTS PASSED ✓")
