"""NYXARA · observe/ — transparency: her thinking, legible from outside (🔍, Rule 6).

A sovereign mind the Master cannot inspect is not sovereign, it is opaque. This package is
how NYXARA stays *readable*:

:class:`~nyxara.observe.mindscope.MindScope` is the trace of her cognition — every
:class:`~nyxara.observe.mindscope.Thought` recorded with its kind, salience and **causes**, so
``explain_last()`` can walk backwards from a decision to the perception that provoked it.

:class:`~nyxara.observe.honesty.HonestyGuard` checks what she is about to say against what she
actually knows, flagging unsupported claims and mis-calibrated confidence before they reach
the Master. Calibration is learned from lived outcomes (``record_outcome``), not asserted.

:class:`~nyxara.observe.self_report.SelfReporter` composes the calibrated status report — what
she did, what she is unsure of, and what she got wrong — the same object behind ``/report``.

These are read paths. Nothing here decides anything; the kernel disposes and the Master is
sovereign. Their whole job is to make both of those auditable.
"""

from __future__ import annotations

from nyxara.observe.honesty import Claim, HonestyGuard, HonestyIssue, HonestyVerdict
from nyxara.observe.mindscope import MindScope, Thought, ThoughtKind
from nyxara.observe.self_report import ReportKind, ReportSection, SelfReport, SelfReporter

__all__ = [
    # the trace of cognition
    "MindScope", "Thought", "ThoughtKind",
    # calibrated honesty
    "HonestyGuard", "HonestyVerdict", "HonestyIssue", "Claim",
    # the status report
    "SelfReporter", "SelfReport", "ReportSection", "ReportKind",
]
