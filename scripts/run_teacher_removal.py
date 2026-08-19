"""The A/B/C measurement, against the live 9B.

    python scripts/run_teacher_removal.py out.json

KNOWN LIMITATION of this driver, recorded because the first run was made with it: all three
arms share ONE NJPBrain, and she learns between turns — the fabric expands and memory
accumulates. Arm C is therefore not "NJP before distillation" but "NJP after seeing the whole
battery twice", so a C > A result from this script cannot be attributed to distillation rather
than to within-run learning.

It did not bite on the first run: C matched A exactly, 0 gained and 0 lost, so within-run
learning changed nothing on this battery. It would bite the moment C moves. Before reading any
future C > A from this driver as transfer, give each arm its own freshly constructed brain.
"""
import json, sys, time
from nyxara.eval.general_novel import build_general_novel_benchmark
from nyxara.eval.teacher_removal import TeacherRemoval, score_benchmark_task
from nyxara.njp.brain import NJPBrain

MAX_TOKENS = 420          # enough to clear the reasoning block at ~1.5 tok/s without stalling

tasks = build_general_novel_benchmark().tasks()
brain = NJPBrain()
cortex = brain.cortex

def njp_alone(task):
    try:
        return str(brain.think(task.prompt).answer or "")
    except Exception:
        return ""

def njp_with_teacher(task):
    """NJP with the teacher available as a proposer — what 'NJP + Qwythos' means operationally."""
    own = njp_alone(task)
    if own.strip():
        return own
    t0 = time.time()
    reply = cortex.ask(task.prompt, max_tokens=MAX_TOKENS)
    print(f"    teacher {time.time()-t0:5.0f}s tok={reply.tokens:4d} "
          f"trunc={reply.truncated} -> {reply.text[:60]!r}", flush=True)
    return reply.text

class _TeacherRemoved:
    """Arm C's core: the same brain with the teacher detached, structurally."""
    class njp:
        cortex = None
        shadow_cognition = None
        fabric = None

print(f"battery: general_novel, {len(tasks)} held-out tasks", flush=True)
print(f"teacher: {cortex.stats()['model']} available={cortex.available()}", flush=True)
t0 = time.time()
result = TeacherRemoval().run(
    tasks, battery="general_novel",
    solve_a=njp_alone, solve_b=njp_with_teacher, solve_c=njp_alone,
    score=score_benchmark_task, core_c=_TeacherRemoved())
print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
print(result.explain())
json.dump(result.to_dict(), open(sys.argv[1], "w"), indent=2, default=str)
