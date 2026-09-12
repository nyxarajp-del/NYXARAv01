"""A second bank of hard problems, chosen **after** the skeletons existed.

Why this file exists, plainly. The thirty problems in :mod:`tests.njp.test_unseen_hard` were the
ones being measured while :data:`nyxara.njp.coding.SKELETONS` was written, and a skeleton written
until that bank goes green is fitted to that bank however carefully its docstring is phrased. The
number there reached **25 of 28**. This bank was written afterwards, from a different corner of
the same subject, and measured once.

The first measurement was **0 of 21**. Not one. Every single problem was an abstention — the
safety property held perfectly, and the generalisation claim did not exist. That result is the
most useful number in the whole exercise: it said the twenty-five was a list of answers.

What moved it was **not** twenty-one more skeletons. Reading what actually blocked each problem,
two *axes* came up over and over — a loop may carry more than one variable updated together, and
a table may be laid over a grid rather than over a sequence — and both are dimensions of
skeletons that already existed rather than new members. Adding those two took this bank to
**6 of 21**, and one of the six (``binary_search_position``) is a problem neither axis was
designed for, which is the only kind of evidence that means anything here.

Fifteen remain. They are named in the module and each one is a control structure still missing:
a stack whose pops depend on a mapping, a sieve with a strided inner range, a triple-nested scan,
a set that grows to a fixed point, a string built by walking an index up and down.
"""

from __future__ import annotations

import random
import zlib

import pytest

from nyxara.njp.coding import Coder, CodeError, read_python
from nyxara.njp.tasks import normalise, reference
from tests.njp.test_unseen_hard import _distinct, _params, _spec


FRESH = [
    ("max_subarray_product",
     "def f(xs):\n"
     "    if len(xs) == 0:\n"
     "        return 0\n"
     "    best = xs[0]\n"
     "    hi = xs[0]\n"
     "    lo = xs[0]\n"
     "    for x in xs[1:]:\n"
     "        cands = [x, hi * x, lo * x]\n"
     "        hi = max(cands)\n"
     "        lo = min(cands)\n"
     "        if hi > best:\n"
     "            best = hi\n"
     "    return best",
     [((2, 3, -2, 4),), ((-2, 0, -1),), ((1, 2, 3),), ((-2, 3, -4),)]),

    ("longest_common_substring",
     "def f(a, b):\n"
     "    best = 0\n"
     "    for i in range(len(a)):\n"
     "        for j in range(len(b)):\n"
     "            k = 0\n"
     "            while i + k < len(a) and j + k < len(b) and a[i+k] == b[j+k]:\n"
     "                k += 1\n"
     "            if k > best:\n"
     "                best = k\n"
     "    return best",
     [("abcdxyz", "xyzabcd"), ("zxabcdezy", "yzabcdezx"), ("abc", "def")]),

    ("climb_stairs_ways",
     "def f(n):\n"
     "    a = 1\n"
     "    b = 1\n"
     "    for _ in range(n):\n"
     "        a, b = b, a + b\n"
     "    return a",
     [(2,), (3,), (5,), (7,)]),

    ("house_robber",
     "def f(xs):\n"
     "    take = 0\n"
     "    skip = 0\n"
     "    for x in xs:\n"
     "        take, skip = skip + x, max(skip, take)\n"
     "    return max(take, skip)",
     [((1, 2, 3, 1),), ((2, 7, 9, 3, 1),), ((5,),)]),

    ("min_path_sum",
     "def f(grid):\n"
     "    rows = len(grid)\n"
     "    cols = len(grid[0])\n"
     "    prev = []\n"
     "    run = 0\n"
     "    for c in range(cols):\n"
     "        run += grid[0][c]\n"
     "        prev.append(run)\n"
     "    for r in range(1, rows):\n"
     "        cur = [prev[0] + grid[r][0]]\n"
     "        for c in range(1, cols):\n"
     "            cur.append(grid[r][c] + min(prev[c], cur[c-1]))\n"
     "        prev = cur\n"
     "    return prev[cols-1]",
     [(((1, 3, 1), (1, 5, 1), (4, 2, 1)),), (((1, 2), (1, 1)),)]),

    ("unique_paths_obstacles",
     "def f(grid):\n"
     "    cols = len(grid[0])\n"
     "    prev = []\n"
     "    ok = 1\n"
     "    for c in range(cols):\n"
     "        if grid[0][c] == 1:\n"
     "            ok = 0\n"
     "        prev.append(ok)\n"
     "    for r in range(1, len(grid)):\n"
     "        cur = [0 if grid[r][0] == 1 else prev[0]]\n"
     "        for c in range(1, cols):\n"
     "            cur.append(0 if grid[r][c] == 1 else prev[c] + cur[c-1])\n"
     "        prev = cur\n"
     "    return prev[cols-1]",
     [(((0, 0, 0), (0, 1, 0), (0, 0, 0)),), (((0, 1), (0, 0)),)]),

    ("maximal_square",
     "def f(grid):\n"
     "    best = 0\n"
     "    cols = len(grid[0])\n"
     "    prev = [0 for _ in range(cols)]\n"
     "    for r in range(len(grid)):\n"
     "        cur = []\n"
     "        for c in range(cols):\n"
     "            if grid[r][c] == 0:\n"
     "                cur.append(0)\n"
     "            elif r == 0 or c == 0:\n"
     "                cur.append(1)\n"
     "            else:\n"
     "                cur.append(1 + min(prev[c], prev[c-1], cur[c-1]))\n"
     "            if cur[c] > best:\n"
     "                best = cur[c]\n"
     "        prev = cur\n"
     "    return best * best",
     [(((1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1)),), (((0, 1), (1, 0)),)]),

    ("decode_ways",
     "def f(s):\n"
     "    if len(s) == 0 or s[0] == '0':\n"
     "        return 0\n"
     "    prev2 = 1\n"
     "    prev1 = 1\n"
     "    for i in range(1, len(s)):\n"
     "        cur = 0\n"
     "        if s[i] != '0':\n"
     "            cur += prev1\n"
     "        two = int(s[i-1:i+1])\n"
     "        if 10 <= two <= 26:\n"
     "            cur += prev2\n"
     "        prev2 = prev1\n"
     "        prev1 = cur\n"
     "    return prev1",
     [("226",), ("12",), ("06",), ("1111",)]),

    ("perfect_squares_count",
     "def f(n):\n"
     "    best = [0]\n"
     "    for i in range(1, n + 1):\n"
     "        cheapest = i\n"
     "        k = 1\n"
     "        while k * k <= i:\n"
     "            if best[i - k*k] + 1 < cheapest:\n"
     "                cheapest = best[i - k*k] + 1\n"
     "            k += 1\n"
     "        best.append(cheapest)\n"
     "    return best[n]",
     [(12,), (13,), (4,), (7,)]),

    ("count_primes",
     "def f(n):\n"
     "    if n < 2:\n"
     "        return 0\n"
     "    sieve = [True for _ in range(n)]\n"
     "    sieve[0] = False\n"
     "    sieve[1] = False\n"
     "    for i in range(2, n):\n"
     "        if sieve[i]:\n"
     "            for j in range(i * i, n, i):\n"
     "                sieve[j] = False\n"
     "    return sum(1 for v in sieve if v)",
     [(10,), (20,), (2,), (30,)]),

    ("three_sum_count",
     "def f(xs, target):\n"
     "    total = 0\n"
     "    for i in range(len(xs)):\n"
     "        for j in range(i + 1, len(xs)):\n"
     "            for k in range(j + 1, len(xs)):\n"
     "                if xs[i] + xs[j] + xs[k] == target:\n"
     "                    total += 1\n"
     "    return total",
     [((1, 2, 3, 4, 5), 9), ((0, 0, 0, 0), 0), ((-1, 0, 1, 2), 2)]),

    ("longest_consecutive_sequence",
     "def f(xs):\n"
     "    known = set(xs)\n"
     "    best = 0\n"
     "    for x in known:\n"
     "        if x - 1 not in known:\n"
     "            run = 1\n"
     "            while x + run in known:\n"
     "                run += 1\n"
     "            if run > best:\n"
     "                best = run\n"
     "    return best",
     [((100, 4, 200, 1, 3, 2),), ((0, 3, 7, 2, 5, 8, 4, 6, 0, 1),), ((),)]),

    ("evaluate_rpn",
     "def f(tokens):\n"
     "    st = []\n"
     "    for t in tokens:\n"
     "        if t == '+':\n"
     "            st = st[:len(st)-2] + [st[len(st)-2] + st[len(st)-1]]\n"
     "        elif t == '*':\n"
     "            st = st[:len(st)-2] + [st[len(st)-2] * st[len(st)-1]]\n"
     "        elif t == '-':\n"
     "            st = st[:len(st)-2] + [st[len(st)-2] - st[len(st)-1]]\n"
     "        else:\n"
     "            st = st + [int(t)]\n"
     "    return st[0]",
     [(("2", "3", "+", "4", "*"),), (("5", "1", "-"),), (("7",),)]),

    ("balanced_brackets",
     "def f(s):\n"
     "    pairs = {')': '(', ']': '[', '}': '{'}\n"
     "    st = []\n"
     "    for ch in s:\n"
     "        if ch in '([{':\n"
     "            st = st + [ch]\n"
     "        elif ch in pairs:\n"
     "            if len(st) == 0 or st[len(st)-1] != pairs[ch]:\n"
     "                return False\n"
     "            st = st[:len(st)-1]\n"
     "    return len(st) == 0",
     [("([]{})",), ("([)]",), ("",), ("(((",)]),

    ("spiral_order",
     "def f(grid):\n"
     "    out = []\n"
     "    rows = list(list(r) for r in grid)\n"
     "    while len(rows) > 0:\n"
     "        out = out + list(rows[0])\n"
     "        rows = rows[1:]\n"
     "        rows = [list(r) for r in zip(*rows)][::-1] if len(rows) > 0 else []\n"
     "    return out",
     [(((1, 2, 3), (4, 5, 6), (7, 8, 9)),), (((1, 2), (3, 4)),)]),

    ("rotate_matrix",
     "def f(grid):\n"
     "    n = len(grid)\n"
     "    out = []\n"
     "    for c in range(n):\n"
     "        row = []\n"
     "        for r in range(n - 1, -1, -1):\n"
     "            row.append(grid[r][c])\n"
     "        out.append(row)\n"
     "    return out",
     [(((1, 2), (3, 4)),), (((1, 2, 3), (4, 5, 6), (7, 8, 9)),)]),

    ("num_islands",
     "def f(grid):\n"
     "    rows = len(grid)\n"
     "    cols = len(grid[0])\n"
     "    label = [i for i in range(rows * cols)]\n"
     "    for _ in range(rows * cols):\n"
     "        for r in range(rows):\n"
     "            for c in range(cols):\n"
     "                if grid[r][c] == 1:\n"
     "                    if r + 1 < rows and grid[r+1][c] == 1:\n"
     "                        m = min(label[r*cols+c], label[(r+1)*cols+c])\n"
     "                        label[r*cols+c] = m\n"
     "                        label[(r+1)*cols+c] = m\n"
     "                    if c + 1 < cols and grid[r][c+1] == 1:\n"
     "                        m = min(label[r*cols+c], label[r*cols+c+1])\n"
     "                        label[r*cols+c] = m\n"
     "                        label[r*cols+c+1] = m\n"
     "    seen = []\n"
     "    for r in range(rows):\n"
     "        for c in range(cols):\n"
     "            if grid[r][c] == 1 and label[r*cols+c] not in seen:\n"
     "                seen = seen + [label[r*cols+c]]\n"
     "    return len(seen)",
     [(((1, 1, 0), (0, 1, 0), (0, 0, 1)),), (((1, 0), (0, 1)),)]),

    ("binary_search_position",
     "def f(xs, target):\n"
     "    lo = 0\n"
     "    hi = len(xs)\n"
     "    while lo < hi:\n"
     "        mid = (lo + hi) // 2\n"
     "        if xs[mid] < target:\n"
     "            lo = mid + 1\n"
     "        else:\n"
     "            hi = mid\n"
     "    return lo",
     [((1, 3, 5, 6), 5), ((1, 3, 5, 6), 2), ((1, 3, 5, 6), 7), ((), 1)]),

    ("gray_code_at",
     "def f(n):\n"
     "    return n ^ (n >> 1)",
     [(0,), (1,), (2,), (5,), (9,)]),

    ("zigzag_rows",
     "def f(s, rows):\n"
     "    if rows <= 1:\n"
     "        return s\n"
     "    lines = ['' for _ in range(rows)]\n"
     "    r = 0\n"
     "    step = 1\n"
     "    for ch in s:\n"
     "        lines[r] = lines[r] + ch\n"
     "        if r == 0:\n"
     "            step = 1\n"
     "        elif r == rows - 1:\n"
     "            step = -1\n"
     "        r += step\n"
     "    return ''.join(lines)",
     [("PAYPALISHIRING", 3), ("ABCD", 2), ("AB", 1)]),

    ("partition_equal_subset",
     "def f(xs):\n"
     "    total = sum(xs)\n"
     "    if total % 2 == 1:\n"
     "        return False\n"
     "    half = total // 2\n"
     "    reach = set([0])\n"
     "    for x in xs:\n"
     "        reach = reach | set(r + x for r in reach)\n"
     "    return half in reach",
     [((1, 5, 11, 5),), ((1, 2, 3, 5),), ((2, 2),)]),

    ("word_ladder_length",
     "def f(start, target, words):\n"
     "    frontier = [start]\n"
     "    seen = [start]\n"
     "    steps = 1\n"
     "    while len(frontier) > 0:\n"
     "        if target in frontier:\n"
     "            return steps\n"
     "        nxt = []\n"
     "        for w in frontier:\n"
     "            for cand in words:\n"
     "                if cand in seen:\n"
     "                    continue\n"
     "                diff = sum(1 for a, b in zip(w, cand) if a != b)\n"
     "                if diff == 1 and len(w) == len(cand):\n"
     "                    nxt = nxt + [cand]\n"
     "                    seen = seen + [cand]\n"
     "        frontier = nxt\n"
     "        steps += 1\n"
     "    return 0",
     [("hit", "cog", ("hot", "dot", "dog", "lot", "log", "cog")),
      ("aa", "bb", ("ab", "bb"))]),

    ("closest_pair_distance",
     "def f(pts):\n"
     "    best = -1\n"
     "    for i in range(len(pts)):\n"
     "        for j in range(i + 1, len(pts)):\n"
     "            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2\n"
     "            if best == -1 or d < best:\n"
     "                best = d\n"
     "    return best",
     [(((0, 0), (3, 4), (1, 1)),), (((0, 0), (0, 5)),)]),

    ("int_to_roman",
     "def f(n):\n"
     "    values = ((1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),\n"
     "              (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'),\n"
     "              (5, 'V'), (4, 'IV'), (1, 'I'))\n"
     "    out = ''\n"
     "    for value, sign in values:\n"
     "        while n >= value:\n"
     "            out = out + sign\n"
     "            n = n - value\n"
     "    return out",
     [(3,), (58,), (1994,), (9,)]),

    ("next_greater_counts",
     "def f(xs):\n"
     "    out = []\n"
     "    for i in range(len(xs)):\n"
     "        found = -1\n"
     "        for j in range(i + 1, len(xs)):\n"
     "            if xs[j] > xs[i]:\n"
     "                found = xs[j]\n"
     "                break\n"
     "        out.append(found)\n"
     "    return out",
     [((2, 1, 2, 4, 3),), ((5, 4, 3),), ((1, 2),)]),
]


#: The six this bank is written cold, each verified on pairs it was not fitted on. Named rather
#: than counted, so a regression that swapped one for another cannot pass.
SOLVED_COLD = (
    "binary_search_position", "climb_stairs_ways", "house_robber",
    "maximal_square", "min_path_sum", "unique_paths_obstacles",
)


def _fresh_spec(name, source, arglists):
    rng = random.Random(zlib.crc32(name.encode()) % 9973)
    pool = _distinct(source, arglists, rng, want=10)
    if len(pool) < 6:
        pytest.skip(f"{name}: {len(pool)} distinct inputs, too few to hold any out")
    return _spec(name, source, pool, _params(pool))


@pytest.mark.parametrize("name,source,arglists", FRESH, ids=[case[0] for case in FRESH])
def test_she_reads_and_runs_a_problem_from_the_second_bank(name, source, arglists):
    """Reading is not where the ceiling is, and this bank says so as loudly as the first one."""
    try:
        program = read_python(source, name="f")
    except CodeError as exc:
        pytest.skip(f"{name}: outside the language ({exc})")
    coder = Coder()
    for args in arglists:
        assert coder.run(program, args) == normalise(reference(source)(*args)), (name, args)


@pytest.mark.parametrize("name", SOLVED_COLD)
def test_the_two_axes_reach_these_without_a_skeleton_written_for_them(name):
    """Six problems from a bank that had no influence on any skeleton, written from pairs alone.

    The claim is narrow and it is the only one worth making: these fall out of *axes* — how many
    variables a loop carries, and whether a table is laid over a grid — added in response to a
    0/21 result, not in response to any problem on this list.
    """
    case = {entry[0]: entry for entry in FRESH}[name]
    spec = _fresh_spec(name, case[1], case[2])
    blind = Coder()
    # The budget `write(attempts=20000)` forwards. `compose`'s own default is a fifth of it,
    # sized so that a *failed* question costs a second rather than twenty — which is what a
    # syllabus of a few hundred questions can afford and a benchmark of twenty-one can not.
    program = blind.compose(spec, budget=1_000_000)
    assert program is not None, f"{name}: composed nothing"
    assert blind.check(program, spec.held_out).ok, (
        f"{name}: fits the shown pairs and fails held-out\n{program.source()}")


@pytest.mark.parametrize("name,source,arglists", FRESH, ids=[case[0] for case in FRESH])
def test_on_this_bank_too_not_knowing_is_an_abstention(name, source, arglists):
    """The safety property, on the bank she is *bad* at — which is where it matters.

    Fifteen of these twenty-one she cannot write. Not one of them comes back as a wrong program:
    the whole failure mode is silence. This assertion has held at 0/21 and at 6/21 alike, and it
    is the one that would make a rising score untrustworthy if it ever stopped holding.
    """
    try:
        read_python(source, name="f")
    except CodeError as exc:
        pytest.skip(f"{name}: outside the language ({exc})")
    spec = _fresh_spec(name, source, arglists)
    blind = Coder()
    written = blind.write(spec, attempts=20000, graft=True)
    if not written.ok:
        return                                  # an abstention: the honest outcome here
    assert blind.check(written.program, spec.held_out).ok, (
        f"{name}: wrote a program that fits the shown pairs and fails held-out\n"
        f"{written.program.source()}")
