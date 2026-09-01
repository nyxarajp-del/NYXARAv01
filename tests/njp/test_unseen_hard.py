"""Problems she was never taught: what she can do with them, and what she cannot.

Nothing here resembles any of the sixty-five families in :mod:`nyxara.njp.tasks`. These are the
classic hard ones — dynamic programming over two dimensions, backtracking, graph search, string
matching with a failure function, geometry — and she is given the **whole curriculum** first, so
a failure below is a failure with all fifty-four taught shapes in hand rather than the failure of
an untaught brain.

The file exists because asking these questions found two real gaps in the reader, each of which
ruled out a whole class of program rather than one line: ``min(a, b, c)`` was refused past two
arguments, which is the expression at the centre of edit distance, and ``range(n, -1, -1)`` was
refused, which is how the backwards pass of a dynamic program is written. Both are now covered
here so they cannot come back.

**What is asserted, and what is deliberately not.** Reading, running, tracing and repairing are
asserted as capabilities. *Writing* one of these from examples alone is not: she solves almost
none of them cold, because her synthesis is enumeration over shapes she has been shown and none
of these shapes has been. Asserting an upper bound would be a test that fails when she improves,
so what is asserted instead is the property that makes the limit safe — **whatever she does write
cold has to be right**. Not knowing shows up as an abstention, never as a wrong program.

Measured when this was written: 76/76 read-and-run, 30/30 trace, 21/24 repaired, 1/28 written
cold; and shown one worked solution, 22/24 written again for data she had not seen, against 5/24
for the same problems with no demonstration.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.coding import (
    Call, Coder, Lit, OPS, Program, Spec, arity_of, read_python,
    _positions, _replace_at,
)
from nyxara.njp.tasks import EVERY_TASK, instance, normalise, reference


HARD = [
    # ---- two-dimensional dynamic programming ----
    ("edit_distance",
     "def f(a, b):\n"
     "    prev = [j for j in range(len(b) + 1)]\n"
     "    for i in range(1, len(a) + 1):\n"
     "        cur = [i]\n"
     "        for j in range(1, len(b) + 1):\n"
     "            c = 0 if a[i-1] == b[j-1] else 1\n"
     "            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + c))\n"
     "        prev = cur\n"
     "    return prev[len(b)]",
     [("kitten", "sitting"), ("flaw", "lawn"), ("", "abc"), ("same", "same")]),

    ("longest_common_subsequence",
     "def f(a, b):\n"
     "    prev = [0 for _ in range(len(b) + 1)]\n"
     "    for i in range(1, len(a) + 1):\n"
     "        cur = [0]\n"
     "        for j in range(1, len(b) + 1):\n"
     "            if a[i-1] == b[j-1]:\n"
     "                cur.append(prev[j-1] + 1)\n"
     "            else:\n"
     "                cur.append(max(prev[j], cur[j-1]))\n"
     "        prev = cur\n"
     "    return prev[len(b)]",
     [("abcde", "ace"), ("abc", "def"), ("aggtab", "gxtxayb")]),

    ("longest_increasing_subsequence",
     "def f(xs):\n"
     "    best = []\n"
     "    for i in range(len(xs)):\n"
     "        run = 1\n"
     "        for j in range(i):\n"
     "            if xs[j] < xs[i] and best[j] + 1 > run:\n"
     "                run = best[j] + 1\n"
     "        best.append(run)\n"
     "    if len(best) == 0:\n"
     "        return 0\n"
     "    return max(best)",
     [((10, 9, 2, 5, 3, 7, 101, 18),), ((0, 1, 0, 3, 2, 3),), ((7, 7, 7),)]),

    ("knapsack",
     "def f(weights, values, cap):\n"
     "    best = [0 for _ in range(cap + 1)]\n"
     "    for i in range(len(weights)):\n"
     "        nxt = []\n"
     "        for c in range(cap + 1):\n"
     "            take = 0\n"
     "            if weights[i] <= c:\n"
     "                take = values[i] + best[c - weights[i]]\n"
     "            nxt.append(max(best[c], take))\n"
     "        best = nxt\n"
     "    return best[cap]",
     [((1, 3, 4, 5), (1, 4, 5, 7), 7), ((2, 2, 3), (3, 4, 5), 5)]),

    ("coin_change",
     "def f(coins, target):\n"
     "    big = target + 1\n"
     "    best = [0]\n"
     "    for amount in range(1, target + 1):\n"
     "        cheapest = big\n"
     "        for c in coins:\n"
     "            if c <= amount and best[amount - c] + 1 < cheapest:\n"
     "                cheapest = best[amount - c] + 1\n"
     "        best.append(cheapest)\n"
     "    if best[target] == big:\n"
     "        return -1\n"
     "    return best[target]",
     [((1, 2, 5), 11), ((2,), 3), ((1,), 0)]),

    ("subset_sum",
     "def f(xs, target):\n"
     "    reach = [True]\n"
     "    for _ in range(target):\n"
     "        reach.append(False)\n"
     "    for x in xs:\n"
     "        for i in range(target, x - 1, -1):\n"
     "            if reach[i - x]:\n"
     "                reach[i] = True\n"
     "    return reach[target]",
     [((3, 34, 4, 12, 5, 2), 9), ((1, 2, 5), 4), ((2, 4), 3)]),

    ("word_break",
     "def f(s, words):\n"
     "    ok = [True]\n"
     "    for i in range(1, len(s) + 1):\n"
     "        good = False\n"
     "        for w in words:\n"
     "            if len(w) <= i and ok[i - len(w)] and s[i - len(w):i] == w:\n"
     "                good = True\n"
     "        ok.append(good)\n"
     "    return ok[len(s)]",
     [("leetcode", ("leet", "code")), ("catsandog", ("cats", "dog", "sand", "and", "cat"))]),

    # ---- backtracking ----
    ("n_queens",
     "def place(cols, n):\n"
     "    row = len(cols)\n"
     "    if row == n:\n"
     "        return 1\n"
     "    total = 0\n"
     "    for c in range(n):\n"
     "        ok = True\n"
     "        for r in range(row):\n"
     "            if cols[r] == c:\n"
     "                ok = False\n"
     "            if row - r == c - cols[r]:\n"
     "                ok = False\n"
     "            if row - r == cols[r] - c:\n"
     "                ok = False\n"
     "        if ok:\n"
     "            total += place(cols + [c], n)\n"
     "    return total\n\n"
     "def f(n):\n"
     "    return place([], n)",
     [(4,), (5,), (6,), (1,)]),

    ("permutation_count",
     "def go(used, n):\n"
     "    if len(used) == n:\n"
     "        return 1\n"
     "    total = 0\n"
     "    for k in range(n):\n"
     "        if k not in used:\n"
     "            total += go(used + [k], n)\n"
     "    return total\n\n"
     "def f(n):\n"
     "    return go([], n)",
     [(3,), (4,), (5,)]),

    # ---- graphs ----
    ("dijkstra",
     "def f(edges, n, start):\n"
     "    big = 10 ** 6\n"
     "    dist = []\n"
     "    for _ in range(n):\n"
     "        dist.append(big)\n"
     "    dist[start] = 0\n"
     "    seen = []\n"
     "    while len(seen) < n:\n"
     "        best = -1\n"
     "        for v in range(n):\n"
     "            if v not in seen:\n"
     "                if best == -1 or dist[v] < dist[best]:\n"
     "                    best = v\n"
     "        if best == -1 or dist[best] == big:\n"
     "            return dist\n"
     "        seen = seen + [best]\n"
     "        for e in edges:\n"
     "            if e[0] == best and dist[best] + e[2] < dist[e[1]]:\n"
     "                dist[e[1]] = dist[best] + e[2]\n"
     "    return dist",
     [(((0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)), 4, 0)]),

    ("topological_order",
     "def f(n, edges):\n"
     "    indeg = []\n"
     "    for _ in range(n):\n"
     "        indeg.append(0)\n"
     "    for e in edges:\n"
     "        indeg[e[1]] = indeg[e[1]] + 1\n"
     "    out = []\n"
     "    while len(out) < n:\n"
     "        pick = -1\n"
     "        for v in range(n):\n"
     "            if indeg[v] == 0 and v not in out:\n"
     "                pick = v\n"
     "        if pick == -1:\n"
     "            return []\n"
     "        out = out + [pick]\n"
     "        indeg[pick] = -1\n"
     "        for e in edges:\n"
     "            if e[0] == pick:\n"
     "                indeg[e[1]] = indeg[e[1]] - 1\n"
     "    return out",
     [(4, ((0, 1), (0, 2), (1, 3), (2, 3))), (3, ((0, 1), (1, 2)))]),

    ("connected_components",
     "def f(n, edges):\n"
     "    parent = [i for i in range(n)]\n"
     "    for e in edges:\n"
     "        a = e[0]\n"
     "        while parent[a] != a:\n"
     "            a = parent[a]\n"
     "        b = e[1]\n"
     "        while parent[b] != b:\n"
     "            b = parent[b]\n"
     "        if a != b:\n"
     "            parent[a] = b\n"
     "    roots = []\n"
     "    for v in range(n):\n"
     "        r = v\n"
     "        while parent[r] != r:\n"
     "            r = parent[r]\n"
     "        if r not in roots:\n"
     "            roots = roots + [r]\n"
     "    return len(roots)",
     [(5, ((0, 1), (1, 2), (3, 4))), (4, ())]),

    # ---- strings, hard ----
    ("kmp_search",
     "def f(text, pat):\n"
     "    fail = [0]\n"
     "    k = 0\n"
     "    for i in range(1, len(pat)):\n"
     "        while k > 0 and pat[i] != pat[k]:\n"
     "            k = fail[k - 1]\n"
     "        if pat[i] == pat[k]:\n"
     "            k += 1\n"
     "        fail.append(k)\n"
     "    k = 0\n"
     "    for i in range(len(text)):\n"
     "        while k > 0 and text[i] != pat[k]:\n"
     "            k = fail[k - 1]\n"
     "        if text[i] == pat[k]:\n"
     "            k += 1\n"
     "        if k == len(pat):\n"
     "            return i - len(pat) + 1\n"
     "    return -1",
     [("ababcabcabababd", "ababd"), ("aaaaa", "bba"), ("abc", "abc")]),

    ("longest_palindromic_substring",
     "def f(s):\n"
     "    best = ''\n"
     "    for i in range(len(s)):\n"
     "        for j in range(i, len(s)):\n"
     "            part = s[i:j+1]\n"
     "            if part == part[::-1] and len(part) > len(best):\n"
     "                best = part\n"
     "    return best",
     [("babad",), ("cbbd",), ("a",)]),

    ("longest_valid_parentheses",
     "def f(s):\n"
     "    best = 0\n"
     "    for i in range(len(s)):\n"
     "        depth = 0\n"
     "        for j in range(i, len(s)):\n"
     "            if s[j] == '(':\n"
     "                depth += 1\n"
     "            else:\n"
     "                depth -= 1\n"
     "            if depth < 0:\n"
     "                break\n"
     "            if depth == 0 and j - i + 1 > best:\n"
     "                best = j - i + 1\n"
     "    return best",
     [("(()",), (")()())",), ("",)]),

    ("regex_match",
     "def go(s, p):\n"
     "    if len(p) == 0:\n"
     "        return len(s) == 0\n"
     "    first = len(s) > 0 and (p[0] == s[0] or p[0] == '.')\n"
     "    if len(p) > 1 and p[1] == '*':\n"
     "        if go(s, p[2:]):\n"
     "            return True\n"
     "        if first:\n"
     "            return go(s[1:], p)\n"
     "        return False\n"
     "    if first:\n"
     "        return go(s[1:], p[1:])\n"
     "    return False\n\n"
     "def f(s, p):\n"
     "    return go(s, p)",
     [("aa", "a"), ("aa", "a*"), ("ab", ".*"), ("mississippi", "mis*is*p*.")]),

    ("min_window_length",
     "def f(s, t):\n"
     "    best = len(s) + 1\n"
     "    for i in range(len(s)):\n"
     "        for j in range(i, len(s)):\n"
     "            part = s[i:j+1]\n"
     "            ok = True\n"
     "            for ch in t:\n"
     "                if part.count(ch) < t.count(ch):\n"
     "                    ok = False\n"
     "            if ok and len(part) < best:\n"
     "                best = len(part)\n"
     "    if best == len(s) + 1:\n"
     "        return 0\n"
     "    return best",
     [("ADOBECODEBANC", "ABC"), ("a", "aa")]),

    # ---- arrays, hard ----
    ("trapping_rain_water",
     "def f(h):\n"
     "    total = 0\n"
     "    for i in range(len(h)):\n"
     "        left = 0\n"
     "        for j in range(i + 1):\n"
     "            if h[j] > left:\n"
     "                left = h[j]\n"
     "        right = 0\n"
     "        for j in range(i, len(h)):\n"
     "            if h[j] > right:\n"
     "                right = h[j]\n"
     "        total += min(left, right) - h[i]\n"
     "    return total",
     [((0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1),), ((4, 2, 0, 3, 2, 5),)]),

    ("median_two_sorted",
     "def f(a, b):\n"
     "    merged = sorted(list(a) + list(b))\n"
     "    n = len(merged)\n"
     "    if n % 2 == 1:\n"
     "        return merged[n // 2]\n"
     "    return (merged[n // 2 - 1] + merged[n // 2]) // 2",
     [((1, 3), (2,)), ((1, 2), (3, 4))]),

    ("largest_rectangle_histogram",
     "def f(h):\n"
     "    best = 0\n"
     "    for i in range(len(h)):\n"
     "        low = h[i]\n"
     "        for j in range(i, len(h)):\n"
     "            if h[j] < low:\n"
     "                low = h[j]\n"
     "            area = low * (j - i + 1)\n"
     "            if area > best:\n"
     "                best = area\n"
     "    return best",
     [((2, 1, 5, 6, 2, 3),), ((2, 4),)]),

    ("merge_intervals",
     "def f(rows):\n"
     "    items = sorted(rows)\n"
     "    out = []\n"
     "    for it in items:\n"
     "        if len(out) > 0 and it[0] <= out[len(out)-1][1]:\n"
     "            last = out[len(out)-1]\n"
     "            out[len(out)-1] = [last[0], max(last[1], it[1])]\n"
     "        else:\n"
     "            out.append([it[0], it[1]])\n"
     "    return out",
     [(((1, 3), (2, 6), (8, 10), (15, 18)),), (((1, 4), (4, 5)),)]),

    # ---- numbers and geometry ----
    ("matrix_determinant",
     "def det(m):\n"
     "    n = len(m)\n"
     "    if n == 1:\n"
     "        return m[0][0]\n"
     "    total = 0\n"
     "    sign = 1\n"
     "    for c in range(n):\n"
     "        sub = []\n"
     "        for r in range(1, n):\n"
     "            row = []\n"
     "            for k in range(n):\n"
     "                if k != c:\n"
     "                    row.append(m[r][k])\n"
     "            sub.append(row)\n"
     "        total += sign * m[0][c] * det(sub)\n"
     "        sign = -sign\n"
     "    return total\n\n"
     "def f(m):\n"
     "    return det(m)",
     [(((1, 2), (3, 4)),), (((6, 1, 1), (4, -2, 5), (2, 8, 7)),)]),

    ("josephus",
     "def f(n, k):\n"
     "    alive = [i for i in range(n)]\n"
     "    at = 0\n"
     "    while len(alive) > 1:\n"
     "        at = (at + k - 1) % len(alive)\n"
     "        alive = alive[:at] + alive[at+1:]\n"
     "    return alive[0]",
     [(7, 3), (5, 2), (1, 1)]),

    ("catalan",
     "def f(n):\n"
     "    c = [1]\n"
     "    for i in range(1, n + 1):\n"
     "        total = 0\n"
     "        for j in range(i):\n"
     "            total += c[j] * c[i - 1 - j]\n"
     "        c.append(total)\n"
     "    return c[n]",
     [(0,), (3,), (6,)]),

    ("pascal_row",
     "def f(n):\n"
     "    row = [1]\n"
     "    for i in range(n):\n"
     "        nxt = [1]\n"
     "        for j in range(len(row) - 1):\n"
     "            nxt.append(row[j] + row[j+1])\n"
     "        nxt.append(1)\n"
     "        row = nxt\n"
     "    return row",
     [(0,), (4,), (6,)]),

    ("convex_hull_size",
     "def f(pts):\n"
     "    ps = sorted(pts)\n"
     "    if len(ps) < 3:\n"
     "        return len(ps)\n"
     "    lower = []\n"
     "    for p in ps:\n"
     "        while len(lower) >= 2 and (lower[len(lower)-1][0] - lower[len(lower)-2][0]) * "
     "(p[1] - lower[len(lower)-2][1]) - (lower[len(lower)-1][1] - lower[len(lower)-2][1]) * "
     "(p[0] - lower[len(lower)-2][0]) <= 0:\n"
     "            lower = lower[:len(lower)-1]\n"
     "        lower = lower + [p]\n"
     "    return len(lower)",
     [(((0, 0), (1, 1), (2, 2), (3, 0)),), (((0, 0), (2, 0), (1, 1)),)]),

    ("sudoku_row_valid",
     "def f(grid):\n"
     "    for row in grid:\n"
     "        seen = []\n"
     "        for v in row:\n"
     "            if v != 0 and v in seen:\n"
     "                return False\n"
     "            if v != 0:\n"
     "                seen = seen + [v]\n"
     "    for c in range(len(grid[0])):\n"
     "        seen = []\n"
     "        for r in range(len(grid)):\n"
     "            v = grid[r][c]\n"
     "            if v != 0 and v in seen:\n"
     "                return False\n"
     "            if v != 0:\n"
     "                seen = seen + [v]\n"
     "    return True",
     [(((1, 2, 0), (0, 3, 1), (2, 0, 3)),), (((1, 1, 0), (0, 0, 0), (0, 0, 0)),)]),

    ("count_inversions",
     "def f(xs):\n"
     "    total = 0\n"
     "    for i in range(len(xs)):\n"
     "        for j in range(i + 1, len(xs)):\n"
     "            if xs[i] > xs[j]:\n"
     "                total += 1\n"
     "    return total",
     [((2, 4, 1, 3, 5),), ((5, 4, 3, 2, 1),)]),

    ("roman_to_int",
     "def f(s):\n"
     "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
     "    total = 0\n"
     "    for i in range(len(s)):\n"
     "        v = vals[s[i]]\n"
     "        if i + 1 < len(s) and v < vals[s[i+1]]:\n"
     "            total -= v\n"
     "        else:\n"
     "            total += v\n"
     "    return total",
     [("MCMXCIV",), ("IX",), ("LVIII",)]),

    ("max_path_triangle",
     "def f(rows):\n"
     "    best = list(rows[len(rows)-1])\n"
     "    for i in range(len(rows) - 2, -1, -1):\n"
     "        nxt = []\n"
     "        for j in range(len(rows[i])):\n"
     "            nxt.append(rows[i][j] + max(best[j], best[j+1]))\n"
     "        best = nxt\n"
     "    return best[0]",
     [(((2,), (3, 4), (6, 5, 7), (4, 1, 8, 3)),)]),
]


LETTERS = "abcde"


@pytest.fixture(scope="module")
def schooled():
    """A coder holding the whole curriculum — every lesson she has ever had."""
    coder, rng = Coder(), random.Random(17)
    for task in EVERY_TASK:
        spec, source = instance(task, rng, task.id + "_lesson")
        coder.learn_python(spec, source)
    return coder


def _like(value, rng):
    """A fresh value of the same shape as ``value``, for generating unseen inputs."""
    if isinstance(value, bool):
        return rng.choice([True, False])
    if isinstance(value, int):
        return rng.randint(0, max(2, min(8, abs(value) + 2)))
    if isinstance(value, str):
        if value and all(ch in "IVXLCDM" for ch in value):
            return value                       # a Roman numeral is not randomisable
        return "".join(rng.choice(LETTERS) for _ in range(max(1, len(value))))
    if isinstance(value, tuple):
        if not value:
            return ()
        return tuple(_like(value[rng.randrange(len(value))], rng) for _ in range(len(value)))
    return value


def _distinct(source, arglists, rng, want=12):
    """Real, distinct inputs the reference survives — never a repeat.

    The first version of this benchmark padded short example lists by repeating rows, so the
    held-out set was a subset of the shown set and four "solves" were coincidences it could not
    catch: ``trapping_rain_water`` was answered with ``return n``. Inputs are generated instead,
    and a problem that cannot reach enough of them is skipped rather than tested against itself.
    """
    fn = reference(source)
    seen, out = set(), []
    for args in list(arglists):
        key = repr(tuple(args))
        if key in seen:
            continue
        try:
            fn(*args)
        except Exception:                      # noqa: BLE001 - an unanswerable input is not one
            continue
        seen.add(key)
        out.append(tuple(args))
    tries = 0
    while len(out) < want and tries < 300:
        tries += 1
        base = arglists[rng.randrange(len(arglists))]
        cand = tuple(_like(a, rng) for a in base)
        if repr(cand) in seen:
            continue
        try:
            fn(*cand)
        except Exception:                      # noqa: BLE001
            continue
        seen.add(repr(cand))
        out.append(cand)
    return out


def _spec(name, source, inputs, params):
    fn = reference(source)
    rows = [(tuple(normalise(a) for a in args), normalise(fn(*args))) for args in inputs]
    cut = max(2, len(rows) - 2)
    return Spec.of(name, "", params, rows[:cut], rows[cut:])


def _params(inputs):
    return tuple(f"a{i}" for i in range(len(inputs[0])))


# --------------------------------------------------------------------------- #
# what she can do with a program she has never seen
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,source,arglists", HARD, ids=[case[0] for case in HARD])
def test_she_reads_and_runs_an_algorithm_she_was_never_taught(name, source, arglists, schooled):
    """The oracle is CPython, on the same inputs. Thirty algorithms, none of them hers."""
    program = read_python(source, name="f")
    for args in arglists:
        assert schooled.run(program, args) == normalise(reference(source)(*args)), (name, args)


@pytest.mark.parametrize("name,source,arglists", HARD, ids=[case[0] for case in HARD])
def test_she_can_say_what_happened_inside_it(name, source, arglists, schooled):
    """A trace has to end where running it plainly ends, or it is a story rather than a record."""
    program = read_python(source, name="f")
    steps = schooled.trace(program, arglists[0])
    assert steps and steps[-1].value == schooled.run(program, arglists[0])


def test_she_repairs_a_corrupted_algorithm_she_was_never_taught(schooled):
    """One node changed, and the fix has to run on the **held-out** pairs before it counts."""
    rng = random.Random(31)
    repaired = attempted = 0
    for name, source, arglists in HARD[:12]:
        program = read_python(source, name="f")
        inputs = _distinct(source, arglists, rng, want=8)
        if len(inputs) < 5:
            continue
        spec = _spec(name, source, inputs, _params(inputs))
        spots = [(p, n) for p, n in _positions(program.body)
                 if (isinstance(n, Lit) and isinstance(n.value, int))
                 or (isinstance(n, Call) and n.op not in ("map", "filter", "fold", "invoke"))]
        rng.shuffle(spots)
        for path, node in spots:
            if isinstance(node, Lit):
                swap = Lit(node.value + 1)
            else:
                same = [o for o in OPS if arity_of(o) == arity_of(node.op) and o != node.op
                        and o not in ("map", "filter", "fold")]
                if not same:
                    continue
                swap = Call(rng.choice(same), node.args)
            broken = Program(program.name, program.params,
                             _replace_at(program.body, path, swap), program.helpers)
            if schooled.matches(broken, spec.shown):
                continue
            attempted += 1
            fixed = schooled.repair(broken, spec, attempts=30000)
            repaired += bool(fixed.ok and schooled.check(fixed.program, spec.held_out).ok)
            break
    assert attempted >= 8
    assert repaired / attempted >= 0.7, f"repaired {repaired} of {attempted}"


# --------------------------------------------------------------------------- #
# what one demonstration buys, and what it does not
# --------------------------------------------------------------------------- #

def test_shown_once_she_writes_it_again_for_data_she_has_not_seen(schooled):
    """The inputs are generated once and **split**: the first half demonstrates, the second half
    examines, and no input appears in both. The control is the same problem, the same budget and
    the same fifty-four taught shapes, with no demonstration.

    What this measures, precisely: retention and reapplication of a shape from a single
    demonstration. It is *not* generalisation to a different variant — the constants are the same,
    and that claim is the one :mod:`nyxara.njp.school` measures over instances that differ.
    """
    rng = random.Random(23)
    shown_ok = cold_ok = total = 0
    for name, source, arglists in HARD[:14]:
        pool = _distinct(source, arglists, rng, want=12)
        if len(pool) < 10:
            continue
        params, half = _params(pool), len(pool) // 2
        lesson = _spec(name + "_lesson", source, pool[:half], params)
        exam = _spec(name, source, pool[half:], params)
        assert not (set(map(repr, lesson.shown + lesson.held_out))
                    & set(map(repr, exam.shown + exam.held_out))), f"{name}: leaked inputs"
        total += 1

        taught = Coder()
        taught.schemas = dict(schooled.schemas)
        taught.learn_python(lesson, source)
        written = taught.write(exam, attempts=30000, graft=False)
        shown_ok += bool(written.ok and taught.check(written.program, exam.held_out).ok)

        cold = Coder()
        cold.schemas = dict(schooled.schemas)
        blind = cold.write(exam, attempts=30000, graft=False)
        cold_ok += bool(blind.ok and cold.check(blind.program, exam.held_out).ok)

    assert total >= 8
    assert shown_ok / total >= 0.7, f"shown once: {shown_ok}/{total}"
    assert shown_ok > cold_ok + 2, f"shown {shown_ok}, never shown {cold_ok} — the lesson did nothing"


def test_what_she_cannot_invent_she_abstains_from_rather_than_guessing(schooled):
    """The property that makes the limit safe, and the one worth guarding.

    She solves almost none of these cold — the shapes are not hers and enumeration does not reach
    them. Asserting *how few* would be a test that fails when she improves, so what is asserted is
    that not knowing shows up as an abstention and never as a wrong program: everything she does
    write cold has to pass the held-out pairs.
    """
    rng = random.Random(41)
    wrote = 0
    for name, source, arglists in HARD[:14]:
        pool = _distinct(source, arglists, rng, want=10)
        if len(pool) < 6:
            continue
        spec = _spec(name, source, pool, _params(pool))
        blind = Coder()
        written = blind.write(spec, attempts=20000, graft=True)
        if not written.ok:
            continue                            # an abstention: the honest outcome here
        wrote += 1
        assert blind.check(written.program, spec.held_out).ok, (
            f"{name}: wrote a program that fits the shown pairs and fails held-out — "
            f"{written.program.source().splitlines()[-1].strip()}")
    assert wrote <= 6, "if she is inventing these now, this benchmark needs harder problems"


def test_the_reader_covers_the_forms_dynamic_programming_is_written_in(schooled):
    """Regression for the two gaps this benchmark found. Each ruled out a whole class of program:
    the three-way minimum is the centre of edit distance, and the descending range is the
    backwards pass that half of dynamic programming is made of."""
    assert schooled.run(read_python("def f(a, b, c):\n    return min(a, b, c)"), (5, 2, 9)) == 2
    assert schooled.run(read_python("def f(a, b, c):\n    return max(a, b, c)"), (5, 2, 9)) == 9
    assert schooled.run(read_python("def f(n):\n    return [i for i in range(n, -1, -1)]"),
                        (3,)) == (3, 2, 1, 0)
    assert schooled.run(read_python("def f(n):\n    return [i for i in range(0, n, 2)]"),
                        (7,)) == (0, 2, 4, 6)
