"""Tests for prompt-grammar evolution — growth/prompt_grammar.py (Part L). Network-free."""
from __future__ import annotations

from nyxara.growth.prompt_grammar import PromptGrammar, PromptGrammarEvolver


def test_render_includes_selected_directives():
    g = PromptGrammar(directives=frozenset({"double_check", "answer_only"}))
    system, user = g.render("what is 6*7?")
    assert "Double-check" in system and "final answer" in system
    assert user == "what is 6*7?"


def _scripted_model(system: str, user: str) -> str:
    """A fake model whose correctness DEPENDS on the grammar: it only answers correctly for the
    arithmetic battery when the grammar told it to double-check."""
    import re
    m = re.search(r"(\d+)\s*\*\s*(\d+)", user)
    if not m:
        return "?"
    correct = str(int(m.group(1)) * int(m.group(2)))
    if "Double-check" in system:
        return correct
    return str(int(correct) + 1)          # an off-by-one error without the double-check directive


def test_evolver_improves_verified_correct_rate():
    tasks = ["what is 6*7?", "what is 8*9?", "what is 3*4?"]
    ev = PromptGrammarEvolver(population=8, generations=8, seed=3)

    baseline = ev.fitness(PromptGrammar(directives=frozenset()), tasks, _scripted_model)
    best, best_fit = ev.evolve(tasks, _scripted_model)

    assert best_fit > baseline                      # search found a better grammar
    assert best_fit >= 0.99                          # and it drives verified-correct to ~100%
    assert "double_check" in best.directives         # discovered the directive that actually helps


def test_deterministic_under_seed():
    tasks = ["what is 6*7?", "what is 8*9?"]
    a = PromptGrammarEvolver(seed=1).evolve(tasks, _scripted_model)
    b = PromptGrammarEvolver(seed=1).evolve(tasks, _scripted_model)
    assert a[0].to_dict() == b[0].to_dict() and a[1] == b[1]
