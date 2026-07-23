"""Tests for nyxara.growth.__main__ — the Master-facing 'grow the brain' CLI (CPU/n-gram)."""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.__main__ import main


def test_grow_trains_and_promotes_on_ngram(tmp_path: Path):
    code = main(["--backend", "ngram", "--generations", "1", "--data-dir", str(tmp_path)])
    assert code == 0
    # a first model always clears the gauntlet (nothing to beat) and becomes active
    assert (tmp_path / "foundry" / "active").exists()


def test_grow_with_bench_reports_handoff(tmp_path: Path, capsys):
    code = main(["--backend", "ngram", "--generations", "1",
                 "--data-dir", str(tmp_path), "--bench"])
    assert code == 0
    out = capsys.readouterr().out
    assert "promoted" in out.lower()
    assert "handoff" in out.lower()          # the wrapper→own-AI metric is surfaced


def test_grow_distill_skips_cleanly_without_a_teacher(tmp_path: Path, capsys):
    # no real teacher is configured under the test profile -> distillation is skipped, not fatal
    code = main(["--distill", "--backend", "ngram", "--data-dir", str(tmp_path)])
    assert code == 0
    assert "skipping distillation" in capsys.readouterr().out.lower()
    assert (tmp_path / "foundry" / "active").exists()


def test_grow_tinyllama_shortcut_sets_lora_base(tmp_path: Path):
    # --tinyllama presets the LoRA backend on the TinyLlama-1.1B base; on a deps-free box it
    # degrades to the n-gram brain but still records the base in the promoted spec.
    import json

    from nyxara.growth.bootstrap import TINYLLAMA_1_1B

    code = main(["--tinyllama", "--generations", "1", "--data-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "foundry" / "active").exists()
    spec = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"][0]["spec"]
    assert spec["base_model"] == TINYLLAMA_1_1B
    assert spec["kind"] == "lora"


def test_grow_qwen_alias_still_works(tmp_path: Path):
    # the old --qwen spelling stays as a deprecated alias for --tinyllama
    import json

    from nyxara.growth.bootstrap import TINYLLAMA_1_1B

    code = main(["--qwen", "--generations", "1", "--data-dir", str(tmp_path)])
    assert code == 0
    spec = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"][0]["spec"]
    assert spec["base_model"] == TINYLLAMA_1_1B


def test_grow_does_not_mutate_global_settings(tmp_path: Path):
    from nyxara.kernel.config import get_settings
    before = get_settings().foundry.enabled
    main(["--backend", "ngram", "--data-dir", str(tmp_path)])
    # the CLI works on a deep copy, so the process-global config is untouched
    assert get_settings().foundry.enabled == before
