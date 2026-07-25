"""Tests for the DistilGPT-2 backend — NYXARA's sole real LLM (local, in-process).

No heavy deps needed: generation-control plumbing is tested against a fake tokenizer/model
injected through ``_ensure_model``, so every knob (sampling policy, top_k, repetition
penalty, beams, seed, stops, JSON nudge, input truncation) is asserted without downloading
weights. Availability tests stay honest on a bare machine.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, LLMRequest, QwenProvider
from nyxara.kernel.errors import LLMError


def _settings() -> NyxaraSettings:
    return NyxaraSettings.for_profile(Profile.TEST)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


# -------------------- enum / registry / config routing -------------------- #
def test_provider_enum_includes_airouter_tool():
    assert LLMProvider.QWEN.value == "qwen"
    # airouter is the one CLOUD tool; the rest run locally.
    assert {p.value for p in LLMProvider} == {"auto", "airouter", "qwen", "self", "native"}


def test_registered_in_facade():
    status = LLM(settings=_settings()).provider_status()
    assert set(status) == {"airouter", "qwen", "self", "native"}


def test_config_routes_model_and_no_keys():
    s = _settings()
    s.llm.provider = LLMProvider.QWEN
    assert s.llm.active_model() == s.llm.qwen_model
    assert s.llm.qwen_model == "distilgpt2"
    assert s.llm.active_key() is None


def test_default_provider_is_auto():
    # the shipped default walks the auto ladder self→qwen→mock: her own promoted
    # weights serve first when they exist; absent every backend it degrades to the mock,
    # so this is safe on a bare box.
    s = NyxaraSettings.for_profile(Profile.DEV)
    assert s.llm.provider is LLMProvider.AUTO


def test_quant_flags_mutually_exclusive():
    s = _settings()
    s.llm.qwen_load_in_4bit = True
    with pytest.raises(Exception):
        s.llm.qwen_load_in_8bit = True


# -------------------- honest availability -------------------- #
def test_availability_is_honest():
    prov = QwenProvider(_settings())
    assert prov.available() is _has_torch()


def test_unavailable_when_adapter_configured_without_peft(tmp_path):
    s = _settings()
    s.llm.qwen_adapter_path = tmp_path / "adapter"
    prov = QwenProvider(s)
    try:
        import peft  # noqa: F401
        has_peft = True
    except Exception:
        has_peft = False
    assert prov.available() is (_has_torch() and has_peft)


def test_complete_with_raises_when_unavailable():
    if _has_torch():
        pytest.skip("torch installed; the unavailable path is not reachable here")
    llm = LLM(settings=_settings())
    with pytest.raises(LLMError):
        llm.complete_with("qwen", LLMRequest.from_prompt("hi"))


def test_facade_falls_back_to_native_when_unavailable():
    if _has_torch():
        pytest.skip("torch installed; fallback path not exercised")
    s = _settings()
    s.llm.provider = LLMProvider.QWEN
    out = LLM(settings=s).generate("hello there")
    assert isinstance(out, str) and out.strip()  # her native own-brain answered, no crash


# -------------------- generation-control plumbing (no real weights) -------------------- #
class _FakeTensor2D:
    """A minimal stand-in for a (1, n) token tensor supporting the ops _complete uses."""

    def __init__(self, ids):
        self.ids = list(ids)

    @property
    def shape(self):
        return (1, len(self.ids))

    def __getitem__(self, key):
        # inputs["input_ids"][:, -budget:] -> truncated 2D tensor
        if isinstance(key, tuple):
            _, sl = key
            return _FakeTensor2D(self.ids[sl])
        # generated[0] -> the row as a 1-D tensor
        if key == 0:
            return _FakeTensor1D(self.ids)
        raise KeyError(key)

    def to(self, _device):
        return self


class _FakeTensor1D:
    def __init__(self, ids):
        self.ids = list(ids)

    @property
    def shape(self):
        return (len(self.ids),)

    def __getitem__(self, sl):
        return _FakeTensor1D(self.ids[sl])


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, n_tokens=10):
        self.n_tokens = n_tokens
        self.chat_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.chat_calls.append(messages)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, texts, return_tensors="pt"):
        return {"input_ids": _FakeTensor2D(range(self.n_tokens)),
                "attention_mask": _FakeTensor2D([1] * self.n_tokens)}

    def decode(self, tensor, skip_special_tokens=True):
        return "GENERATED " + "x" * len(tensor.ids)


class _FakeModel:
    device = "cpu"

    def __init__(self, new_tokens=5):
        self.new_tokens = new_tokens
        self.gen_kwargs = None

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.gen_kwargs = kwargs
        return _FakeTensor2D(list(input_ids.ids) + list(range(self.new_tokens)))


class _FakeTorch:
    seeds = []

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def no_grad(self):
        return self._NoGrad()

    def manual_seed(self, seed):
        _FakeTorch.seeds.append(seed)


def _fake_provider(settings=None, *, n_prompt_tokens=10, n_new_tokens=5):
    prov = QwenProvider(settings or _settings())
    tok = _FakeTokenizer(n_prompt_tokens)
    model = _FakeModel(n_new_tokens)
    prov._model, prov._tokenizer = model, tok
    prov._loaded_key = (prov.default_model(), "")
    prov._torch = _FakeTorch()
    prov._ensure_model = lambda _m: (model, tok)  # bypass the real HF load
    return prov, tok, model


def test_gen_kwargs_merge_request_and_config():
    s = _settings()
    s.llm.qwen_top_k = 40
    s.llm.qwen_repetition_penalty = 1.3
    s.llm.qwen_no_repeat_ngram_size = 4
    s.llm.qwen_min_new_tokens = 2
    prov, tok, model = _fake_provider(s)
    req = LLMRequest.from_prompt("hi", temperature=0.5, top_p=0.9, max_tokens=64)
    kw = prov._gen_kwargs(req, tok)
    assert kw["max_new_tokens"] == 64
    assert kw["do_sample"] is True and kw["temperature"] == 0.5 and kw["top_p"] == 0.9
    assert kw["top_k"] == 40 and kw["repetition_penalty"] == 1.3
    assert kw["no_repeat_ngram_size"] == 4 and kw["min_new_tokens"] == 2
    assert kw["pad_token_id"] == 0 and kw["eos_token_id"] == 2


def test_do_sample_auto_greedy_drops_sampling_knobs():
    prov, tok, _ = _fake_provider()
    kw = prov._gen_kwargs(LLMRequest.from_prompt("hi", temperature=0.0), tok)
    assert kw["do_sample"] is False
    assert "temperature" not in kw and "top_p" not in kw and "top_k" not in kw


def test_do_sample_never_forces_greedy():
    s = _settings()
    s.llm.qwen_do_sample = "never"
    prov, tok, _ = _fake_provider(s)
    kw = prov._gen_kwargs(LLMRequest.from_prompt("hi", temperature=0.9), tok)
    assert kw["do_sample"] is False


def test_do_sample_always_forces_sampling():
    s = _settings()
    s.llm.qwen_do_sample = "always"
    prov, tok, _ = _fake_provider(s)
    kw = prov._gen_kwargs(LLMRequest.from_prompt("hi", temperature=0.0), tok)
    assert kw["do_sample"] is True and kw["temperature"] == pytest.approx(1e-3)


def test_top_k_zero_is_omitted():
    s = _settings()
    s.llm.qwen_top_k = 0
    prov, tok, _ = _fake_provider(s)
    kw = prov._gen_kwargs(LLMRequest.from_prompt("hi", temperature=0.7), tok)
    assert "top_k" not in kw


def test_beam_search_carries_length_penalty():
    s = _settings()
    s.llm.qwen_num_beams = 4
    s.llm.qwen_length_penalty = 0.8
    prov, tok, _ = _fake_provider(s)
    kw = prov._gen_kwargs(LLMRequest.from_prompt("hi"), tok)
    assert kw["num_beams"] == 4 and kw["length_penalty"] == 0.8


def test_chat_template_and_system_message():
    s = _settings()
    s.llm.qwen_use_chat_template = True   # DistilGPT-2 defaults to flat; opt into the chat path here
    prov, tok, model = _fake_provider(s)
    req = LLMRequest.from_prompt("who are you?", system="You are NYXARA.")
    text, finish, usage, raw = prov._complete(req, "m")
    assert tok.chat_calls, "apply_chat_template must be used"
    roles = [m["role"] for m in tok.chat_calls[0]]
    assert roles[0] == "system" and "NYXARA" in tok.chat_calls[0][0]["content"]
    assert raw["qwen"] is True


def test_json_mode_injects_json_nudge():
    s = _settings()
    s.llm.qwen_use_chat_template = True   # exercise the chat-template JSON nudge path
    prov, tok, _ = _fake_provider(s)
    prov._complete(LLMRequest.from_prompt("give json", json_mode=True), "m")
    system = tok.chat_calls[0][0]
    assert system["role"] == "system" and "JSON" in system["content"]


def test_flat_prompt_when_chat_template_disabled():
    s = _settings()
    s.llm.qwen_use_chat_template = False
    prov, tok, _ = _fake_provider(s)
    prov._complete(LLMRequest.from_prompt("plain prompt"), "m")
    assert tok.chat_calls == []  # template never applied


def test_seed_is_applied():
    prov, tok, _ = _fake_provider()
    _FakeTorch.seeds.clear()
    prov._complete(LLMRequest.from_prompt("hi", seed=1234), "m")
    assert 1234 in _FakeTorch.seeds


def test_input_truncated_to_context_budget():
    s = _settings()
    s.llm.qwen_max_input_tokens = 64
    prov, tok, model = _fake_provider(s, n_prompt_tokens=500)
    _, _, usage, _ = prov._complete(LLMRequest.from_prompt("long", max_tokens=32), "m")
    assert usage.prompt_tokens == 64 - 32  # left-truncated to max_input - max_new


def test_stop_sequences_truncate_and_set_finish_reason():
    prov, tok, model = _fake_provider(n_new_tokens=5)
    tok.decode = lambda t, skip_special_tokens=True: "hello STOP world"
    text, finish, usage, _ = prov._complete(
        LLMRequest.from_prompt("hi", stop=("STOP",), max_tokens=64), "m")
    assert text == "hello" and finish == "stop"


def test_finish_reason_length_when_budget_filled():
    prov, tok, model = _fake_provider(n_new_tokens=8)
    text, finish, usage, _ = prov._complete(
        LLMRequest.from_prompt("hi", max_tokens=8), "m")
    assert finish == "length" and usage.completion_tokens == 8


def test_usage_is_token_accurate():
    prov, tok, model = _fake_provider(n_prompt_tokens=10, n_new_tokens=5)
    _, _, usage, _ = prov._complete(LLMRequest.from_prompt("hi", max_tokens=99), "m")
    assert usage.prompt_tokens == 10 and usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_adapter_path_recorded_in_raw(tmp_path):
    s = _settings()
    s.llm.qwen_adapter_path = tmp_path / "adapter"
    prov, tok, _ = _fake_provider(s)
    _, _, _, raw = prov._complete(LLMRequest.from_prompt("hi"), "m")
    assert raw["adapter"] == str(tmp_path / "adapter")


# -------------------- env-driven knob overrides -------------------- #
def test_env_overrides_generation_knobs(monkeypatch):
    monkeypatch.setenv("NYXARA_LLM__PROVIDER", "qwen")
    monkeypatch.setenv("NYXARA_LLM__QWEN_TOP_K", "20")
    monkeypatch.setenv("NYXARA_LLM__QWEN_REPETITION_PENALTY", "1.25")
    monkeypatch.setenv("NYXARA_LLM__QWEN_DO_SAMPLE", "never")
    monkeypatch.setenv("NYXARA_LLM__QWEN_DTYPE", "float16")
    s = NyxaraSettings()
    assert s.llm.provider is LLMProvider.QWEN
    assert s.llm.qwen_top_k == 20
    assert s.llm.qwen_repetition_penalty == 1.25
    assert s.llm.qwen_do_sample == "never"
    assert s.llm.qwen_dtype == "float16"


def test_env_overrides_foundry_knobs(monkeypatch):
    monkeypatch.setenv("NYXARA_FOUNDRY__LORA_TARGET_MODULES", '["c_attn","c_fc"]')
    monkeypatch.setenv("NYXARA_FOUNDRY__GRAD_ACCUM_STEPS", "8")
    monkeypatch.setenv("NYXARA_FOUNDRY__LR_SCHEDULER", "cosine")
    monkeypatch.setenv("NYXARA_FOUNDRY__TRAIN_EPOCHS", "3")
    s = NyxaraSettings()
    assert s.foundry.lora_target_modules == ["c_attn", "c_fc"]
    assert s.foundry.grad_accum_steps == 8
    assert s.foundry.lr_scheduler == "cosine"
    assert s.foundry.train_epochs == 3


def test_foundry_default_base_and_targets():
    s = NyxaraSettings()
    # default base is her single small base: DistilGPT-2
    assert s.foundry.base_model == "distilgpt2"
    # DistilGPT-2 (GPT-2 arch) uses Conv1D names — the default pins attention (c_attn) + MLP (c_fc/c_proj)
    assert s.foundry.lora_target_modules == ["c_attn", "c_proj", "c_fc"]


def test_foundry_quant_flags_mutually_exclusive():
    s = NyxaraSettings()
    s.foundry.load_in_4bit = True
    with pytest.raises(Exception):
        s.foundry.load_in_8bit = True
