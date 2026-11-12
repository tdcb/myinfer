"""DeepSeek V4.1 backend adapter, exercised against a stub of oMLX's package (no Metal here)."""
import sys
import types

import mlx.core as mx
import pytest

from m5engine import deepseek_v41 as D


class _Slots:
    def __init__(self, hits, misses, capacity, count):
        self.hits, self.misses = hits, misses
        self.plan = types.SimpleNamespace(capacity=capacity, count=count)


class _Offloaded:
    def __init__(self, slots):
        self.slots = slots


class _Scope:
    active = False
    entered = []

    def __call__(self):
        outer = self

        class _CM:
            def __enter__(self_inner):
                outer.active = True
                outer.entered.append("in")

            def __exit__(self_inner, *a):
                outer.active = False
                outer.entered.append("out")

        return _CM()


class _Model:
    """Enough of oMLX's mlx-lm facade for the adapter: make_cache, named_modules, prompt scope."""

    def __init__(self, scope):
        self._scope = scope
        self.calls = []
        self._mods = [("l0", _Offloaded(_Slots(3, 1, 8, 32))), ("l1", _Offloaded(_Slots(5, 3, 8, 32))), ("x", object())]

    def _omlx_prompt_scope(self):
        return self._scope()

    def named_modules(self):
        return self._mods

    def make_cache(self):
        return [_Cache() for _ in range(2)]

    def __call__(self, ids, cache=None):
        n = ids.shape[1]
        self.calls.append((n, self._scope.active))
        for c in cache:
            c.state = ([mx.zeros((1, 100 + 10 * n)), (mx.zeros((1, 4)),)], None, None)  # nested like oMLX's
        return mx.zeros((1, n, 4))


class _Cache:
    state = []


@pytest.fixture
def stub_omlx(monkeypatch):
    """Install a fake ``omlx.patches`` with the three modules the adapter touches."""
    mo = types.ModuleType("omlx.patches.deepseek_v41.moe_offload")
    mo.OffloadedExpert = _Offloaded
    mo.fit_resident_fraction = lambda path, budget, engram_ssd_offload=True: (None if budget < 10 else min(1.0, budget / 100))
    mo.admission_bytes = lambda path, fraction, engram_ssd_offload=True: int(20 + 80 * fraction)
    res = types.ModuleType("omlx.patches.deepseek_v41.residency")
    res.deepseek_v41_residency_estimate = lambda path: types.SimpleNamespace(engram_bytes=7)
    loading = types.ModuleType("omlx.patches.deepseek_v41.loading")
    facade = types.ModuleType("omlx.patches.deepseek_v41_mlx_lm.model")
    for name, m in {
        "omlx": types.ModuleType("omlx"), "omlx.patches": types.ModuleType("omlx.patches"),
        "omlx.patches.deepseek_v41": types.ModuleType("omlx.patches.deepseek_v41"),
        "omlx.patches.deepseek_v41.moe_offload": mo, "omlx.patches.deepseek_v41.residency": res,
        "omlx.patches.deepseek_v41.loading": loading, "omlx.patches.deepseek_v41_mlx_lm.model": facade,
        "omlx.patches.deepseek_v41_mlx_lm": types.ModuleType("omlx.patches.deepseek_v41_mlx_lm"),
    }.items():
        monkeypatch.setitem(sys.modules, name, m)
    return types.SimpleNamespace(moe=mo, loading=loading, facade=facade)


def test_available_reports_missing_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "omlx", None)  # import raises ImportError
    ok, detail = D.available()
    assert not ok and "Error" in detail
    with pytest.raises(RuntimeError, match="install_deepseek_backend"):
        D.require()


def test_plan_residency_fits_budget(stub_omlx, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v41", "text_config": {"n_routed_experts": 384}}')
    plan = D.plan_residency(tmp_path, 50)
    assert plan.resident_fraction == 0.5 and plan.resident_experts == 192 and plan.experts_per_layer == 384
    assert plan.admission_bytes == 60 and plan.full_admission_bytes == 100 and plan.engram_bytes == 7
    d = plan.to_dict()
    assert d["backend"] == "omlx" and d["engram_on_ssd"] and d["resident_experts_per_layer"] == 192
    pinned = D.plan_residency(tmp_path, 50, fraction=0.25)
    assert pinned.resident_fraction == 0.25 and pinned.resident_experts == 96
    with pytest.raises(RuntimeError, match="does not fit"):
        D.plan_residency(tmp_path, 5)


def test_expert_stats_aggregates_layers(stub_omlx):
    model = _Model(_Scope())
    st = D.expert_stats(model)
    assert st == {"backend": "omlx", "moe_layers": 2, "hits": 8, "misses": 4, "hit_rate": 0.6667,
                  "resident_experts": 16, "total_experts": 64, "resident_fraction": 0.25}
    assert D.expert_stats(types.SimpleNamespace(named_modules=lambda: [])) is None


def test_measure_cache_bytes_separates_fixed_and_per_token():
    model = _Model(_Scope())
    per_tok, fixed = D.measure_cache_bytes(model, tokens=8)
    # Two layers, each 10 floats (40 bytes) per token plus (100 + 4) floats fixed.
    assert per_tok == pytest.approx(2 * 40)
    assert fixed == 2 * (104 * 4)


def test_prefill_scope_covers_only_the_first_step():
    scope = _Scope()
    model = _Model(scope)
    seen = []

    def steps():
        seen.append(("prefill", scope.active))
        yield "tok1"
        seen.append(("decode", scope.active))
        yield "tok2"

    out = list(D.generate_with_prefill_scope(model, steps()))
    assert out == ["tok1", "tok2"]
    assert seen == [("prefill", True), ("decode", False)]
    assert scope.entered == ["in", "out"]
    assert list(D.generate_with_prefill_scope(model, iter(()))) == []


def test_load_uses_omlx_loader_and_facade(stub_omlx, tmp_path):
    class HF:
        eos_token_id = 1
        bos_token_id = 0
        chat_template = None
        apply_chat_template = None  # newer TokenizerWrapper reads it at construction

        def encode(self, text, add_special_tokens=False):
            return [3, 4, 5]

        def decode(self, ids, **kw):
            return "x"

        def convert_ids_to_tokens(self, ids, **kw):
            return ["t"] * len(ids)

        def convert_tokens_to_ids(self, t):
            return 0

        vocab_size = 64

        def get_vocab(self):
            return {"a": 0}

        @property
        def all_special_tokens(self):
            return []

        @property
        def all_special_ids(self):
            return []

        clean_up_tokenization_spaces = False

    class Processor:
        tokenizer = HF()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            Processor.last = (messages, kw)
            return [7, 8, 9] if tokenize else "text"

    class Raw:
        pass

    class Facade(Raw):
        pass

    calls = {}

    def fake_load(path, **kw):
        calls.update(kw)
        return Raw(), Processor()

    stub_omlx.loading.load = fake_load
    stub_omlx.facade.Model = Facade
    plan = D.ResidencyPlan(0.5, 384, 192, 60, 100, 7, True, 50)
    loaded = D.load(tmp_path, plan, ced_prefill=True)
    assert calls == {"engram_ssd_offload": True, "moe_expert_offload_resident_fraction": 0.5,
                     "ced_prefill": True, "preserve_mtp": None}
    assert type(loaded.model) is Facade
    assert loaded.tokenizer.eos_token_ids == {1}
    ids = D.encode_chat(loaded, [{"role": "user", "content": "hi"}], {"reasoning_effort": "low"})
    assert ids == [7, 8, 9] and Processor.last[1]["enable_thinking"] is True
    assert Processor.last[1]["reasoning_effort"] == "low"
    full = D.ResidencyPlan(1.0, 384, 384, 100, 100, 7, False, 200)
    D.load(tmp_path, full)
    assert calls["moe_expert_offload_resident_fraction"] is None and calls["engram_ssd_offload"] is False
