"""The model manager without Lemonade: speech-to-text choices, where models run, the NPU lock, exiting."""

from __future__ import annotations

import asyncio

import pytest

from ai_buddy.config import Config
from ai_buddy.lemonade import LemonadeError
from ai_buddy.models import ModelManager, on_npu


class FakeLemonade:
    def __init__(self):
        self.loaded: dict[str, dict] = {}
        self.unloaded_all = False

    async def loaded_models(self):
        return dict(self.loaded)

    async def load(self, model, ctx_size=None, **options):
        self.loaded[model] = options
        self.ctx = getattr(self, "ctx", {})
        self.ctx[model] = ctx_size

    async def transcribe(self, wav, model, language="en"):
        await asyncio.sleep(0.01)
        return f"heard by {model}"

    async def unload_all(self):
        self.unloaded_all = True
        self.loaded.clear()


def test_flm_models_run_on_the_npu():
    assert on_npu("whisper-v3-turbo-FLM") and on_npu("qwen3.5-9b-FLM")
    assert not on_npu("Whisper-Large-v3-Turbo") and not on_npu("kokoro-v1")


async def test_whisper_cpp_loads_with_its_backend_and_skips_the_npu_lock():
    cfg = Config()
    lm = FakeLemonade()
    models = ModelManager(lm, cfg, stt="Whisper-Large-v3-Turbo")
    assert models.device("stt") == "CPU" and models.device("llm") == "NPU" and models.device("tts") == "CPU"
    await models.ensure("stt")
    assert lm.loaded["Whisper-Large-v3-Turbo"] == {"whispercpp_backend": "cpu"}
    async with models._npu:          # an LLM stream holds the NPU: whisper.cpp on the CPU doesn't wait for it
        assert await asyncio.wait_for(models.transcribe(b"\0\0" * 1600), 1) == "heard by Whisper-Large-v3-Turbo"
    await models.switch_stt("whisper-v3-turbo-FLM")
    assert lm.loaded["whisper-v3-turbo-FLM"] == {} and models.device("stt") == "NPU"
    async with models._npu:          # FLM Whisper must wait for the NPU (two FLM jobs at once corrupt both)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(models.transcribe(b"\0\0" * 1600), 0.2)


async def test_exit_unloads_everything_and_loads_nothing_more():
    lm = FakeLemonade()
    models = ModelManager(lm, Config())
    await models.ensure("stt", "llm", "tts")
    await models.unload_all()
    assert lm.unloaded_all and not lm.loaded and set(models.state.values()) == {"standby"}
    with pytest.raises(LemonadeError, match="shutting down"):
        await models.ensure("llm")


async def test_the_embedding_model_loads_on_the_cpu_with_a_small_context():
    cfg = Config()
    cfg.models.embed = "Qwen3-Embedding-0.6B-GGUF"
    lm = FakeLemonade()
    models = ModelManager(lm, cfg)
    await models.ensure("llm", "embed")
    assert lm.loaded["Qwen3-Embedding-0.6B-GGUF"] == {"llamacpp_backend": "cpu", "llamacpp_args": "--threads 4"}
    assert lm.ctx == {cfg.models.llm: 32768, "Qwen3-Embedding-0.6B-GGUF": 8192}
    assert models.device("embed") == "CPU"


class OutdatedLemonade(FakeLemonade):
    """Lemonade after an upgrade: FLM needs updating before its models load (they are then "not found")."""

    def __init__(self):
        super().__init__()
        self.flm = "update_required"
        self.installs: list[tuple[str, str]] = []

    async def backends(self):
        return {"flm": {"npu": {"state": self.flm, "version": "v1.0.6",
                                "release_url": "https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.7"}},
                "kokoro": {"cpu": {"state": "installed", "version": "b21"}},
                "llamacpp": {"cpu": {"state": "update_required", "version": "b10825"},
                             "vulkan": {"state": "installable"}}}

    async def install_backend(self, recipe, backend):
        self.installs.append((recipe, backend))
        for pct in (10, 60, 100):
            yield {"percent": pct}
        if recipe == "flm":
            self.flm = "installed"

    async def load(self, model, ctx_size=None, **options):
        if on_npu(model) and self.flm != "installed":
            raise LemonadeError(f"Loading {model} failed: model_not_found. The FLM backend is not ready")
        await super().load(model, ctx_size, **options)


async def test_a_failed_load_finds_the_backend_update_and_updating_fixes_it():
    cfg = Config()
    cfg.models.embed = "Qwen3-Embedding-0.6B-GGUF"
    lm = OutdatedLemonade()
    models = ModelManager(lm, cfg)
    seen: list[dict] = []
    models.subscribe(lambda snap: seen.append(snap["backends"]["updating"]))
    with pytest.raises(LemonadeError):
        await models.ensure("stt", "llm", "tts")
    needed = models.snapshot()["backends"]["needed"]
    # FLM for Whisper and the LLM; llama.cpp's CPU backend for note search (vulkan is not ours, so not asked).
    assert [(u["spec"], u["action"], u["version"], u["latest"]) for u in needed] == [
        ("flm:npu", "update", "v1.0.6", "v1.0.7"), ("llamacpp:cpu", "update", "b10825", "")]
    assert needed[0]["roles"] == ["stt", "llm"] and needed[1]["roles"] == ["embed"]

    await models.update_backends()
    assert lm.installs == [("flm", "npu"), ("llamacpp", "cpu")]
    assert {"spec": "flm:npu", "label": "FastFlowLM (NPU)", "percent": 60} in seen
    assert models.updating is None and models.backend_error is None
    assert [u["spec"] for u in models.backend_updates] == ["llamacpp:cpu"]   # the fake only "fixes" FLM
    await models.ensure("stt", "llm", "tts")
    assert {"whisper-v3-turbo-FLM", "kokoro-v1"} <= set(lm.loaded)


async def test_a_failed_backend_update_is_reported_and_can_be_retried():
    class Failing(OutdatedLemonade):
        async def install_backend(self, recipe, backend):
            yield {"percent": 5}
            raise LemonadeError("Installing flm:npu failed: disk full")

    models = ModelManager(Failing(), Config())
    await models.check_backends()
    await models.update_backends()
    snap = models.snapshot()["backends"]
    assert snap["error"] == "Installing flm:npu failed: disk full" and snap["updating"] is None
    assert snap["needed"][0]["spec"] == "flm:npu"


async def test_backend_check_keeps_quiet_when_lemonade_is_down():
    class Down(FakeLemonade):
        async def backends(self):
            raise LemonadeError("Lemonade server not reachable")

    models = ModelManager(Down(), Config())
    assert await models.check_backends() == []
