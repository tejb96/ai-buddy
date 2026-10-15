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
