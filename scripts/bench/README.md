# Benchmarks (Phase 0)

Scripts behind the numbers in [docs/phase0-findings.md](../../docs/phase0-findings.md). They talk to Lemonade on
`localhost:13305` and load the models they need. Run them from this folder with the project's environment:

```bash
cd scripts/bench
../../.venv/bin/python stt_length.py whisper-v3-turbo-FLM
```

Synthesized speech (Kokoro) is cached in `audio_cache/`, JSON results go to `results/`; both are git-ignored.
Keep runs short and spaced out: long back-to-back loops heat the laptop up. Never send Whisper and LLM
requests at the same time (FLM 1.0.6 corrupts both results).

| Script | What it measures |
|---|---|
| `stt_length.py <model>` | Speech-to-text latency vs. clip length (the fixed per-call cost). |
| `stt_accuracy.py <model>...` | WER on chat-style prompts (clean and white noise), disfluent answers and real LibriSpeech speech, plus filler retention. Env: `PROMPT` (Whisper prompt), `LOAD_ARGS` (JSON load options), `MAX_S` (longest LibriSpeech clip), `TAG`. |
| `stt_fillers.py <model>...` | Filler words kept, with and without a disfluent Whisper prompt. |
| `stt_robust.py <model>...` | WER and fillers under babble noise (10 dB) and reverb (0.5 s), with the filler prompt. |
| `stt_chunking.py <model> [min_s] [pause_s]` | Accuracy cost of cutting speech at pauses, and the leftover tail. |
| `stt_realtime.py <model> [vad\|manual]` | Lemonade's `/realtime` WebSocket: interim and final transcript timing. |
| `extract_librispeech.py <parquet> librispeech` | Converts the LibriSpeech sample to PCM + manifest (see below). |
| `audio_llm.py whisper:<llm>\|audio:<llm>\|heard:<llm> [sections]` | Same spoken test set through Whisper + a text LLM, or straight into an audio-input LLM (Gemma 4): tool choice, multi-turn no-guessing, understanding/reply style, transcription. `heard:` + section `heard`: the model writes a `<heard>` transcript before replying; scored against Whisper on the same clips (`LS_N` LibriSpeech clips). `REDUCED=1` runs ~20 cases. Records temperatures. |
| `gemma_audio_cache.py [llm]` | Context cost of audio input (tokens per second) and which history format keeps FLM's prompt cache (audio kept vs. replaced by text). |
| `tool_choice.py [llm]` | Whether FLM honors `tool_choice`, whether it changes the prompt, and the cache after an aborted stream. |
| `stability.py [llm] [whisper\|heard]` | A 20-turn spoken conversation with the real tools and a sample notes folder (turns 8 s apart), then a practice talk transcribed by Whisper and feedback from the LLM: crashes, evictions (PIDs), prompt-cache misses, drift, tool use, temperatures. Whisper never runs alongside the LLM (two FLM jobs at once corrupt each other). |
| `llm_bench.py <model> [ctx] [sections]` | Time to first spoken word in a growing conversation, prefill speed, 24 tool-calling prompts, tool result -> spoken answer. |
| `llm_tune.py <model> <ctx> '<flm_args>'` | First-token latency under FLM launch args (`--pmode turbo`, `--prefill-chunk-len N`). |
| `llm_cache_greeting.py` | Whether a generated greeting primes FLM's prompt cache for the first user turn. |
| `llm_cache_tool_turn.py [model]` | Whether the turn after a tool call hits the prompt cache (native tool messages, `preserve_thinking`, results as a user message). |
| `llm_preamble.py <model>` | Whether asking the model to announce tool calls ("let me look that up") hurts tool use. |
| `llm_strict_prompt.py <model>` | Tool calling with a stricter system prompt (date + "don't answer from memory"). |
| `coresidency.py <llm> <stt> [ctx] [embed]` | Whether LLM, speech-to-text and an embedding model stay loaded together (PIDs, memory). |
| `e2e_latency.py <ws_url> <origin>` | The running app, like the browser: "stop speaking -> transcript" and "-> first audio", with server timing. |
| `app_smoke.py <port> [turn...]` | Scripted conversation against the running app (greeting, plain, calc, notes, web, time, news turns). |
| `lemonade_logs.py [regex] [n]` | Recent Lemonade server log lines (load/evict/errors) over its log WebSocket. |

The real-speech set is Hugging Face's `hf-internal-testing/librispeech_asr_dummy` (73 utterances, 9 MB):

```bash
mkdir -p /tmp/ls && curl -sL -o /tmp/ls/validation.parquet \
  https://huggingface.co/api/datasets/hf-internal-testing/librispeech_asr_dummy/parquet/clean/validation/0.parquet
uv run --no-project --with pyarrow python -I extract_librispeech.py /tmp/ls/validation.parquet librispeech
```

The whisper.cpp (iGPU) runs in the findings used `lemonade backends install whispercpp:vulkan` and
`lemonade pull Whisper-Small` etc.; those were removed afterwards to keep everything on the NPU.
