# AI Buddy

> **Status (branch `kira-agent`):** the assistant is now **Kira**, an agent with web search, notes search and other
> tools, running one model (`qwen3.6-moe-35b-a3b-FLM`) for everything. This README still describes the previous
> version in places; see [docs/phase0-findings.md](docs/phase0-findings.md) for the current state, benchmarks,
> known issues and what's left to do.

A voice companion that runs entirely on this laptop. Talk to it out loud to chat, rehearse a presentation, or practice a conversation (job interview, tough conversation, small talk). After practice sessions it gives you feedback on filler words, pace, long pauses, rambling, and clarity, plus 2–3 specific things to improve. Transcripts, metrics, and feedback are saved locally so you can track progress.

Everything runs locally through [Lemonade](https://lemonade-server.ai/). The LLMs and speech-to-text are FastFlowLM (FLM) models on the Ryzen AI NPU; text-to-speech is Kokoro on the CPU. The app makes no network calls except to Lemonade on `localhost`.

| Role | Default model | Runs on |
|---|---|---|
| Speech-to-text | `whisper-v3-turbo-FLM` | NPU |
| Conversation | `qwen3-it-4b-FLM` (non-thinking) | NPU |
| Feedback coach | `qwen3.6-moe-35b-a3b-FLM` | NPU |
| Voice | `kokoro-v1` | CPU |

## Setup

Tested on a ThinkPad P14s Gen 6 AMD (Ryzen AI 7 PRO 350, XDNA 2 NPU, 64 GB RAM), Ubuntu 26.04.1, kernel 7.0, GNOME 50 on Wayland, Lemonade 2026.41.1 with its bundled FLM v1.0.6. No sudo or reboot was needed: the in-tree `amdxdna` driver works, and FLM raises its own memlock limit when Lemonade launches it.

1. **Lemonade running with the NPU visible:**
   ```bash
   systemctl status lemond          # should be active (running)
   lemonade status                  # "Server is running on port 13305"
   ```
2. **Models and the Kokoro backend** (skip any you already have):
   ```bash
   lemonade backends install kokoro:cpu
   lemonade pull whisper-v3-turbo-FLM
   lemonade pull qwen3-it-4b-FLM
   lemonade pull qwen3.6-moe-35b-a3b-FLM
   lemonade pull kokoro-v1
   ```
3. **Python environment** (uses [uv](https://docs.astral.sh/uv/); Python 3.13 is pinned in `.python-version`):
   ```bash
   cd ~/Documents/projects/ai-buddy
   uv sync
   ```
4. **Check everything:** `scripts/phase0/check_env.sh` reports on the server, NPU, FLM, and models. `flm validate` inside it flags an 8 MB memlock limit. That limit belongs to your shell, not the Lemonade service, so you can ignore it unless you want to run `flm` by hand.
5. **Unmute your speakers.** The speaker output was muted at 0% when this was set up.

## Run

```bash
uv run ai-buddy
```

This starts the app on `http://127.0.0.1:8765` and opens it in its own Chrome app window. The first start takes about 10 s to load and warm the models. The first time you talk, Chrome asks for microphone access; allow it, and the dedicated profile remembers your choice. Closing the window stops the app. Running it again while it's open just opens another window onto the running app.

Options: `--no-window` (server only; open the URL yourself), `--port 9000`, `--config path/to/config.toml`.

To get an **AI Buddy** entry in GNOME's Activities search, run `scripts/install-desktop-entry.sh` (`--uninstall` removes it).

## How to use it

**Chat.** Hold **Space** (or press and hold the round button), talk, and let go. Kira answers out loud and starts speaking while the rest of the reply is still being written. Press Space while it talks to cut it off and say something else; **Esc** just stops the voice.

**Presentation.** Press **Start talk** and give your talk; the buddy stays silent. Your speech is transcribed in ~25-second chunks while you talk, and the 35B coach model loads in the background, so feedback starts right after you press **Stop**. You get the measured numbers immediately, then the coach's written feedback streams in. **Read the top tips aloud** speaks the three improvements.

**Conversation practice.** Pick a scenario (job interview, tough conversation, small talk, or describe your own), optionally add details ("junior data analyst at a healthcare startup"), and press **Start**. The buddy plays the other person and opens the conversation; you answer with push-to-talk. Press **End & get feedback** when you're done.

**History.** Charts your pace, filler words per minute, and long pauses across practice sessions, with a table of every session. Click a session to see its transcript, numbers, and feedback, or to delete it.

**Hands-free (beta).** The checkbox in the corner listens for your voice instead of push-to-talk: talk normally and pause for about a second to hand over. While the buddy is talking the microphone is ignored, so it can't hear itself; use Space to interrupt. See the limitations below.

## What the feedback measures

| Measure | How |
|---|---|
| Filler words | Counted in the transcript: um, uh, er, ah, hmm, plus "like", "so", "you know", "I mean", "kind of", "sort of", "basically", "literally". Context rules keep legitimate uses out ("I like pizza", "so much", "what kind of"). |
| Pace | Words ÷ time from your first to your last word (pauses included). The comfortable range defaults to 120–160 wpm. |
| Long pauses | Silences of 2 s or more *inside* your speech, measured from the audio itself, because FLM's Whisper returns no timestamps. |
| Rambling and structure | Sentence lengths, phrases you repeat, and per-answer length in conversations. These go to the coach model as evidence. |
| Clarity, plus 2–3 improvements | Written by the 35B coach model from the transcript and the numbers above. It is told to quote you and not to invent numbers. |

## Configuration

Edit `config.toml` and restart. The most useful settings:

| Setting | Default | What it does |
|---|---|---|
| `models.stt` / `chat` / `feedback` / `tts` | see above | Model for each role; any name from `lemonade list` works. |
| `models.chat_ctx_size` / `feedback_ctx_size` | 8192 / 16384 | Context passed on load. Lemonade's 64k default makes the 4B model use 13 GB of RAM instead of 5 GB. |
| `feedback.think` | `false` | Lets the 35B "think" before writing feedback: deeper, but 30 s or more slower. |
| `voice.tts_voice` | `af_heart` | Kokoro voice (`af_bella`, `am_michael`, `bf_emma`, `bm_george`, …). |
| `persona.name` / `style` | Kira / warm, encouraging and honest | The assistant's name and personality. |
| `practice.long_pause_s` | 2.0 | What counts as a long pause. |
| `practice.target_wpm_low` / `high` | 120 / 160 | Comfortable pace range. |
| `app.data_dir` | `~/.local/share/ai-buddy` | Where sessions are stored. |
| `app.save_audio` | `false` | Also keep WAV recordings of practice sessions. |

## How it works

```
Chrome app window                      Python backend (FastAPI, 127.0.0.1:8765)        Lemonade :13305
 mic → AudioWorklet (16 kHz PCM) ─ws─►  buffer + pause detection ──WAV──►  whisper-v3-turbo-FLM (NPU)
 Space / button = push-to-talk          transcript ──stream──────────────►  qwen3-it-4b-FLM (NPU)
 speaker ◄── PCM chunks ◄────────ws──  clause splitter → TTS per clause ─►  kokoro-v1 (CPU)
                                        end of practice ─────────────────►  qwen3.6-moe-35b-a3b-FLM (NPU)
                                        SQLite + a Markdown file per session
```

- **Low latency:** the LLM reply streams, gets cut at the first clause or sentence, and each piece goes to Kokoro immediately. The browser plays audio chunks as they arrive. Releasing push-to-talk sends everything recorded so far; the worklet keeps 300 ms of pre-roll so a late key press doesn't clip your first word.
- **Keeping models loaded:** the app loads Whisper, Kokoro, and the 4B chat model at startup and warms them up, then checks before each turn that nothing has evicted them. Through Lemonade, FLM can keep Whisper *and* one LLM on the NPU at the same time, as separate processes; this was verified with stable process IDs across alternating turns. It can't hold two LLMs: loading the 35B coach evicts the 4B, even with Lemonade's `max_loaded_models` raised. So the app swaps:
  - Presentation mode loads the 35B while you talk.
  - Conversation practice swaps to it when you press End (about 18 s), then reloads the 4B in the background.
- **Barge-in:** closing the streaming request stops FLM's generation immediately, so interrupting costs nothing.
- **FLM quirk handled:** FLM's `think` flag is sticky per server process; a request that omits it inherits the previous one. The app sends `think` explicitly on every request.

## Measured performance (this laptop)

| Step | Latency |
|---|---|
| Whisper speech-to-text | 2.1–2.4 s per utterance, almost regardless of length. Whisper always encodes a 30 s window, so a 1 s clip costs the same as an 8 s clip. |
| 4B chat model | First token ≈ 0.95 s, ~21 tokens/s. Stays flat as the conversation grows, because FLM reuses its prompt cache. |
| Kokoro first audio | ≈ 0.3 s for a short sentence (≈ 7× faster than real time). |
| **You stop talking → buddy starts talking** | **≈ 3.6–4.0 s** (3.90 s measured in the browser). |
| 35B coach model | ~16 s to load. First token ≈ 2.5–3 s, prefill ≈ 86 tokens/s, ~18 tokens/s. |
| Feedback ready after Stop / End | Presentation: numbers in ~3 s, feedback done in ~30 s for a short talk (longer talks add prefill time). Conversation: ~45–50 s, including the model swap. |

## Data and privacy

Everything stays in `~/.local/share/ai-buddy/`:
- `buddy.db` (SQLite): sessions, turns, metrics, and feedback.
- `sessions/<id>.md`: a readable Markdown copy of each session.
- `audio/`: recordings, only if `save_audio = true`.
- `chrome-profile/`: the app window's browser profile.

The app talks only to Lemonade on localhost. Lemonade itself may check Hugging Face for model updates (its `auto_check_model_updates` setting); that's Lemonade configuration, not part of this app.

## Tests

```bash
uv run pytest -m "not lemonade"                              # unit tests (fast, no models needed)
uv run pytest tests/test_e2e_backend.py -m lemonade -s       # every mode over the WebSocket with real models (~3 min)
uv run pytest tests/test_e2e_browser.py -m lemonade -s       # the real UI in headless Chrome (~4 min); screenshots in test-artifacts/
scripts/phase0/check_env.sh                                  # environment check
python3 scripts/phase0/test_llm.py qwen3-it-4b-FLM            # per-model latency smoke tests (also test_stt.py, test_tts.py, test_coresidency.py)
```

The end-to-end tests speak to the app with Kokoro-synthesized speech. The browser tests replace the microphone with a scripted audio stream; everything after the microphone (resampling worklet, WebSocket, models, playback, UI) is the production path.

## Known limitations

- **The ~2.2 s speech-to-text floor dominates response time.** Total turn latency is about 3.5–4 s. A CPU model (Moonshine-Small via Lemonade) was ~5× faster in testing but dropped words, so it's not used. You can still try it with `models.stt`; it shares Lemonade's single transcription slot with Whisper.
- **Filler counts are a lower bound.** Whisper tends to clean up disfluencies: in testing it kept "um", "like", "so", and "you know" but dropped "uh". That test used a synthetic voice, so check with your own. Telling filler "like" and "so" from real uses is heuristic.
- **Punctuation comes from the recognizer**, not from you. The coach is told not to judge sentence structure from it, but sentence-length stats depend on it (they are skipped when the transcript has too little punctuation).
- **Pause detection uses audio energy.** A noisy room, or very soft speech, can blur pauses and pace.
- **Only one LLM fits on the NPU at a time**, so the coach model is swapped in for feedback (~18 s in conversation practice). Switching back to Chat right after feedback can show "Loading models…" for ~6 s while the 4B reloads.
- **Long talks mean longer feedback.** The 35B prefills ~86 tokens/s, so a 10-minute talk adds roughly 30 s before the feedback starts.
- **Hands-free mode is beta.** Detection was verified only with synthetic speech, not in a real room. It ignores the microphone while the buddy talks (no voice barge-in) and may misfire on background noise or cut you off during a long pause. Push-to-talk is the reliable path.
- **Push-to-talk needs the app window focused:** Wayland doesn't allow global hotkeys.
- **The microphone stays open while the window is open** (GNOME shows the mic indicator), but audio is only sent while you hold talk, during a presentation recording, or in hands-free mode.
- **A rare NPU error from FLM's Whisper** (`qds_device::wait() unexpected command state`) occurred once during testing and could not be reproduced. The app retries once and reloads Whisper if the retry also fails.
- English-only prompts and filler list; single user.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Lemonade server not reachable" | `sudo systemctl restart lemond`, then `lemonade status`. |
| A model pill shows *error* | Run `lemonade list --downloaded` to check the model exists; `lemonade pull <name>` if not. Restarting `lemond` clears NPU state. |
| No sound | Unmute the speaker (`wpctl set-mute @DEFAULT_AUDIO_SINK@ 0`), then raise the volume. |
| Mic blocked | In the app window, click the site-info icon → Microphone → Allow, then reload. |
| Wrong microphone | Choose the input in GNOME Settings → Sound; the app uses the default input. |
| Port 8765 busy | `uv run ai-buddy --port 8766`, or change `app.port`. |

## Project layout

```
ai_buddy/
  __main__.py     launcher (server + Chrome app window)
  server.py       FastAPI app, REST API, WebSocket endpoint
  session.py      per-connection state machine for the three modes
  models.py       keeps models loaded, swaps the coach LLM, retries STT
  lemonade.py     async client for Lemonade's OpenAI-compatible API
  audio.py        WAV, speech/pause detection, presentation chunking
  text.py         streaming clause splitter, TTS text cleanup
  metrics.py      filler words, pace, pauses, rambling indicators
  prompts.py      persona, scenarios, coach prompt
  storage.py      SQLite + Markdown export
  static/         UI: index.html, app.js, capture-worklet.js, charts.js, markdown.js, style.css
scripts/phase0/   environment check and per-model latency smoke tests
tests/            unit tests, backend and browser end-to-end tests
config.toml       settings
```
