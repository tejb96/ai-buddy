# AI Buddy

A local voice AI agent with plug-in skills: a voice-first coach and creative partner that runs on your laptop's
NPU. You talk; it answers out loud. It looks things up, searches your own notes, uses tools from MCP servers
(asking before it changes anything), and coaches you through presentations and practice conversations with
honest, specific feedback.

The assistant is named after the voice you pick: **Bella** by default (Emma, George, Michael… in Settings).

Everything runs on the laptop through [Lemonade](https://lemonade-server.ai/): the language model and speech
recognition on the Ryzen AI NPU (FastFlowLM), the voice and note search on the CPU. What leaves the laptop:
web searches and the pages read for them, and whatever the MCP servers you switch on do.

Where it's going: [docs/ROADMAP.md](docs/ROADMAP.md). Measurements, decisions and open issues:
[docs/phase0-findings.md](docs/phase0-findings.md).

## What it does

- **Talk to it.** Hold Space (or the round button), talk, let go. It starts speaking while it's still writing
  the reply. Press Space to cut it off; Esc only stops the voice. Details and sources appear on screen, not
  read aloud.
- **Skills, each switchable** (Settings → Skills):
  - **Web search**: searches DuckDuckGo (or your SearXNG) and reads the top pages. For live data (weather,
    scores, prices) it goes by the newest source and shows when each page was updated.
  - **Notes**: searches your knowledge folder (Markdown, text, PDF, Word, HTML) by meaning and keywords.
    Passages have to actually match: practice sessions only use notes that are close in meaning to the talk or
    interview, so an unrelated resume never turns into "talking points".
  - **Clock** and **Calculator**: times anywhere, days until a date, exact arithmetic.
  - **Speaking coach**: the Presentation and Conversation practice tabs (they disappear when it's off).
- **MCP servers as skills** (Settings → MCP servers): the official **files** (a folder you choose), **fetch**
  (read a web page) and **time** servers are ready to switch on, and any other server can be added in
  `config.toml`. Settings shows each one's status. Tools that change something ask first, out loud and with
  a Yes/No card; you can answer by voice.
- **No guessing.** It uses a tool or says it doesn't know, and asks instead of assuming (which city, which
  day). If you question a fact it got from a tool ("isn't it 16 degrees?"), it checks again before answering.
  If the sources still disagree, it says so and gives both, instead of agreeing to please you.
- **Practice and feedback.** *Presentation*: give a talk, get numbers (pace, filler words, long pauses) and
  written feedback that quotes you and checks your talking points. *Conversation practice*: it plays an
  interviewer, a difficult colleague or anyone you describe, then coaches you.
- **History**: charts of your pace, filler words and pauses across practice sessions; every session's
  transcript and feedback.
- **Exit** (power button, top right): stops the app, unloads every model from Lemonade and closes the window.

## Models

| Role | Default | Runs on | Options in Settings |
|---|---|---|---|
| Assistant (chat, tools, coaching) | `qwen3.5-9b-FLM`, thinking off | NPU | `gemma4-it-12b-FLM`, `qwen3.6-moe-35b-a3b-FLM` |
| Speech-to-text | `whisper-v3-turbo-FLM` | NPU | `Whisper-Large-v3-Turbo` (whisper.cpp, CPU) |
| Voice | `kokoro-v1` | CPU | 9 voices |
| Note search | `Qwen3-Embedding-0.6B-GGUF` (llama.cpp) | CPU | keywords only (`models.embed = ""`) |

Whisper transcribes what you say and the assistant reads that text. Earlier builds let Gemma 4 hear the
recording itself, but it misheard short phrases that Whisper got right.

## Setup

Tested on a ThinkPad P14s Gen 6 AMD (Ryzen AI 7 PRO 350, XDNA 2 NPU, 64 GB RAM), Ubuntu 26.04, Lemonade 2026.41.1
with FLM 1.0.6. No sudo was needed for the app.

1. **Lemonade running:** `systemctl status lemond` and `lemonade status`.
2. **Backends and models** (skip what you have):
   ```bash
   lemonade backends install kokoro:cpu
   lemonade backends install llamacpp:cpu          # note search embeddings
   lemonade pull whisper-v3-turbo-FLM
   lemonade pull qwen3.5-9b-FLM
   lemonade pull kokoro-v1
   lemonade pull Qwen3-Embedding-0.6B-GGUF
   # optional: the other models in Settings
   lemonade pull gemma4-it-12b-FLM
   lemonade pull qwen3.6-moe-35b-a3b-FLM
   lemonade backends install whispercpp:cpu && lemonade pull Whisper-Large-v3-Turbo
   ```
3. **Python** (with [uv](https://docs.astral.sh/uv/)): `uv sync`.
4. **For the MCP servers:** Node.js (for `npx`; nvm is fine, the app finds it) and `uvx` (comes with uv).
5. `scripts/phase0/check_env.sh` checks the server, the NPU and the models.

## Run

```bash
uv run ai-buddy
```

It opens in its own Chrome app window on `http://127.0.0.1:8765`; the first start loads and warms the models.
Allow the microphone when asked. **Exit** (or closing the window) stops the app; only Exit also unloads the
models. Options: `--no-window`, `--port 9000`, `--config path/to/config.toml`.
`scripts/install-desktop-entry.sh` adds AI Buddy to GNOME's app search.

## Settings (gear icon)

- **Voice**: pick a voice (and so the assistant's name), play a sample, stop it.
- **Language model** and **Speech-to-text**: the choices in the table above.
- **Skills**: switch built-in skills on and off.
- **MCP servers**: switch servers on and off, see whether they're connected (and why not), choose the files
  server's folder, and add servers (below).
- **Knowledge folder**: where your notes live (default `~/Documents/ai-buddy-notes`), re-indexed automatically.

Choices made here are saved in `~/.local/share/ai-buddy/settings.json` and override `config.toml`; only the ones
that differ from `config.toml` are saved.

## MCP servers

Every `[mcp.servers.<name>]` entry in `config.toml` is a skill named `<name>`; its tools reach the model as
`<name>_<tool>`. A tool asks before it runs unless the server marks it read-only, or it's listed in
`auto_approve`. The ones that come with the app:

| Server | Package | Tools the model sees |
|---|---|---|
| `files` | `@modelcontextprotocol/server-filesystem` (npx) | read, list, search and file info freely; write, edit, move and create folders ask first. Only inside the folder you choose. |
| `fetch` | `mcp-server-fetch` (uvx) | `fetch`: read a web page as text |
| `time` | `mcp-server-time` (uvx) | current time anywhere, convert between zones |

**Adding a server.** Add an entry to `config.toml`, then press **Reload servers** in Settings (no restart):

```toml
[mcp.servers.weather]
command = "npx"                                  # or url = "http://127.0.0.1:8000/mcp" for a running server
args = ["-y", "some-weather-mcp-server@1.0.0"]   # pin versions
description = "Forecasts from the weather service"
tools = ["get_forecast"]                         # optional: only these tools (each costs prompt space)
```

In `command`, `args`, `env` and `headers`: `{folder}` is a folder you pick in Settings, `{timezone}` your time
zone, and `${NAME}` a secret from `~/.local/share/ai-buddy/secrets.env` (`NAME=value` lines; `chmod 600` it).
That keeps tokens out of `config.toml`, which is in git. A local server's own messages go to
`~/.local/share/ai-buddy/logs/mcp-<name>.log`.

### GitHub

Not tested yet: it needs a token. Steps:

1. Create a fine-grained personal access token at <https://github.com/settings/personal-access-tokens/new>, for
   the repositories you want, with read access to Contents, Issues, Pull requests and Metadata. Add write
   access only if you'll drop read-only mode below.
2. Put it in `~/.local/share/ai-buddy/secrets.env`: `GITHUB_TOKEN=github_pat_…`, then `chmod 600` that file.
3. In `config.toml`, uncomment one of the two `[mcp.servers.github]` blocks:
   - **Local binary** (recommended): download `github-mcp-server_Linux_x86_64.tar.gz` from
     <https://github.com/github/github-mcp-server/releases>, and put `github-mcp-server` in `~/.local/bin`. The
     block runs `github-mcp-server stdio` with `GITHUB_TOOLSETS = "repos,issues,pull_requests"` (fewer tools
     help a 9B model choose) and `GITHUB_READ_ONLY = "1"`.
   - **GitHub's hosted server**: `url = "https://api.githubcopilot.com/mcp/readonly"` with
     `headers = { Authorization = "Bearer ${GITHUB_TOKEN}" }`. Nothing to install.
4. Press **Reload servers** in Settings, switch **Github** on, and check that it says Connected. Then ask, for
   example, "what are the open issues in my ai-buddy repo?".

## How it answers

```
Chrome app window                      Python (FastAPI, 127.0.0.1:8765)                 Lemonade :13305
 mic → AudioWorklet (16 kHz) ──ws──►  recording ─────WAV──────────────────►  whisper-v3-turbo-FLM (NPU)
 hold Space = push-to-talk             transcript → router → tool calls ─►  qwen3.5-9b-FLM (NPU)
                                        skills: web, notes, clock, calculator, MCP servers
 speaker ◄── PCM ◄──────────────ws──  reply, clause by clause → TTS ─────►  kokoro-v1 (CPU)
                                        notes index (SQLite FTS5 + vectors) ►  Qwen3-Embedding-0.6B (CPU)
```

- **One NPU job at a time.** When Whisper and the LLM (two FLM processes) run at the same moment, both
  results come back corrupted, so the app serializes them.
- **The router** makes unambiguous tool calls in code before the model runs. These are the time somewhere,
  "weather in Calgary", "18% of 240", "my notes / resume / talking points", "search the web for…", and
  re-checking a fact you questioned. FLM ignores `tool_choice`, so this is how such calls are guaranteed.
- **Spoken style**: one to three sentences, no lists or links read aloud.
- **Prompt cache**: tool results go back as a user message, so earlier turns render identically and FLM only
  reads what's new.

## Configuration

`config.toml` holds everything; restart after editing (MCP servers: Reload servers). The most useful settings:

| Setting | Default | What it does |
|---|---|---|
| `models.llm`, `models.llm_choices` | `qwen3.5-9b-FLM` | The assistant's model, and those offered in Settings. |
| `models.stt`, `models.stt_choices`, `models.stt_backend` | FLM Whisper; `cpu` | Speech-to-text, and the whisper.cpp backend for non-FLM models. |
| `models.embed` | `Qwen3-Embedding-0.6B-GGUF` | Note search embeddings on llama.cpp's CPU backend; `""` = keywords only. |
| `voice.tts_voice` | `af_bella` | The voice, and so the assistant's name. |
| `user.location` | `""` | Your city, e.g. "Calgary, Alberta": assumed for "weather", "downtown", "nearby" when you don't name a place. Worth setting. |
| `skills.enabled`, `skills.dirs` | built-ins | Skills on at start (Settings overrides), folders with your own skills. |
| `mcp.servers.<name>` | files, fetch, time | MCP servers (see above). |
| `tools.search_backend` | `duckduckgo` | Or `searxng` with `tools.searxng_url`. |
| `rag.folder` | `~/Documents/ai-buddy-notes` | The knowledge folder. |
| `practice.long_pause_s`, `target_wpm_low/high` | 2.0, 120–160 | What counts as a long pause; comfortable pace. |
| `app.save_audio` | `false` | Also keep WAV recordings of practice sessions. |

Your own skills: a folder with `SKILL.md` (the [Agent Skills](https://agentskills.io/specification) format), plus
optional `tools.py` (subclasses of `ai_buddy.tools.Tool`) and `modes.py`. Built-ins in `ai_buddy/skills/builtin/`.

## What the feedback measures

| Measure | How |
|---|---|
| Filler words | Counted in Whisper's verbatim transcript (um, uh, like, so, you know, I mean…), with rules that keep real uses out ("I like pizza"). |
| Pace | Words per minute from your first to your last word; 120–160 is the default comfortable range. |
| Long pauses | Silences of 2 s or more inside your speech, measured from the audio. |
| Structure | Sentence lengths and repeated phrases, given to the coach as evidence. |
| Clarity, what worked, top 3 to improve, talking points | Written by the assistant from the transcript and the numbers; it quotes you and doesn't invent numbers. |

## Data and privacy

Everything stays in `~/.local/share/ai-buddy/`: `buddy.db` (sessions, turns, metrics, feedback),
`sessions/<id>.md` (a readable copy of each), `knowledge.db` (the notes index), `settings.json`, `secrets.env`
(if you make one), `logs/`, `audio/` (only with `save_audio`) and `chrome-profile/`.

Leaving the laptop: search queries (DuckDuckGo by default), the pages read for them, and whatever the MCP servers
you switch on do (fetch reads web pages; GitHub talks to GitHub). Lemonade may check Hugging Face for model updates
(its own setting).

## Tests

```bash
uv run pytest -m "not lemonade"                           # unit tests, no models needed (a few seconds)
uv run pytest tests/test_e2e_backend.py -m lemonade -s    # every mode over the WebSocket, real models
uv run pytest tests/test_e2e_browser.py -m lemonade -s    # the real UI in headless Chrome; screenshots in test-artifacts/
```

The end-to-end tests speak to the app with Kokoro-synthesized speech, and the browser tests feed it through a
scripted microphone, so everything after the microphone is the production path. Benchmark scripts are in
[`scripts/bench/`](scripts/bench/README.md).

## Known limitations

- **Speed is traded for accuracy.** Whisper on the NPU takes ~2 s per turn whatever its length, and the 9B
  model writes ~10 tokens/s, so a reply starts a few seconds after you stop talking; tool turns take longer.
  Practice feedback for a long talk takes a minute or two.
- **The model can still slip.** It's told not to guess and the router covers the clear cases, but a 9B model
  sometimes misreads results or runs long. Sources are on screen to check. A known case: "my sister is visiting
  from Vancouver, what can we do downtown?" gets downtown Vancouver; set `user.location` so "downtown" has a meaning.
- **Web pages vary.** Some sites render in JavaScript or block readers, so a search may find only snippets; it
  then says it couldn't find the answer.
- **Filler counts are a lower bound**: Whisper tidies some disfluencies away.
- **Hands-free mode is beta**: energy-based voice detection, tested with synthetic speech only. Push-to-talk is
  the reliable path.
- **Push-to-talk needs the window focused** (Wayland has no global hotkeys).
- English only; one user.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Lemonade server not reachable" | `sudo systemctl restart lemond`, then `lemonade status`. |
| A model pill shows *error* | `lemonade list --downloaded`; `lemonade pull <name>` if it's missing. |
| An MCP server says Error | The message in Settings says why (missing folder, program not installed, unset `${NAME}`); the log file has the server's own output. |
| No sound | Unmute: `wpctl set-mute @DEFAULT_AUDIO_SINK@ 0`. |
| Mic blocked | In the app window: site-info icon → Microphone → Allow, then reload. |
| Port 8765 busy | `uv run ai-buddy --port 8766`. |

## Project layout

```
ai_buddy/
  __main__.py     launcher (server + Chrome app window; closes it on Exit)
  server.py       FastAPI app, REST API (settings, skills, MCP, exit), WebSocket endpoint
  session.py      one conversation: turns, the agent loop, asking permission, practice modes
  router.py       tool calls made in code: clear intents, and re-checking a questioned fact
  models.py       keeps models loaded, one NPU job at a time, switching models, unloading on Exit
  lemonade.py     client for Lemonade's OpenAI-compatible API
  prompts.py      the assistant's persona (named after its voice) and standing rules
  tools/          the Tool base class and registry
  skills/         skill loading (SKILL.md), MCP servers as skills, built-in skills
  web.py          search, page reading, how current a page is
  knowledge.py    the notes index: chunking, FTS5 keywords + embeddings
  audio.py, text.py, metrics.py, storage.py, settings.py
  static/         the UI
scripts/          bench/ (benchmarks), phase0/ (environment checks), FLM update and smoke test, desktop entry
tests/            unit tests, backend and browser end-to-end tests
config.toml       settings
```
