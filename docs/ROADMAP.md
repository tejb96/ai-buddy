# Roadmap

AI Buddy is a local voice AI agent with plug-in skills: a voice-first coach and creative partner that runs on the
laptop's NPU. The assistant is named after the voice you pick (Bella by default). What's built is described in the
[README](../README.md); this page is where it's going. Everything new arrives as a skill you can switch on or off.

Each item is marked **built** or **planned**; planned items aren't in any particular order yet.

## 1. Practice

| Skill | Status |
|---|---|
| Speaking coach: presentations, with feedback on filler words, pace, pauses, structure and your talking points | built |
| Conversation practice: job interview, tough conversation, small talk, or a scenario you describe | built |
| Mock interviews from a real job posting: paste or fetch the posting; questions follow its requirements and your resume | planned |
| Language conversation practice: role-play in the language you're learning, with gentle corrections afterwards | planned |

## 2. Learn

| Skill | Status |
|---|---|
| Teaching mode for a topic, a document or a repository | planned |
| Quizzes and exercises drawn from what you're learning | planned |
| Explain-it-back: you explain a concept aloud and the assistant finds the gaps | planned |
| Spaced repetition (for example FSRS) for the concepts you got wrong | planned |

## 3. Create

**Music producer, driving real Ubuntu tools** (planned):
- Audacity through its scripting pipe, and Ardour through Lua;
- SoX/FFmpeg and Rubber Band for edits, tempo and pitch;
- Demucs to split stems and Basic Pitch to turn audio into MIDI;
- FluidSynth to render MIDI;
- ACE-Step through Lemonade for generating music.

**Writing help** (planned): lyrics, rhymes and song structure.

**GPU on demand** (planned): heavy music models may use the iGPU, loaded only in music mode and unloaded when you
leave it. Everything else stays on the NPU (models) and the CPU (voice, note embeddings).

## 4. Ecosystem

| Item | Status |
|---|---|
| Any MCP server becomes a skill: switch, status and folder in Settings; tools that change something ask first | built |
| The official filesystem, fetch and time servers, ready to switch on | built |
| GitHub through its MCP server | documented, untested until there's a token |
| Google and Microsoft accounts (mail, calendar, files) through MCP servers | planned |

## Out of scope

Home automation and general control of the computer.
