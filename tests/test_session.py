"""The agent loop, with a scripted model: tool rounds, routing, audio turns, transcripts, asking permission."""

from __future__ import annotations

import asyncio

import pytest

from ai_buddy.prompts import NUDGE
from ai_buddy.session import AUDIO_WORDS_PER_S, split_heard, yes_or_no
from ai_buddy.tools import RESULTS_PREFIX
from fakes import FakeSearch, SaveNote, call, events, make_session, say, speech


def last_user(request: dict) -> dict:
    return [m for m in request["messages"] if m["role"] == "user"][-1]


# ---------- text turns (Whisper first) ----------

async def test_plain_answer(tmp_path):
    s, lm, models = make_session(tmp_path, [say("Why did the cat sit on the computer? To watch the mouse.")],
                                 ["Tell me a joke about cats."])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    end = events(s, "reply_end")[-1]
    assert end["text"] == "Why did the cat sit on the computer? To watch the mouse." and not end["interrupted"]
    assert [m["role"] for m in s.history] == ["system", "user", "assistant"]
    assert "".join(lm.spoken).replace(" ", "") == end["text"].replace(" ", "")
    assert lm.requests[0]["tools"] == ["calculate", "get_datetime"]


async def test_a_clear_time_question_is_answered_from_the_clock(tmp_path):
    """The router makes the call before the model runs, so the model answers from the result in one round."""
    s, lm, _ = make_session(tmp_path, [say("It's just after two in the morning in Tokyo.")],
                            ["Nice. And what time is it in Tokyo right now?"])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert len(lm.requests) == 1
    sent = lm.requests[0]["messages"]
    assert sent[-2]["tool_calls"][0]["function"]["name"] == "get_datetime"
    assert sent[-1]["content"].startswith(RESULTS_PREFIX) and "Asia/Tokyo" in sent[-1]["content"]
    start = events(s, "tool_start")[0]
    assert start["name"] == "get_datetime" and start["routed"]
    assert events(s, "reply_end")[-1]["text"] == "It's just after two in the morning in Tokyo."


async def test_model_tool_round(tmp_path):
    s, lm, _ = make_session(tmp_path, [[call("calculate", expression="137.5 * 1.15 / 4")],
                                       say("Each person pays about thirty-nine fifty.")],
                            ["If I split a hundred and thirty seven fifty bill four ways with a fifteen percent tip, "
                             "how much does each person pay?"])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert len(lm.requests) == 2
    assert '"result": 39.53125' in lm.requests[1]["messages"][-1]["content"]
    assert lm.spoken[0] == "Let me work that out."                      # the filler line while it runs
    tool_end = events(s, "tool_end")[0]
    assert tool_end["ok"] and tool_end["summary"] == "137.5 * 1.15 / 4 = 39.53125"
    assert s.history[-1] == {"role": "assistant", "content": "Each person pays about thirty-nine fifty."}


async def test_announcing_a_lookup_without_calling_gets_a_nudge(tmp_path):
    s, lm, _ = make_session(tmp_path, [say("Let me check that for you."), [call("get_datetime")], say("It's 3 PM.")],
                            ["Is it afternoon yet?"])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert lm.requests[1]["messages"][-1] == {"role": "user", "content": NUDGE}
    assert len(lm.requests) == 3 and s.history[-1]["content"] == "It's 3 PM."


async def test_tool_calls_are_capped(tmp_path):
    s, lm, _ = make_session(tmp_path, [[call("get_datetime")]] * 8, ["Keep checking the time."])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert len(events(s, "tool_start")) == 3                             # tools.max_calls_per_turn
    assert "couldn't pin that down" in events(s, "reply_end")[-1]["text"]


# ---------- audio turns (the model hears the recording) ----------

async def test_audio_turn_keeps_the_recording_and_shows_whispers_words(tmp_path):
    s, lm, models = make_session(tmp_path, [say("Ha, good one."), say("Sure.")], ["Tell me a joke.", "Another?"],
                                 hears_audio=True)
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert models.overlaps == 0            # Whisper ran first, never while the model was writing
    sent = last_user(lm.requests[0])
    assert sent["content"][0]["type"] == "input_audio" and set(sent) == {"role", "content"}   # no private keys
    assert events(s, "user_text")[-1]["text"] == "Tell me a joke." and models.transcribed == 1
    assert s.history[1]["_transcript"] == "Tell me a joke."
    assert "You hear the user's voice directly." in s.history[0]["content"]
    await s._user_turn(speech(), None)
    # The earlier recording is still audio in the next request: the prompt only grows (FLM's cache).
    assert lm.requests[1]["messages"][1]["content"][0]["type"] == "input_audio"
    turns = s.store.get_session(s.session_id)["turns"]
    assert [t["text"] for t in turns] == ["Tell me a joke.", "Ha, good one.", "Another?", "Sure."]


async def test_audio_turn_routes_before_the_model_runs(tmp_path):
    """Whisper's words come first, so the clock is read before the model answers (one LLM round)."""
    s, lm, models = make_session(tmp_path, [say("It's one thirty at night in Tokyo.")],
                                 ["And what time is it in Tokyo right now?"], hears_audio=True)
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert len(lm.requests) == 1 and models.overlaps == 0
    sent = lm.requests[0]["messages"]
    assert sent[-3]["content"][0]["type"] == "input_audio"
    assert sent[-2]["tool_calls"][0]["function"]["name"] == "get_datetime" and "Asia/Tokyo" in sent[-1]["content"]


async def test_heard_mode_stops_an_answer_that_skips_a_needed_tool(tmp_path):
    s, lm, models = make_session(tmp_path, [say("<heard>And what time is it in Tokyo right now?</heard> "
                                                "It's seven thirty in the morning there."),
                                            say("It's one thirty at night in Tokyo.")],
                                 [], hears_audio=True, transcript="heard")
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert models.transcribed == 0
    shown = "".join(e["text"] for e in events(s, "reply_delta"))
    assert "seven thirty" not in shown and not any("seven" in t for t in lm.spoken)
    assert events(s, "tool_start")[0]["name"] == "get_datetime" and events(s, "tool_start")[0]["routed"]
    second = lm.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"] == "<heard>And what time is it in Tokyo right now?</heard>"   # the answer is gone
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_datetime"
    assert events(s, "reply_end")[-1]["text"] == "It's one thirty at night in Tokyo."


async def test_heard_mode_adds_a_missing_routed_call_to_the_models_own(tmp_path):
    search = FakeSearch()
    s, lm, _ = make_session(tmp_path, [say("<heard>What's the weather in Paris, and what time is it there?</heard>")
                                       + [call("web_search", query="weather in Paris, France")],
                                       say("Sunny and fourteen, and it's ten at night there.")],
                            [], hears_audio=True, transcript="heard", extra_tools=[search])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    names = [e["name"] for e in events(s, "tool_start")]
    assert sorted(names) == ["get_datetime", "web_search"] and search.queries == ["weather in Paris, France"]
    results = lm.requests[1]["messages"][-1]["content"]
    assert "Europe/Paris" in results and "Sunny, high 14." in results
    assert events(s, "reply_end")[-1]["sources"] == [{"title": "Forecast", "url": "https://example.com"}]


async def test_heard_mode_takes_the_transcript_from_the_reply(tmp_path):
    s, lm, models = make_session(tmp_path, [say("<heard>Um, tell me a joke.</heard> Why did the cat nap? It was tired.")],
                                 [], hears_audio=True, transcript="heard")
    await s._start("chat", None, "")
    assert s.live is None
    await s._user_turn(speech(), None)
    user = events(s, "user_text")[-1]
    assert user["text"] == "Um, tell me a joke." and user["source"] == "heard" and models.transcribed == 0
    end = events(s, "reply_end")[-1]
    assert end["text"] == "Why did the cat nap? It was tired." and "heard" not in "".join(lm.spoken)
    assert s.history[-1]["content"].startswith("<heard>Um, tell me a joke.</heard>")   # as the model wrote it
    assert "<heard></heard>" in s.history[0]["content"]


async def test_heard_mode_falls_back_to_whisper_after_the_stream(tmp_path):
    s, lm, models = make_session(tmp_path, [say("Sure, here's one about a cat."), [call("get_datetime")],
                                            say("It's 3 PM.")],
                                 ["Tell me a joke.", "What's the time?"], hears_audio=True, transcript="heard")
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)                    # no <heard> block before the answer
    assert events(s, "reply_end")[-1]["text"] == "Sure, here's one about a cat."
    await asyncio.sleep(0.05)                             # Whisper runs once the reply is done
    assert events(s, "user_text")[-1]["text"] == "Tell me a joke." and models.transcribed == 1
    await s._user_turn(speech(), None)                    # a tool call with no words at all
    assert events(s, "user_text")[-1]["text"] == "What's the time?" and models.transcribed == 2
    assert models.overlaps == 0


async def test_heard_mode_drops_a_repeated_transcript_after_tools(tmp_path):
    s, lm, _ = make_session(tmp_path, [say("<heard>Is it afternoon?</heard>") + [call("get_datetime")],
                                       say("<heard>Is it afternoon?</heard> Yes, it's 3 PM.")],
                            [], hears_audio=True, transcript="heard")
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert events(s, "reply_end")[-1]["text"] == "Yes, it's 3 PM."
    assert not any("heard" in t or "afternoon" in t for t in lm.spoken)


def test_split_heard():
    assert split_heard("<hea") == ("wait", None, "")
    assert split_heard("  <heard>Hi there") == ("wait", None, "")
    assert split_heard("<heard>Hi  there.</heard> Hello!") == ("done", "Hi there.", "Hello!")
    assert split_heard("Hello!") == ("none", None, "Hello!")


# ---------- asking permission ----------

async def answer_when_asked(s, allow: bool | None):
    for _ in range(200):
        await asyncio.sleep(0.01)
        if s.pending_confirm:
            if allow is not None:
                s._resolve_confirm(s.pending_confirm[0], allow)
            return


@pytest.mark.parametrize("allow", [True, False, None])
async def test_actions_ask_first(tmp_path, allow):
    note = SaveNote()
    s, lm, _ = make_session(tmp_path, [[call("save_note", text="buy milk")], say("Done." if allow else "Okay, I won't.")],
                            ["Save a note: buy milk."], extra_tools=[note])
    s.cfg.skills.confirm_timeout_s = 0.3
    await asyncio.gather(s._user_turn(speech(), None) if s.mode else _start_then_turn(s), answer_when_asked(s, allow))
    req = events(s, "confirm_request")[0]
    assert req["tool"] == "save_note" and req["text"] == "Should I go ahead with saving a note (buy milk)?"
    assert req["text"] in lm.spoken                      # Kira asks out loud
    assert events(s, "confirm_end")[0]["allowed"] is bool(allow)
    assert note.saved == (["buy milk"] if allow else [])
    if not allow:
        assert "said no" in lm.requests[1]["messages"][-1]["content"]


async def _start_then_turn(s):
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)


async def test_a_spoken_yes_answers_the_question(tmp_path):
    note = SaveNote()
    s, lm, models = make_session(tmp_path, [[call("save_note", text="call mum")], say("Saved.")],
                                 ["Save a note to call mum.", "Yes please."], extra_tools=[note])
    await s._start("chat", None, "")
    turn = asyncio.create_task(s._user_turn(speech(), None))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if s.pending_confirm:
            break
    s._rec_start()                    # talking now answers the question instead of cutting Kira off
    assert not turn.done()
    s.rec = bytearray(speech())
    s._rec_stop()
    await turn
    assert note.saved == ["call mum"] and events(s, "reply_end")[-1]["text"].endswith("Saved.")


@pytest.mark.parametrize("text,answer", [("Yes please.", True), ("Yeah, go ahead.", True), ("Sure.", True),
                                         ("Okay do it", True), ("No, don't.", False), ("Wait, not yet.", False),
                                         ("Yes. Actually no.", False), ("What was the question?", None)])
def test_yes_or_no(text, answer):
    assert yes_or_no(text) is answer


# ---------- history, prompts and modes ----------

async def test_trimming_turns_old_recordings_into_their_transcripts(tmp_path):
    s, lm, _ = make_session(tmp_path, [say("Okay.")] * 4, ["one", "two", "three", "four"], hears_audio=True)
    s.cfg.chat.history_word_budget = int(1.6 * AUDIO_WORDS_PER_S * 2.2)   # a bit over two recordings
    await s._start("chat", None, "")
    for _ in range(3):
        await s._user_turn(speech(), None)
    kinds = [type(m["content"]).__name__ for m in s.history if m["role"] == "user"]
    assert kinds[-1] == "list"                         # the turn being answered stays audio
    assert "str" in kinds and all(not isinstance(m["content"], list) for m in s.history[1:-2] if m["role"] == "user")


async def test_switching_skills_and_models_updates_the_prompt(tmp_path):
    s, lm, models = make_session(tmp_path, [say("Hi.")] * 2, ["hello", "again"], hears_audio=True)
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert "calculate" in s.history[0]["content"]
    s.svc.skills.set_enabled("calculator", False)
    models.audio = False                               # e.g. switched to the 35B, which reads text
    await s._user_turn(speech(), None)
    assert "calculate" not in s.history[0]["content"]
    assert "speech recognition" in s.history[0]["content"]
    assert s.history[1]["content"] == "hello"          # the earlier recording became its transcript
    assert lm.requests[1]["tools"] == ["get_datetime"]


async def test_practice_needs_the_speaking_coach(tmp_path):
    s, _, _ = make_session(tmp_path, skills=("clock",))
    await s._start("presentation", None, "")
    assert "speaking coach skill" in events(s, "error")[-1]["text"] and s.mode is None


async def test_interview_uses_the_coach_skill(tmp_path):
    s, lm, _ = make_session(tmp_path, [say("Hi, I'm Alex. Tell me about yourself.")])
    await s._start("interview", "job_interview", "data analyst")
    await s.turn_task
    system = lm.requests[0]["messages"][0]["content"]
    assert "interviewing the user for: data analyst" in system and lm.requests[0]["tools"] == []
