"""The agent loop, with a scripted model: tool rounds, routing, transcripts, asking permission."""

from __future__ import annotations

import asyncio

import pytest

from ai_buddy.prompts import NUDGE
from ai_buddy.session import yes_or_no
from ai_buddy.tools import RECHECK_SUFFIX, RESULTS_PREFIX
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


# ---------- Whisper's words are what the model reads ----------

async def test_the_model_reads_whispers_words_and_whisper_never_overlaps_it(tmp_path):
    s, lm, models = make_session(tmp_path, [say("Ha, good one."), say("Sure.")], ["Tell me a joke.", "Another?"])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    await s._user_turn(speech(), None)
    assert models.overlaps == 0            # Whisper ran first, never while the model was writing
    assert [m["content"] for m in lm.requests[1]["messages"][1:]] == ["Tell me a joke.", "Ha, good one.", "Another?"]
    assert all(set(m) <= {"role", "content", "tool_calls"} for m in lm.requests[1]["messages"])
    assert "speech recognition" in s.history[0]["content"]
    turns = s.store.get_session(s.session_id)["turns"]
    assert [t["text"] for t in turns] == ["Tell me a joke.", "Ha, good one.", "Another?", "Sure."]


# ---------- questioning a fact ----------

async def test_questioning_a_fact_from_the_web_checks_again(tmp_path):
    """"Isn't it 16 degrees?" after a weather answer: the app searches again (reading one more source) before the
    model answers, and asks it to compare honestly instead of just agreeing."""
    search = FakeSearch()
    s, lm, _ = make_session(tmp_path, [[call("web_search", query="Calgary weather now")],
                                       say("It's nine degrees and cloudy in Calgary."),
                                       say("You're right: a newer reading says sixteen, feeling like twelve.")],
                            ["What's it like outside in Calgary?", "Isn't it 16 degrees and feels like 12?"],
                            extra_tools=[search])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    await s._user_turn(speech(), None)
    assert search.queries == ["Calgary weather now", "Calgary weather now"] and len(lm.requests) == 3
    sent = lm.requests[2]["messages"]
    assert sent[-3]["content"] == "Isn't it 16 degrees and feels like 12?"
    again = sent[-2]["tool_calls"][0]["function"]
    assert again["name"] == "web_search" and '"double_check": true' in again["arguments"]
    assert sent[-1]["content"].startswith(RESULTS_PREFIX) and RECHECK_SUFFIX in sent[-1]["content"]
    assert events(s, "tool_start")[-1]["routed"]


async def test_chat_that_merely_sounds_like_a_question_does_not_recheck(tmp_path):
    search = FakeSearch()
    s, lm, _ = make_session(tmp_path, [[call("web_search", query="Calgary weather now")], say("Sunny, fourteen."),
                                       say("It really is.")],
                            ["What's it like outside in Calgary?", "Isn't it lovely?"], extra_tools=[search])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    await s._user_turn(speech(), None)
    assert search.queries == ["Calgary weather now"] and len(lm.requests) == 3


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

async def test_trimming_drops_whole_exchanges(tmp_path):
    s, lm, _ = make_session(tmp_path, [say("Okay, noted.")] * 4, ["one two three", "four five six", "seven eight nine",
                                                                  "ten eleven twelve"])
    s.cfg.chat.history_word_budget = 12
    await s._start("chat", None, "")
    for _ in range(4):
        await s._user_turn(speech(), None)
    said = [m["content"] for m in s.history[1:]]
    assert said[-2:] == ["ten eleven twelve", "Okay, noted."] and "one two three" not in said
    assert [m["role"] for m in s.history[1:]] == ["user", "assistant"] * (len(said) // 2)


async def test_switching_skills_updates_the_prompt(tmp_path):
    s, lm, models = make_session(tmp_path, [say("Hi.")] * 2, ["hello", "again"])
    await s._start("chat", None, "")
    await s._user_turn(speech(), None)
    assert "calculate" in s.history[0]["content"]
    s.svc.skills.set_enabled("calculator", False)
    await s._user_turn(speech(), None)
    assert "calculate" not in s.history[0]["content"]
    assert s.history[1]["content"] == "hello"
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
