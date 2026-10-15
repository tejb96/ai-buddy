"""Does FLM honor `tool_choice`, and does it keep the prompt cache? Usage: python tool_choice.py [llm]

Kira forces the right tool in code when the user's intent is unambiguous ("what time is it in Tokyo"),
instead of trusting the model or spending an extra LLM call. That only works if FLM honors
tool_choice ("required" or a named function) without changing the prompt (a changed prompt means
FLM re-reads the whole conversation). Also checks what happens when a stream is aborted early.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from ai_buddy.config import load_config  # noqa: E402
from ai_buddy.prompts import Persona, chat_system_prompt  # noqa: E402
from ai_buddy.skills import SkillSet  # noqa: E402

from common import http, load  # noqa: E402

LLM = sys.argv[1] if len(sys.argv) > 1 else "gemma4-it-12b-FLM"
CFG = load_config()
SKILLS = SkillSet.from_config(CFG)
REG = SKILLS.registry()


def ask(messages, tool_choice=None, max_tokens=60):
    body = {"model": LLM, "messages": messages, "stream": False, "think": False, "temperature": 0.6,
            "max_completion_tokens": max_tokens, "tools": REG.schemas()}
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    r = http.post("/chat/completions", json=body)
    time.sleep(2)
    if r.status_code != 200:
        return {"error": f"{r.status_code} {r.text[:200]}"}
    d = r.json()
    msg = d["choices"][0]["message"]
    return {"text": (msg.get("content") or "")[:80], "calls": [(c["function"].get("name"), c["function"].get("arguments"))
                                                              for c in msg.get("tool_calls") or []],
            "prefilled": d.get("usage", {}).get("prompt_tokens")}


if __name__ == "__main__":
    load(LLM, ctx_size=CFG.models.llm_ctx_size)
    system = {"role": "system", "content": f"[{time.time_ns()}] " + chat_system_prompt(Persona.of(CFG), "America/Edmonton",
                                                                          SKILLS.chat_instructions())}
    joke = [system, {"role": "user", "content": "Tell me a joke about cats."}]
    print("baseline (no tool_choice):", ask(joke))
    print("same prompt again, no tool_choice:", ask(joke))
    print("same prompt, tool_choice=required:", ask(joke, "required"))
    named = {"type": "function", "function": {"name": "get_datetime"}}
    print("same prompt, tool_choice=get_datetime:", ask(joke, named))
    print("same prompt, tool_choice=none:", ask(joke, "none"))
    tokyo = [system, {"role": "user", "content": "Tell me a joke about cats."},
             {"role": "assistant", "content": "Why did the cat sit on the computer? To keep an eye on the mouse!"},
             {"role": "user", "content": "Ha. What time is it in Tokyo right now?"}]
    print("next turn, tool_choice=get_datetime:", ask(tokyo, named))

    # Abort a stream after a few tokens: does the next request still hit the cache?
    body = {"model": LLM, "messages": tokyo + [{"role": "assistant", "content": "It's late there."},
                                               {"role": "user", "content": "Tell me a long story about a dragon."}],
            "stream": True, "think": False, "temperature": 0.6, "max_completion_tokens": 300}
    n = 0
    with http.stream("POST", "/chat/completions", json=body) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and '"content"' in line:
                n += 1
                if n >= 5:
                    break   # closes the connection
    time.sleep(3)
    print(f"aborted a stream after {n} chunks; same prompt again:",
          ask(body["messages"], max_tokens=10))
    print(json.dumps({k: v.get("pid") for k, v in {m["model_name"]: m for m in
                                                    http.get("/health").json()["all_models_loaded"]}.items()}))
