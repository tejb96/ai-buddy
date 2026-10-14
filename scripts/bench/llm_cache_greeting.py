"""Does FLM's prompt cache survive (a) a generated greeting as the first turn and (b) a tool round-trip?"""
import json
import sys
import time

sys.argv = ["x", "qwen3.6-moe-35b-a3b-FLM"]
from common import http, load  # noqa: E402
import llm_bench as B  # noqa: E402

M = B.MODEL
RESULT = {"query": "Calgary weather tomorrow", "results": [
    {"title": "Calgary, AB 7-Day Forecast - Environment Canada", "url": "https://weather.gc.ca/",
     "text": "Tomorrow: A mix of sun and cloud. 40 percent chance of showers in the afternoon. Wind west 20 km/h "
             "gusting to 40. High 14. Tomorrow night: Cloudy periods. Low plus 2. Thursday: Sunny. High 17."},
    {"title": "Calgary weather | The Weather Network", "url": "https://www.theweathernetwork.com/",
     "text": "Wednesday: High 13, low 1. Showers possible after 2 PM with about 2 mm of rain. Winds W 25 km/h."}]}


def run(messages, label, max_tokens=120):
    t0 = time.perf_counter()
    out, calls, first, usage = "", [], None, {}
    with http.stream("POST", "/chat/completions", json={"model": M, "messages": messages, "tools": B.TOOLS,
                                                         "stream": True, "think": False, "temperature": 0.6,
                                                         "max_completion_tokens": max_tokens}) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content"):
                        first = first or time.perf_counter() - t0
                        out += delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        first = first or time.perf_counter() - t0
                        calls.append(tc)
    print(f"{label:<34} first output {first or 0:5.2f}s  new prompt tokens {usage.get('prompt_tokens')}  "
          f"text={out[:70]!r} calls={[(c['function']['name'], c['function']['arguments']) for c in calls]}")
    return out, calls


load(M, ctx_size=16384)
sysmsg = {"role": "system", "content": B.SYSTEM}
greet, _ = run([sysmsg, {"role": "user", "content": "[The user just opened the app. Greet them in one short sentence.]"}],
               "1 greeting (fresh)", 40)
base = [sysmsg, {"role": "user", "content": "[The user just opened the app. Greet them in one short sentence.]"},
        {"role": "assistant", "content": greet}]
q = {"role": "user", "content": "What's the weather going to be like in Calgary tomorrow?"}
text, calls = run(base + [q], "2 user turn after greeting", 300)
print("   full text:", repr(text))
if not calls:
    calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": json.dumps({"query": "Calgary weather tomorrow"})}}]
    text = ""
if calls:
    call = calls[0]
    name = call["function"]["name"] or "web_search"
    for variant in ("string args", "dict args"):
        args = call["function"]["arguments"]
        tc = {"id": call.get("id", "call_1"), "type": "function",
              "function": {"name": name, "arguments": args if variant == "string args" else json.loads(args)}}
        msgs = base + [q, {"role": "assistant", "content": text, "tool_calls": [tc]},
                       {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(RESULT)}]
        run(msgs, f"3 tool result ({variant})", 90)
