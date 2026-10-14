"""Will the model speak a short preamble before a tool call (so Kira isn't silent while deciding)?
Usage: python llm_preamble.py <model>"""
import re
import sys

from common import load
import llm_bench as B

MODEL = sys.argv[1]
B.MODEL = MODEL
B.SYSTEM = B.SYSTEM.replace(
    "Never invent facts a tool could give you.",
    "Never invent facts a tool could give you. When you use a tool, first say a very short natural phrase such as "
    "\"Let me look that up.\" or \"One sec, checking.\", then call the tool in the same reply.")
load(MODEL, ctx_size=16384)
cases = [c for c in B.TOOL_CASES if c[1] != [set()]][:12] + [c for c in B.TOOL_CASES if c[1] == [set()]][:3]
n_tool = n_pre = n_ok = 0
for prompt, expected, arg_re in cases:
    o = B.stream([{"role": "system", "content": B.SYSTEM}, {"role": "user", "content": prompt}], max_tokens=160)
    names = {tc["function"]["name"] for tc in o["tool_calls"]}
    pre = o["text"].strip()
    if names:
        n_tool += 1
        n_pre += bool(pre)
    good = names in expected
    n_ok += good
    print(f"{'OK ' if good else 'BAD'} {prompt[:55]:<55} tools={sorted(names) or '-'} preamble@{o['first_token'] or 0:.2f}s "
          f"{pre[:50]!r} tool@{o.get('tool_at') or 0:.2f}s")
print(f"{MODEL}: correct {n_ok}/{len(cases)}, tool turns with a spoken preamble {n_pre}/{n_tool}")
