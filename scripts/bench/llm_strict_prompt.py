"""Tool reliability with a stricter system prompt (date + stale-knowledge rule). Usage: python llm_prompt2.py <model>"""
import sys

from common import dump, load
import llm_bench as B

MODEL = sys.argv[1]
B.MODEL = MODEL
B.SYSTEM = B.SYSTEM.replace(
    "You have tools. Use them when they help, and answer directly when they don't:",
    "Today is Wednesday, October 14, 2026. Your own knowledge stops in 2024 and is out of date, so never answer "
    "questions about anything that changes over time from memory.\n\n"
    "You have tools. Use them when they help, and answer directly when they don't:").replace(
    "- calculate for any arithmetic beyond the trivial.",
    "- calculate for any arithmetic beyond the trivial, including percentages, tips and splitting bills.").replace(
    "Never invent facts a tool could give you.",
    "You must call web_search for news, sports results, prices, exchange rates, weather, opening hours, schedules and "
    "anything described as latest, current, today or this year. Never invent facts a tool could give you.")
load(MODEL, ctx_size=16384)
res = B.bench_tools()
dump(f"llm_prompt2_{MODEL}", res)
