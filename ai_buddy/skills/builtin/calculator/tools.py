"""calculate: exact arithmetic with a small, safe expression evaluator (no eval)."""

from __future__ import annotations

import ast
import math
import operator
import re

from ai_buddy.tools import Tool, ToolContext, ToolResult

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "log": math.log,
          "log10": math.log10, "log2": math.log2, "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial, "pow": pow}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalcError(ValueError):
    pass


def evaluate(expression: str) -> float | int:
    text = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-").strip()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)    # thousands separators ("1,250"), not argument commas
    if len(text) > 300:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"can't read {expression!r}") from e
    return _eval(tree.body)


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e100):
            raise CalcError("number too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as e:
            raise CalcError("division by zero") from e
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords):
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and (not args or args[0] > 170):
            raise CalcError("number too large")
        try:
            return _FUNCS[node.func.id](*args)
        except (ValueError, TypeError) as e:
            raise CalcError(f"{node.func.id}: {e}") from e
    raise CalcError("only numbers, + - * / // % ** and math functions are allowed")


def tidy(value) -> float | int:
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer() and abs(value) < 1e15:
            return int(value)
        return float(f"{value:.10g}")
    return value


_NUM = r"\$?\d[\d,]*(?:\.\d+)?"
# Whole questions only ("what's 18% of 240?", "what is 12 times 37"), so "I got 18% of the votes" isn't one.
_PERCENT_Q = re.compile(rf"^(?:what(?:'s| is)|how much is|calculate|work out)\s+(?P<p>{_NUM})\s*(?:%|percent)\s+of\s+"
                        rf"(?P<n>{_NUM})\s*[?.!]?$", re.I)
_BINARY_Q = re.compile(rf"^(?:what(?:'s| is)|how much is|calculate|work out)\s+(?P<a>{_NUM})\s+(?P<op>times|x|"
                       rf"multiplied by|plus|minus|divided by|over)\s+(?P<b>{_NUM})\s*[?.!]?$", re.I)
_OPS = {"times": "*", "x": "*", "multiplied by": "*", "plus": "+", "minus": "-", "divided by": "/", "over": "/"}
_LEAD = re.compile(r"^\s*(?:(?:um|uh|so|okay|ok|quick one|hey kira|kira)[,:.\s]+)+", re.I)


def _n(text: str) -> str:
    return text.replace("$", "").replace(",", "")


class Calculator(Tool):
    name = "calculate"
    description = ("Evaluate an arithmetic expression exactly, e.g. '137.50 * 1.15 / 4' or 'sqrt(729 * 3)'. Supports "
                   "+ - * / // % ** and parentheses, sqrt, round, abs, min, max, log, exp, sin, cos, tan, pi.")
    parameters = {"expression": {"type": "string", "description": "The expression to evaluate."}}
    required = ["expression"]
    label = "Calculating"
    filler = "Let me work that out."

    def describe(self, args: dict) -> str:
        return args.get("expression", "")

    def route(self, text: str, ctx: ToolContext) -> dict | None:
        q = _LEAD.sub("", text.strip())
        q = re.sub(r"^.*?[.!:]\s+(?=(?:what|how|calculate|work)\b)", "", q, flags=re.I)  # "Okay. Quick one: what's..."
        if m := _PERCENT_Q.match(q):
            return {"expression": f"{_n(m['p'])} / 100 * {_n(m['n'])}"}
        if m := _BINARY_Q.match(q):
            return {"expression": f"{_n(m['a'])} {_OPS[m['op'].lower()]} {_n(m['b'])}"}
        return None

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        expr = str(args["expression"])
        try:
            result = tidy(evaluate(expr))
        except (CalcError, OverflowError) as e:
            return ToolResult({"expression": expr, "error": str(e)}, summary=str(e), ok=False)
        return ToolResult({"expression": expr, "result": result}, summary=f"{expr} = {result}")
