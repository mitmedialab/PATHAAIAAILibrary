"""ai.evaluate: replacing "it seems better" with a number."""

from ._state import state
from ._words import numbers


class EvalResult:
    """What an evaluation found. ``print(result)`` shows ``3/4 (75%)``."""

    def __init__(self, label, passed, total, failures, tokens, cost_usd):
        self.label = label
        self.passed = passed
        self.total = total
        self.score = passed / total if total else 0.0
        self.failures = failures
        self.tokens = tokens
        self.cost_usd = cost_usd

    def __str__(self):
        return f"{self.passed}/{self.total} ({self.score:.0%})"

    def __repr__(self):
        name = f" {self.label!r}" if self.label else ""
        return f"<EvalResult{name} {self}>"


def evaluate(fn, cases, check=None, label="", verbose=True):
    """Call ``fn(input)`` for every case and score the answers.

    cases: (input, expected) pairs, or dicts with "input" and "expected" keys.
        expected may be a string to find (any case), a number, a list of
        strings that must all appear, or None for "any answer".
    check: "exact", "number", or your own check(answer, expected) -> True/False.

    A case that raises an error counts as a failure and the eval goes on.
    """
    tokens_before = state.input_tokens + state.output_tokens
    cost_before = state.cost_usd
    passed, failures = 0, []
    cases = list(cases)
    for number, case in enumerate(cases, start=1):
        given, expected = (case["input"], case.get("expected")) if isinstance(case, dict) else case
        error = None
        try:
            answer = fn(given)
            ok = _check(answer, expected, check)
        except Exception as e:
            answer, ok, error = None, False, f"{type(e).__name__}: {e}"
        if ok:
            passed += 1
        else:
            failures.append({"case": number, "input": given, "expected": expected, "answer": answer, "error": error})
        if verbose:
            shown = error or _short(answer)
            print(f"  {'pass' if ok else 'FAIL'}  {_short(given)} -> {shown}", flush=True)
    result = EvalResult(
        label, passed, len(cases), failures,
        tokens=state.input_tokens + state.output_tokens - tokens_before,
        cost_usd=round(state.cost_usd - cost_before, 6),
    )
    if verbose:
        name = f"[{label}] " if label else ""
        print(f"{name}passed {result} · {result.tokens:,} tokens · ${result.cost_usd:.4f}", flush=True)
    return result


def _check(answer, expected, check):
    if callable(check):
        return bool(check(answer, expected))
    if check == "exact":
        return str(answer).strip().lower() == str(expected).strip().lower()
    if check == "number" or (isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        found = [answer] if isinstance(answer, (int, float)) else numbers(answer)
        return any(abs(float(n) - float(expected)) < 1e-6 for n in found)
    if check is not None:
        raise ValueError('check must be "exact", "number" or a function')
    if expected is None:
        return answer is not None and str(answer).strip() != ""
    if isinstance(expected, (list, tuple)):
        return all(str(part).lower() in str(answer).lower() for part in expected)
    if isinstance(expected, bool):
        return answer == expected
    return str(expected).lower() in str(answer).lower()


def _short(value, width=60):
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "…"
