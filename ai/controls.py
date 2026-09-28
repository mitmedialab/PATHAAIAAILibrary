"""The knobs and switches, and the run's usage meter."""

from . import _state
from ._engine import count_tokens
from ._state import DEFAULTS, MODELS, PRICES, state
from .errors import AiError


def set_model(name):
    """Choose the model by its short name.

        Claude:  "haiku" (fast, cheap)     "sonnet" (stronger, dearer)
        OpenAI:  "gpt-mini" (fast, cheap)  "gpt" (stronger, dearer)
        Gemini:  "flash" (fast, cheap)     "pro" (stronger, dearer)
    """
    name = str(name).lower().strip()
    if name not in MODELS:
        choices = ", ".join(f'"{m}"' for m in MODELS)
        raise AiError("bad_request", f"model must be one of {choices}, not {name!r}")
    if MODELS[name]["provider"] != MODELS[state.model]["provider"]:
        state.fell_back = False  # a different provider gets a fresh chance
    state.model = name
    if state.engine != "sim" and not _state.has_key(MODELS[name]["provider"]):
        _state.say(f"{_state.missing_key_line()}, so the simulator will answer.")


def set_temperature(t):
    """0 (repeatable) to 1 (adventurous). Models that reject it (Sonnet) ignore it."""
    if t is not None and not 0 <= t <= 1:
        raise AiError("bad_request", "temperature must be between 0 and 1")
    state.temperature = t


def backend_name():
    """Which engine answers, like "claude (haiku)", "openai (gpt-mini)", "gemini (flash)" or "sim"."""
    return _state.backend_name()


def use_sim():
    """Use the simulator: free, offline and repeatable. Handy for tests."""
    state.engine = "sim"


def _use(provider):
    if MODELS[state.model]["provider"] != provider:
        state.model = DEFAULTS[provider]
    state.engine = "api"
    state.fell_back = False
    if not _state.has_key(provider):
        if state.strict:
            raise AiError("unavailable", f"{_state.missing_key_line()}.")
        _state.say(f"{_state.missing_key_line()}, so the simulator will keep answering.")


def use_claude():
    """Use a real Claude model (haiku, unless a Claude model is already chosen)."""
    _use("anthropic")


def use_openai():
    """Use a real OpenAI model (gpt-mini, unless an OpenAI model is already chosen)."""
    _use("openai")


def use_gemini():
    """Use a real Gemini model (flash, unless a Gemini model is already chosen)."""
    _use("gemini")


def strict(on=True):
    """Raise AiError("unavailable") instead of falling back to the simulator."""
    state.strict = bool(on)


def usage():
    """The run's meter so far, as a dict."""
    return {
        "calls": state.calls,
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "cost_usd": round(state.cost_usd, 6),
        "estimated": state.estimated,
    }


def usage_report():
    """Print the meter as one line."""
    note = " (simulator counts are estimates, and free)" if state.estimated else ""
    print(
        f"[ai] {state.calls} calls · {state.input_tokens:,} tokens in · "
        f"{state.output_tokens:,} out · ${state.cost_usd:.4f}{note}",
        flush=True,
    )


def last_response():
    """The last reply as a plain dict: text, tool_call, stop_reason, model, usage."""
    return dict(state.last) if state.last else None


def estimate_cost(input_tokens, output_tokens):
    """Dollars for that many tokens at the selected model's list price."""
    price = PRICES[state.model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def status():
    """Print one line: engine, model, temperature, strict mode and the meter."""
    temperature = "default" if state.temperature is None else state.temperature
    print(
        f"[ai] engine: {backend_name()} · model: {state.model} ({MODELS[state.model]['id']}) · temperature: {temperature} · "
        f"strict: {'on' if state.strict else 'off'} · {state.calls} calls, "
        f"{state.input_tokens + state.output_tokens:,} tokens, ${state.cost_usd:.4f}",
        flush=True,
    )


__all__ = [
    "set_model", "set_temperature", "backend_name", "use_sim", "use_claude", "use_openai",
    "use_gemini", "strict",
    "usage", "usage_report", "last_response", "count_tokens", "estimate_cost", "status",
]
