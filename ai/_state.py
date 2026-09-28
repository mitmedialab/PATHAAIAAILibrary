"""Run-wide settings, the model list and the usage meter.

Everything here resets when a new Python process starts, which is this
library's equivalent of "every press of Run starts afresh".
"""

import os

try:  # A .env file next to the student's program may hold the API keys.
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # python-dotenv is optional
    pass


# The providers the course supports, and where each one's key lives.
PROVIDERS = {
    "anthropic": {"label": "Claude", "short": "claude", "keys": ["ANTHROPIC_API_KEY"]},
    "openai": {"label": "OpenAI", "short": "openai", "keys": ["OPENAI_API_KEY"]},
    "gemini": {"label": "Gemini", "short": "gemini", "keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"]},
}

# The models a student may choose, by short name. This list is the cost policy.
# Prices are US dollars per million tokens, at list price.
#   temperature: whether the model accepts the temperature knob
#   settings: extra settings that keep "thinking" from eating small max_tokens
MODELS = {
    "haiku": {
        "provider": "anthropic", "id": "claude-haiku-4-5", "input": 1.00, "output": 5.00,
        "temperature": True, "settings": {},
    },
    "sonnet": {
        "provider": "anthropic", "id": "claude-sonnet-5", "input": 2.00, "output": 10.00,
        "temperature": False, "settings": {"thinking": {"type": "disabled"}},
    },
    "gpt-mini": {
        "provider": "openai", "id": "gpt-5.4-mini", "input": 0.75, "output": 4.50,
        "temperature": True, "settings": {"reasoning_effort": "none"},
    },
    "gpt": {
        "provider": "openai", "id": "gpt-5.4", "input": 2.50, "output": 15.00,
        "temperature": True, "settings": {"reasoning_effort": "none"},
    },
    "flash": {
        # Google's introductory price; it rises to $1.50 / $7.50 on 2027-01-01.
        "provider": "gemini", "id": "gemini-3.7-flash", "input": 0.75, "output": 3.75,
        "temperature": True, "settings": {"thinking_level": "minimal"},
    },
    "pro": {
        "provider": "gemini", "id": "gemini-3.1-pro-preview", "input": 2.00, "output": 12.00,
        "temperature": True, "settings": {"thinking_level": "low"},
    },
}

# The fast default for each provider, in the order they are tried.
DEFAULTS = {"anthropic": "haiku", "openai": "gpt-mini", "gemini": "flash"}

PRICES = {name: {"input": m["input"], "output": m["output"]} for name, m in MODELS.items()}


def has_key(provider):
    return any(os.environ.get(key) for key in PROVIDERS[provider]["keys"])


def default_model():
    """AI_MODEL if the teacher set one, else the fast model of the first provider with a key."""
    chosen = os.environ.get("AI_MODEL", "").strip().lower()
    if chosen in MODELS:
        return chosen
    for provider, model in DEFAULTS.items():
        if has_key(provider):
            return model
    return "haiku"


class _State:
    def __init__(self):
        self.engine = "auto"  # "auto", "api" or "sim"
        self.fell_back = False  # True once the real model failed and the sim took over
        self.model = default_model()
        self.temperature = None
        self.strict = False
        self.announced = False
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.estimated = False  # True if any call was answered by the simulator
        self.last = None  # the last reply, as a plain dict


state = _State()


def spec():
    """The current model's entry in MODELS."""
    return MODELS[state.model]


def provider():
    return PROVIDERS[spec()["provider"]]


def using_api():
    """True when the next call will go to a real model."""
    if state.fell_back or state.engine == "sim":
        return False
    return has_key(spec()["provider"])


def backend_name():
    if using_api():
        return f"{provider()['short']} ({state.model})"
    return "sim"


def say(line):
    """Print one of the library's [ai] console lines."""
    print(f"[ai] {line}", flush=True)


def missing_key_line():
    return f"No {' or '.join(provider()['keys'])} is set for {state.model}"


def announce_once():
    if state.announced:
        return
    state.announced = True
    if using_api():
        say(f"Talking to {provider()['label']} ({state.model})…")
    elif state.engine == "sim":
        say("Using the simulator (SimLLM), as requested. Its answers are imitations, not a real model.")
    elif not any(has_key(p) for p in PROVIDERS):
        say(
            "No AI model is configured (no API key for Claude, OpenAI or Gemini), so the "
            "simulator (SimLLM) will answer. Its answers are imitations, not a real model."
        )
    else:
        say(f"{missing_key_line()}, so the simulator (SimLLM) will answer. Its answers are imitations.")


def record(input_tokens, output_tokens, on_api):
    state.calls += 1
    state.input_tokens += input_tokens
    state.output_tokens += output_tokens
    if on_api:
        price = spec()
        state.cost_usd += (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    else:
        state.estimated = True
