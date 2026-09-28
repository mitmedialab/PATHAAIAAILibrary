"""The ai library: one plain name for each idea in the course.

    import ai
    print(ai.ask("What is an agent?"))

With an API key for Claude, OpenAI or Gemini set (in the environment or a .env
file) the answers come from that model. Without one they come from SimLLM, a
free offline simulator. ai.set_model() switches between models.
"""

from ._engine import DevanModel
from ._state import PRICES
from .agent import Agent
from .controls import (
    backend_name, count_tokens, estimate_cost, last_response, set_model, set_temperature,
    status, strict, usage, usage_report, use_claude, use_gemini, use_openai, use_sim,
)
from .data import Verdict, ask_json, classify, extract, judge
from .errors import AiError
from .evals import EvalResult, evaluate
from .retrieval import Hit, Index, chunk, cosine, embed
from .talk import Chat, ask
from .tools import Tool, tool


def langchain_model(max_tokens=1024):
    """The library's engine as a LangChain chat model, for moving on to LangChain.

    It follows ai.set_model() and falls back to the simulator like everything else.
    """
    return DevanModel(max_tokens=max_tokens)


__all__ = [
    "ask", "Chat", "ask_json", "extract", "classify", "judge", "Verdict",
    "tool", "Tool", "Agent", "embed", "cosine", "chunk", "Index", "Hit",
    "evaluate", "EvalResult", "usage", "usage_report", "last_response",
    "count_tokens", "estimate_cost", "PRICES", "status", "set_model",
    "set_temperature", "backend_name", "use_sim", "use_claude", "use_openai",
    "use_gemini", "strict",
    "AiError", "langchain_model",
]
