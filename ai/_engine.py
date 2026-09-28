"""The engine: one LangChain chat model that answers from a real model or the simulator.

``DevanModel`` is an ordinary LangChain ``BaseChatModel``. Every helper in the
library, and the LangChain agent behind ``ai.Agent``, calls it. On each call it
decides which engine answers (Claude, OpenAI, Gemini or SimLLM), falls back to
the simulator when the real model is out of reach, maps API errors to
``AiError`` and feeds the usage meter.

Only this file (and the model list in ``_state.py``) knows about providers.
"""

import json
import re
from typing import Any

import httpx
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelError,
    ModelInvalidRequestError,
    ModelRateLimitError,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from . import _sim
from ._state import announce_once, record, say, spec, state, using_api
from .errors import AiError

DECLINED = "The model declined to answer that."


class DevanModel(BaseChatModel):
    """A LangChain chat model that follows the library's settings."""

    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "devan-ai"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=list(tools), **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        kwargs.setdefault("max_tokens", self.max_tokens)
        message = complete(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])


def complete(messages, tools=None, schema=None, task=None, examples=0,
             max_tokens=1024, temperature=None, tool_choice=None, **_ignored):
    """Answer a list of LangChain messages with whichever engine is active.

    ``schema`` asks for JSON matching that schema. ``task`` and ``examples``
    are hints only the simulator reads.
    """
    announce_once()
    if using_api():
        try:
            return _api(messages, tools, schema, max_tokens, temperature, tool_choice)
        except AiError:
            raise
        except ModelRateLimitError as e:
            if "insufficient_quota" in str(e):  # OpenAI: the account is out of credit
                _fall_back("the account's credit is used up", e)
            else:
                raise AiError("rate_limited", "Too many calls too quickly. Wait a minute, then run again.") from e
        except ContextOverflowError as e:
            raise AiError("bad_request", "The request is too long for the model.") from e
        except ModelInvalidRequestError as e:
            if "credit balance" not in str(e).lower():  # Anthropic: out of credit
                raise AiError("bad_request", _api_message(e)) from e
            _fall_back("the account's credit is used up", e)
        except (ModelError, httpx.TransportError) as e:
            _fall_back(_api_message(e), e)
    return _simulate(messages, tools, schema, task, examples)


def _fall_back(reason, error):
    label = {"anthropic": "Claude", "openai": "OpenAI", "gemini": "Gemini"}[spec()["provider"]]
    if state.strict:
        raise AiError("unavailable", f"{label} is unavailable: {reason}") from error
    state.fell_back = True
    say(f"{label} is unavailable ({reason}). Finishing this run on the simulator.")


def _api_message(error):
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message", "")
    else:
        message = str(error)
    message = message.strip().rstrip(".")
    return message[:200] or type(error).__name__


# ------------------------------------------------------------ real models


def chat_model(max_tokens=1024, temperature=None):
    """A LangChain chat model for the selected model, with the library's settings."""
    model = spec()
    settings = dict(model["settings"])
    if temperature is None:
        temperature = state.temperature
    if not model["temperature"]:
        temperature = None  # the model rejects it, so the knob is dropped
    common = {"model": model["id"], "max_tokens": max_tokens, "temperature": temperature, "timeout": 60}
    if model["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**common, max_retries=2, **settings)
    if model["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**common, max_retries=2, **settings)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(**common, retries=2, **settings)


def _api(messages, tools, schema, max_tokens, temperature, tool_choice):
    llm = chat_model(max_tokens, temperature)
    if schema:
        return _api_json(llm, messages, schema)
    if tools:
        llm = llm.bind_tools(tools, tool_choice=tool_choice) if tool_choice else llm.bind_tools(tools)
    reply = llm.invoke(messages)
    if stop_reason(reply) == "refusal":
        reply = AIMessage(content=DECLINED, usage_metadata=reply.usage_metadata,
                          response_metadata={**reply.response_metadata, "stop_reason": "refusal"})
    _remember(reply, on_api=True)
    return reply


def _api_json(llm, messages, schema):
    """Structured outputs: the provider constrains the reply to the schema."""
    options = {"method": "json_schema", "include_raw": True}
    if spec()["provider"] == "openai":
        options["strict"] = True
    try:
        out = llm.with_structured_output(schema, **options).invoke(messages)
        raw, parsed = out["raw"], out["parsed"]
    except ModelInvalidRequestError as e:
        if isinstance(e, ContextOverflowError):
            raise
        # The API would not take the schema. Retry once asking for JSON in prose.
        raw = llm.invoke(list(messages) + [_json_instruction(schema)])
        parsed = _parse_json(raw.text)
    _remember(raw, on_api=True)
    if stop_reason(raw) == "refusal":
        raise AiError("bad_json", DECLINED)
    if parsed is None:
        raise AiError("bad_json", "The reply could not be read as JSON.")
    return AIMessage(content=json.dumps(parsed), usage_metadata=raw.usage_metadata,
                     response_metadata=raw.response_metadata)


def _json_instruction(schema):
    return HumanMessage(
        "Reply with only a JSON object, no other text, matching this JSON schema:\n" + json.dumps(schema)
    )


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


_REFUSALS = {"refusal", "content_filter", "safety", "prohibited_content", "blocklist", "spii", "recitation"}


def stop_reason(reply):
    """Why the model stopped, in one vocabulary for every provider:
    "end_turn", "tool_use", "max_tokens" or "refusal"."""
    meta = reply.response_metadata or {}
    raw = str(meta.get("stop_reason") or meta.get("finish_reason") or "").lower()
    if reply.tool_calls:
        return "tool_use"
    if raw in _REFUSALS or reply.additional_kwargs.get("refusal"):
        return "refusal"
    if raw in ("max_tokens", "length"):
        return "max_tokens"
    return "end_turn"


# ---------------------------------------------------------------- simulator


def _sim_tool(tool):
    """Any LangChain tool, in the {name, description, input_schema} form the simulator reads."""
    function = convert_to_openai_tool(tool)["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {}),
    }


def _simulate(messages, tools, schema, task, examples):
    sim_tools = [_sim_tool(t) for t in tools] if tools else None
    answer = _sim.respond(messages, tools=sim_tools, schema=schema, task=task, examples=examples)
    tokens_in = count_tokens(" ".join(str(m.content) for m in messages))
    if "json" in answer:
        reply = AIMessage(content=json.dumps(answer["json"]))
    elif "tool_calls" in answer:
        reply = AIMessage(content="", tool_calls=answer["tool_calls"])
    else:
        reply = AIMessage(content=answer["text"])
    tokens_out = count_tokens(reply.content or json.dumps(answer.get("tool_calls", "")))
    reply.usage_metadata = {"input_tokens": tokens_in, "output_tokens": tokens_out,
                            "total_tokens": tokens_in + tokens_out}
    reply.response_metadata = {"model": "sim",
                               "stop_reason": "tool_use" if reply.tool_calls else "end_turn"}
    _remember(reply, on_api=False)
    return reply


def count_tokens(text):
    """A rough count: about four characters to a token."""
    text = str(text)
    return max(1, round(len(text) / 4)) if text.strip() else 0


# ------------------------------------------------------------------ meter


def _remember(reply, on_api):
    usage = reply.usage_metadata or {}
    tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    record(tokens_in, tokens_out, on_api)
    call = reply.tool_calls[0] if reply.tool_calls else None
    meta = reply.response_metadata or {}
    state.last = {
        "text": reply.text,
        "tool_call": {"name": call["name"], "input": call["args"]} if call else None,
        "stop_reason": stop_reason(reply),
        "model": meta.get("model") or meta.get("model_name") or spec()["id"],
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }
