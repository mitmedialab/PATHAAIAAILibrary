"""@ai.tool: a function the model may call."""

import inspect
import re
import types
import typing

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from ._state import MODELS, state

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def tool(func=None, *, name=None, description=None):
    """Mark a function the model may call.

    The model sees the name, the docstring and the typed parameters, never the
    code. Describe each parameter in an ``Args:`` section of the docstring.

        @ai.tool
        def price_lookup(item: str) -> str:
            '''Look up the price of an item in the school store.

            Args:
                item: the thing to look up, like "milk"
            '''
            return PRICES.get(item, "not sold here")

    ``@ai.tool(name=..., description=...)`` changes what the model sees without
    renaming the Python function.
    """
    if func is None:  # used as @ai.tool(name=..., description=...)
        return lambda f: Tool(f, name=name, description=description)
    return Tool(func, name=name, description=description)


class Tool:
    """A function wrapped so a model can call it. It still works as a function."""

    def __init__(self, func, name=None, description=None):
        if not callable(func):
            raise TypeError("@ai.tool goes on a function")
        self.func = func
        self.name = name or func.__name__
        summary, param_docs = _parse_docstring(func.__doc__ or "")
        self.description = description or summary or f"The {self.name} function."
        self.parameters, self.required = _parameters(func, param_docs)
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def schema(self, provider=None):
        """The exact JSON the selected model's API receives for this tool.

        Pass provider="anthropic", "openai" or "gemini" to see another
        provider's format for the same tool.
        """
        provider = provider or MODELS[state.model]["provider"]
        if provider == "anthropic":
            from langchain_anthropic import convert_to_anthropic_tool

            return convert_to_anthropic_tool(self.as_langchain())
        if provider == "openai":
            return convert_to_openai_tool(self.as_langchain())
        if provider == "gemini":
            from langchain_google_genai._function_utils import convert_to_genai_function_declarations

            declarations = convert_to_genai_function_declarations([self.as_langchain()])
            return declarations[0].model_dump(exclude_none=True, mode="json")
        raise ValueError('provider must be "anthropic", "openai" or "gemini"')

    def as_langchain(self):
        """This tool as a LangChain StructuredTool (what ai.Agent hands the agent)."""
        input_schema = {"type": "object", "properties": self.parameters, "required": self.required}

        def run(**kwargs):
            try:
                result = self.func(**kwargs)
            except Exception as e:  # the model sees the error and can recover
                return f"Error: {type(e).__name__}: {e}"
            return result if isinstance(result, str) else repr(result)

        return StructuredTool.from_function(
            func=run, name=self.name, description=self.description, args_schema=input_schema
        )

    def __repr__(self):
        return f"<ai.tool {self.name}({', '.join(self.parameters)})>"


def _parse_docstring(doc):
    """Split a docstring into its summary and per-parameter descriptions.

    Understands Google style (``Args:`` sections) and reST style (``:param x:``).
    """
    doc = inspect.cleandoc(doc)
    params = {}
    for pname, text in re.findall(r"^\s*:param\s+(?:\w+\s+)?(\w+):\s*(.+)$", doc, re.M):
        params[pname] = text.strip()
    section = re.search(r"^\s*(?:Args|Arguments|Parameters):\s*$(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    if section:
        current = None
        for line in section.group(1).splitlines():
            match = re.match(r"^\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if match:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
    summary_lines = []
    for line in doc.splitlines():
        if re.match(r"^\s*(Args|Arguments|Parameters|Returns|Raises):|^\s*:(param|return|raises)", line):
            break
        summary_lines.append(line)
    return " ".join(" ".join(summary_lines).split()), params


def _parameters(func, docs):
    hints = typing.get_type_hints(func)
    properties, required = {}, []
    for pname, param in inspect.signature(func).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = _json_type(hints.get(pname, str))
        if pname in docs:
            prop["description"] = docs[pname]
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = param.default
        properties[pname] = prop
    return properties, required


def _json_type(hint):
    origin = typing.get_origin(hint)
    if origin in (list, tuple, set):
        args = typing.get_args(hint)
        return {"type": "array", "items": _json_type(args[0]) if args else {"type": "string"}}
    if origin is typing.Literal:
        return {"type": "string", "enum": [str(a) for a in typing.get_args(hint)]}
    if origin in (typing.Union, types.UnionType):  # Optional[x] or x | None
        options = [a for a in typing.get_args(hint) if a is not type(None)]
        return _json_type(options[0]) if options else {"type": "string"}
    return {"type": _JSON_TYPES.get(origin or hint, "string")}
