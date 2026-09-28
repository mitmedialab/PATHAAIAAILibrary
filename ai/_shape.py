"""Turning a student's *shape* into a JSON schema, and a reply back into Python.

A shape is a dict of field names to what each should hold::

    {"name": str, "age": int}            # types
    {"name": "Sam", "age": 14}           # example values ("like this")
    {"mood": ["happy", "sad"]}           # allowed answers (pick one)
    {"tags": [str]}                      # a list of strings
    {"address": {"city": str}}           # a nested object
    {"count": (int, "how many apples")}  # a type with a description
"""

from .errors import AiError

MAX_FIELDS = 30
MAX_DEPTH = 4

_TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def to_schema(shape):
    """The JSON schema for a shape, checked against the size limits."""
    if not isinstance(shape, dict) or not shape:
        raise AiError("bad_request", "shape must be a dict of field names, like {\"name\": str, \"age\": int}")
    counter = [0]
    schema = _object(shape, 1, counter)
    schema["title"] = "answer"
    schema["description"] = "The answer, as data."
    return schema


def _object(shape, depth, counter):
    if depth > MAX_DEPTH:
        raise AiError("bad_request", f"the shape is nested more than {MAX_DEPTH} levels deep")
    properties = {}
    for name, entry in shape.items():
        counter[0] += 1
        if counter[0] > MAX_FIELDS:
            raise AiError("bad_request", f"the shape has more than {MAX_FIELDS} fields")
        properties[str(name)] = _field(entry, depth, counter, str(name))
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _field(entry, depth, counter, name):
    # (int, "a count"): a type or example, with a description the model reads.
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], str):
        schema = _field(entry[0], depth, counter, name)
        schema["description"] = entry[1]
        return schema
    if isinstance(entry, type) and entry in _TYPE_NAMES:
        return {"type": _TYPE_NAMES[entry]}
    if entry is list:
        return {"type": "array", "items": {"type": "string"}}
    if entry is dict:
        # Structured outputs cannot describe a free-form object, so a bare
        # `dict` travels as a list of key/value pairs and is rebuilt on return.
        pair = {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        }
        return {"type": "array", "items": pair, "description": "key/value pairs"}
    # Example values. bool is checked before int because True is also an int.
    if isinstance(entry, bool):
        return {"type": "boolean"}
    if isinstance(entry, int):
        return {"type": "integer"}
    if isinstance(entry, float):
        return {"type": "number"}
    if isinstance(entry, str):
        return {"type": "string", "description": f"like {entry!r}"}
    if isinstance(entry, dict):
        return _object(entry, depth + 1, counter)
    if isinstance(entry, list):
        if len(entry) == 1 and not isinstance(entry[0], str):
            return {"type": "array", "items": _field(entry[0], depth + 1, counter, name)}
        if entry and all(isinstance(x, str) for x in entry):
            return {"type": "string", "enum": list(entry)}
        if not entry:
            return {"type": "array", "items": {"type": "string"}}
    raise AiError("bad_request", f"field {name!r}: don't know how to describe {entry!r} in a shape")


def decode(value, shape):
    """Coerce a parsed reply so every field has the type the shape asked for."""
    if not isinstance(value, dict):
        raise AiError("bad_json", "the reply was not a JSON object")
    return {str(name): _decode_field(value.get(str(name)), entry) for name, entry in shape.items()}


def _decode_field(value, entry):
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], str):
        return _decode_field(value, entry[0])
    if entry is dict:
        if isinstance(value, dict):
            return value
        return {p.get("key", ""): p.get("value", "") for p in value or [] if isinstance(p, dict)}
    if isinstance(entry, dict):
        return decode(value if isinstance(value, dict) else {}, entry)
    if isinstance(entry, list):
        if entry and all(isinstance(x, str) for x in entry):  # pick one
            for option in entry:
                if str(value).strip().lower() == option.lower():
                    return option
            return entry[-1] if value is None else value
        item = entry[0] if len(entry) == 1 else str
        return [_decode_field(v, item) for v in (value or [])]
    kind = entry if isinstance(entry, type) else type(entry)
    return _coerce(value, kind)


def _coerce(value, kind):
    try:
        if kind is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("true", "yes", "1")
            return bool(value)
        if kind is int:
            return int(float(value)) if value not in (None, "") else 0
        if kind is float:
            return float(value) if value not in (None, "") else 0.0
        if kind is list:
            return list(value or [])
        if kind is str:
            return "" if value is None else str(value)
    except (TypeError, ValueError):
        return 0 if kind in (int, float) else value
    return value
