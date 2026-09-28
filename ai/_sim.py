"""SimLLM: a rule-based imitation of a language model.

It is free, offline and deterministic, and it is *not* a neural network. It
exists so every lesson runs without an API key, and it imitates the famous
failure modes on purpose: it does arithmetic "in its head" and gets it wrong,
and it answers confidently about things it was never told.
"""

import re
import zlib

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ._words import STOPWORDS, WORD_NUMBERS, content_words, numbers, sentences, stem, words


def respond(messages, tools=None, schema=None, task=None, examples=0):
    """Answer a list of LangChain messages.

    Returns {"text": ...}, {"tool_calls": [...]} or {"json": {...}}.
    """
    system, history, current, after = _split(messages)
    if task:
        return {"json": _TASKS[task["kind"]](task, schema)}
    if schema:
        return {"json": fill(schema, current)}
    if tools:
        call = _choose_tool(tools, current, after)
        if call:
            return {"tool_calls": [call]}
    observations = [m for m in after if isinstance(m, ToolMessage)]
    if observations:
        return {"text": _answer_from_observations(observations)}
    return {"text": _answer(system, history, current, examples)}


def _split(messages):
    system = " ".join(_text(m) for m in messages if isinstance(m, SystemMessage))
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    last_human = max((i for i, m in enumerate(rest) if isinstance(m, HumanMessage)), default=-1)
    current = _text(rest[last_human]) if last_human >= 0 else ""
    return system, rest[: max(last_human, 0)], current, rest[last_human + 1 :]


def _text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


# ---------------------------------------------------------------- plain answers


def _answer(system, history, current, examples):
    text = current.strip()
    low = text.lower()
    prefix = "Arr! " if "pirate" in system.lower() else ""

    boundary = re.search(
        r"only (?:answer|help with|talk about|discuss|respond to)(?: questions)?(?: about)? ([a-z][a-z \-]+)",
        system.lower(),
    )
    if boundary:
        topic = boundary.group(1).strip()
        if not set(content_words(topic)) & set(content_words(text)):
            return prefix + f"Sorry, I can only help with {topic}."

    if examples:
        pairs = [(_text(history[i]), _text(history[i + 1])) for i in range(0, 2 * examples, 2)]
        return prefix + _imitate(pairs, text)

    for rule in (_grounded, _memory, _greeting, _arithmetic, _yes_or_no, _summarize, _write_about):
        reply = rule(text, low, history)
        if reply:
            return prefix + reply
    return prefix + _confident_guess(text, low)


def _grounded(text, low, history):
    if "context:" not in low or "question:" not in low:
        return None
    context, question = re.split(r"(?i)question:", re.split(r"(?i)context:", text, maxsplit=1)[1], maxsplit=1)
    q_words = set(content_words(question))
    best, best_score, best_source = None, 0, None
    for source, passage in re.findall(r"\[chunk (\d+)\]\s*(.+?)(?=\n\[chunk \d+\]|\Z)", context, re.S):
        for sentence in sentences(passage):
            score = len(q_words & set(content_words(sentence)))
            if score > best_score:
                best, best_score, best_source = sentence, score, source
    if not best:
        return "I don't know; the documents don't say."
    return f"{best} [source: chunk {best_source}]"


def _facts(history):
    facts = {}
    for message in history:
        if not isinstance(message, HumanMessage):
            continue
        said = _text(message)
        for key, value in re.findall(r"\bmy ([a-z ]{2,30}?) is ([^.,!?\n]+)", said, re.I):
            facts[key.lower().strip()] = value.strip()
        for value in re.findall(r"\b(?:i'm|i am|call me) ([A-Z][a-z]+)", said):
            facts["name"] = value
    return facts


def _memory(text, low, history):
    asked = re.search(r"what(?:'s| is) my ([a-z ]+?)\s*\??$", low)
    if asked:
        key = asked.group(1).strip()
        value = _facts(history).get(key)
        if value:
            return f"Your {key} is {value}."
        return f"I don't know your {key}. You haven't told me."
    if re.search(r"what did i (?:just )?(?:say|ask|tell you)", low):
        said = [_text(m) for m in history if isinstance(m, HumanMessage)]
        return f'You said: "{said[-1]}"' if said else "You haven't said anything to me before this."
    told = re.search(r"\bmy ([a-z ]{2,30}?) is ([^.,!?\n]+)", text, re.I)
    if told:
        key, value = told.group(1).lower().strip(), told.group(2).strip()
        if key == "name":
            return f"Nice to meet you, {value}!"
        return f"Got it. Your {key} is {value}."
    return None


def _greeting(text, low, history):
    if re.match(r"(hi|hello|hey|good (morning|afternoon|evening))\b", low):
        return "Hello! How can I help you today?"
    return None


_OPS = {
    "+": "+", "plus": "+", "-": "-", "minus": "-", "*": "*", "x": "*", "×": "*",
    "times": "*", "multiplied by": "*", "/": "/", "÷": "/", "divided by": "/",
}


def _arithmetic(text, low, history):
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(\+|-|\*|x|×|/|÷|plus|minus|times|multiplied by|divided by)\s*(-?\d+(?:\.\d+)?)",
        low,
    )
    if not match:
        return None
    a, op, b = float(match.group(1)), _OPS[match.group(2)], float(match.group(3))
    if op == "/" and b == 0:
        return "You can't divide by zero."
    exact = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
    answer = exact
    # "In its head": small sums come out right, anything harder comes out wrong.
    if op in "*/" or abs(exact) >= 100:
        answer = exact + ((int(a) * 7 + int(b) * 3) % 9 + 1) * (1 if exact >= 0 else -1)
    shown = int(answer) if float(answer).is_integer() else round(answer, 2)
    return f"{match.group(1)} {match.group(2)} {match.group(3)} = {shown}."


def _yes_or_no(text, low, history):
    if "yes or no" in low:
        return "Yes, I believe so."
    return None


def _summarize(text, low, history):
    if not re.match(r"(summari[sz]e|give me a summary|tl;?dr)", low):
        return None
    body = text.split(":", 1)[1] if ":" in text else text.split("\n", 1)[-1]
    first = sentences(body)
    return f"In short: {first[0]}" if first else "There is nothing to summarize."


def _write_about(text, low, history):
    match = re.search(r"\b(?:write|compose|draft|create)\b.*?\babout ([^.?!\n]+)", low)
    if not match:
        return None
    topic = match.group(1).strip()
    return f"{topic.capitalize()} is wonderful in its own way,\nand there is always more to say about {topic}."


def _confident_guess(text, low):
    first = (words(low) or [""])[0]
    if first == "who":
        return "That was Dr. Elena Marsh, back in 1987. I'm quite sure of it."
    if first == "when":
        return "That was in March 1987."
    if first == "where":
        return "That's in Springfield, just north of the river."
    if low.startswith(("how many", "how much")):
        return "Exactly 42."
    if first == "why":
        return "Mainly because of the weather. That's the well-known reason."
    if first in ("is", "are", "can", "does", "do", "will", "should", "was", "were", "did", "could"):
        return "Yes, absolutely."
    topic = re.search(r"\b(?:what(?:'s| is| are| was)|explain|describe|define)\s+([^?.!]+)", text, re.I)
    if topic:
        subject = re.sub(r"^(?:an?|the)\s+", "", topic.group(1).strip(), flags=re.I)
        return f"{subject[:1].upper()}{subject[1:]} is well documented: it was first described in 1987 and is used all over the world today."
    if "?" in text:
        return "Yes, and that is well known."
    return "Sure! I can help with that."


def _imitate(pairs, text):
    """Few-shot prompting: continue the pattern the examples set."""
    transforms = [
        str.upper, str.lower, str.title,
        lambda s: " ".join(reversed(s.split())),
        lambda s: s[::-1],
    ]
    for transform in transforms:
        if all(transform(i) == o for i, o in pairs):
            return transform(text)
    outputs = []
    for _, output in pairs:
        if output not in outputs:
            outputs.append(output)
    scores = _label_scores(text, outputs, "")
    for example_input, output in pairs:
        scores[output] += 2 * len(set(content_words(example_input)) & set(content_words(text)))
    return max(outputs, key=lambda o: scores[o])


def _answer_from_observations(observations):
    parts = [f"I used {m.name or 'a tool'} and got: {_text(m)}." for m in observations]
    return " ".join(parts)


# ------------------------------------------------------------------ tool calls

_POSITIVE = {"on", "true", "yes", "enable", "enabled", "start", "open", "activate", "turn on", "switch on"}
_NEGATIVE = {"off", "false", "no", "disable", "disabled", "stop", "close", "deactivate", "turn off", "switch off"}
# Everyday phrasings that point at a tool whose name uses a different word.
_SYNONYMS = {
    "how much": "price cost", "cost": "price", "costs": "price", "price": "cost",
    "cold": "temperature heater weather", "hot": "temperature weather", "warm": "temperature heater",
    "freezing": "temperature heater", "rain": "weather", "sunny": "weather",
    "plus": "add calculate math", "times": "multiply calculate math", "+": "add calculate math",
    "*": "multiply calculate math", "minus": "subtract calculate math", "/": "divide calculate math",
    "search": "find lookup", "define": "definition dictionary", "meaning": "definition dictionary",
}
_QUESTION = {"what", "whats", "much", "many", "cost", "costs", "tell", "please", "find", "look", "lookup", "check", "today"}


def _choose_tool(tools, current, after):
    used = {call["name"] for m in after if isinstance(m, AIMessage) for call in m.tool_calls}
    task = set(content_words(current))
    low = current.lower()
    for phrase, related in _SYNONYMS.items():
        if (phrase in low) if not phrase.isalpha() or " " in phrase else phrase in words(low):
            task |= set(content_words(related))
    best, best_score = None, 0
    for tool in tools:
        if tool["name"] in used:
            continue
        name_words = set(content_words(tool["name"].replace("_", " ")))
        desc_words = set(content_words(tool.get("description", "")))
        score = 2 * len(name_words & task) + len(desc_words & task)
        if score > best_score:
            best, best_score = tool, score
    # A second tool has to match clearly, or the sim would call everything once.
    if not best or best_score < (2 if used else 1):
        return None
    return {
        "name": best["name"],
        "args": _tool_args(best, current),
        "id": f"sim_call_{zlib.crc32((best['name'] + current).encode()) % 10**8}",
        "type": "tool_call",
    }


def _tool_args(tool, current):
    schema = tool.get("input_schema", {})
    required = set(schema.get("required", []))
    tool_words = set(content_words(tool["name"].replace("_", " ") + " " + tool.get("description", "")))
    found_numbers = numbers(current)
    quoted = [a or b or c for a, b, c in re.findall(r'"([^"]+)"|“([^”]+)”|(?<!\w)\'([^\']+)\'(?!\w)', current)]
    candidates = quoted or _candidate_words(current, tool_words)
    args = {}
    for name, prop in schema.get("properties", {}).items():
        kind = prop.get("type")
        if "enum" in prop:
            low = current.lower()
            args[name] = next((e for e in prop["enum"] if str(e).lower() in low), prop["enum"][0])
        elif kind in ("integer", "number"):
            if found_numbers:
                value = found_numbers.pop(0)
                args[name] = int(value) if kind == "integer" else float(value)
            elif name in required:
                args[name] = 0
        elif kind == "boolean":
            args[name] = _bool_arg(current, prop.get("default", False))
        elif kind == "array":
            args[name] = list(candidates)
        elif candidates:
            args[name] = candidates.pop(0)
        elif name in required:
            args[name] = current
    return args


def _candidate_words(text, exclude):
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    capitalised, plain = [], []
    for position, token in enumerate(tokens):
        low = token.lower().replace("'s", "")
        if low in STOPWORDS or low in _QUESTION or stem(low) in exclude or low in WORD_NUMBERS:
            continue
        if token[0].isupper() and position > 0:
            capitalised.append(token)
        else:
            plain.append(low)
    return capitalised + plain


def _bool_arg(text, default):
    low = " " + " ".join(words(text)) + " "
    positive = any(f" {w} " in low for w in _POSITIVE)
    negative = any(f" {w} " in low for w in _NEGATIVE)
    if positive and not negative:
        return True
    if negative and not positive:
        return False
    return bool(default)


# --------------------------------------------------------- data (ask_json etc.)


def fill(schema, text, missing_blank=False):
    """Fill a JSON schema from text with keyword rules."""
    result = {}
    for name, prop in schema.get("properties", {}).items():
        result[name] = _fill_field(name, prop, text, missing_blank)
    return result


def _fill_field(name, prop, text, missing_blank):
    kind = prop.get("type")
    if "enum" in prop:
        return classify_label(text, prop["enum"], "")
    if kind in ("integer", "number"):
        value = _number_for(name, text)
        return int(value) if kind == "integer" else float(value)
    if kind == "boolean":
        return _bool_for(name, text)
    if kind == "object":
        return fill(prop, text, missing_blank)
    if kind == "array":
        items = prop.get("items", {})
        if items.get("type") == "object":
            return _quantity_objects(items, text)
        if items.get("type") in ("integer", "number"):
            return numbers(text)
        return _list_for(name, text)
    return _string_for(name, text, prop.get("description", ""))


def _name_words(name):
    return [w for w in re.split(r"[_\s]+", name.lower()) if w and w not in ("is", "has", "the")]


def _number_for(name, text):
    low = text.lower()
    number = r"(-?\d+(?:\.\d+)?|" + "|".join(k for k in WORD_NUMBERS if k not in ("a", "an")) + r")"
    for word in _name_words(name):
        for pattern in (
            rf"\b{word}s?\s*(?:#|no\.?|number|is|was|of|:|=)?\s*{number}\b",
            rf"\b{number}\s+{word}s?\b",
        ):
            match = re.search(pattern, low)
            if match:
                return numbers(match.group(1))[0]
        if word == "age":
            match = re.search(rf"\b{number}\s*(?:years?|yrs?)(?:\s+old)?", low)
            if match:
                return numbers(match.group(1))[0]
    found = numbers(text)
    return found[0] if len(found) == 1 else 0


def _bool_for(name, text):
    tokens = words(text)
    stems = [stem(t) for t in tokens]
    for word in _name_words(name):
        if stem(word) in stems:
            i = stems.index(stem(word))
            before = tokens[max(0, i - 3) : i]
            return not any(w in ("not", "no", "never", "without") or w.endswith("n't") for w in before)
    return False


_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
    "date": r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?)",
    "url": r"https?://\S+",
    "time": r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
}


def _string_for(name, text, description):
    low_name = name.lower()
    for key, pattern in _PATTERNS.items():
        if key in low_name:
            match = re.search(pattern, text, re.I)
            return match.group(0) if match else ""
    if "name" in low_name:
        for pattern in (
            r"\b(Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+([A-Z][a-z]+)",
            r"\b(?:name is|i'm|i am|called|named)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
            r"\b([A-Z][a-z]+)\s+is\b",
        ):
            match = re.search(pattern, text)
            if match:
                return " ".join(g for g in match.groups() if g)
    labelled = re.search(rf"\b{re.escape(name.replace('_', ' '))}\s*(?:is|was|:|=)\s*([^,.;\n]+)", text, re.I)
    if labelled:
        return labelled.group(1).strip()
    if any(w in low_name for w in ("city", "location", "place", "country", "town")):
        match = re.search(r"\b(?:in|from|to|at|near)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text)
        if match:
            return match.group(1)
    if any(w in low_name for w in ("summary", "topic", "title", "description", "main")):
        first = sentences(text)
        return first[0] if first else ""
    return ""


_QUANTITY = r"\b(\d+|" + "|".join(WORD_NUMBERS) + r")\s+((?:[a-z]+ )?[a-z]+?)(?=[\s,.;!?]|$)"


def _quantity_pairs(text):
    pairs = []
    for count, noun in re.findall(_QUANTITY, text.lower()):
        noun = " ".join(w for w in noun.split() if w not in STOPWORDS)
        if noun and noun not in WORD_NUMBERS:
            pairs.append((numbers(count)[0] if numbers(count) else 1, noun))
    return pairs


def _quantity_objects(items, text):
    props = items.get("properties", {})
    number_field = next((n for n, p in props.items() if p.get("type") in ("integer", "number")), None)
    text_field = next((n for n, p in props.items() if p.get("type") == "string"), None)
    if not number_field:
        return []
    result = []
    for count, noun in _quantity_pairs(text):
        entry = {n: ("" if p.get("type") == "string" else 0) for n, p in props.items()}
        if number_field:
            entry[number_field] = count
        if text_field:
            entry[text_field] = noun
        result.append(entry)
    return result


def _list_for(name, text):
    labelled = re.search(rf"\b{re.escape(name.replace('_', ' '))}\s*(?:are|is|:)\s*([^.\n]+)", text, re.I)
    if labelled:
        return [p.strip() for p in re.split(r",|\band\b", labelled.group(1)) if p.strip()]
    pairs = _quantity_pairs(text)
    if pairs:
        return [noun for _, noun in pairs]
    if any(w in name.lower() for w in ("name", "people", "person")):
        return re.findall(r"(?<![.!?]\s)\b[A-Z][a-z]+\b", text)
    seen = []
    for word in content_words(text):
        if word not in seen and not word.isdigit():
            seen.append(word)
    return seen[:3]


# ------------------------------------------------------------ classification

_HINTS = {
    "refund": "refund money back return reimburse charged twice overcharged",
    "question": "? how what when where why which could wonder",
    "complaint": "angry terrible awful worst broken disappointed complain unacceptable",
    "praise": "love great excellent awesome thanks thank amazing wonderful",
    "positive": "love great excellent awesome amazing wonderful happy good best enjoy fantastic",
    "negative": "hate bad terrible awful worst sad angry disappointed boring poor",
    "neutral": "okay fine average",
    "spam": "free winner prize click offer buy now limited congratulations",
    "urgent": "urgent asap immediately emergency now quickly",
    "billing": "bill invoice charge payment paid card",
    "technical": "error crash bug broken login password install",
    "bug": "error crash bug broken fails",
    "sales": "price buy purchase quote discount",
    "greeting": "hello hi hey morning",
    "cancel": "cancel unsubscribe stop quit",
    "shipping": "ship shipping delivery package arrive tracking",
    "yes": "yes yeah sure correct true",
    "no": "no nope not incorrect false",
    "safe": "",
    "unsafe": "kill hurt weapon bomb attack password",
}
_FALLBACK_LABELS = ("other", "neutral", "unknown", "none", "general", "misc")


def _label_scores(text, labels, instructions):
    text_words = set(content_words(text)) | set(words(text))
    scores = {}
    for label in labels:
        score = 3 * len(set(content_words(str(label))) & text_words)
        hint = _HINTS.get(str(label).lower(), "")
        score += 2 * len(set(content_words(hint)) & text_words)
        if "?" in hint and "?" in text:
            score += 2
        defined = re.search(
            rf"\b{re.escape(str(label))}\s*(?:=|:|means|is|-)\s*([^,;\n.]+)", instructions, re.I
        )
        if defined:
            score += 2 * len(set(content_words(defined.group(1))) & text_words)
        scores[label] = score
    return scores


def classify_label(text, labels, instructions):
    labels = list(labels)
    scores = _label_scores(text, labels, instructions)
    best = max(labels, key=lambda label: scores[label])
    if scores[best] > 0:
        return best
    for label in labels:
        if str(label).lower() in _FALLBACK_LABELS:
            return label
    return labels[0]


def _classify_task(task, schema):
    return {"label": classify_label(task["text"], task["labels"], task.get("instructions", ""))}


# --------------------------------------------------------------------- judge

_DETECTORS = {
    "personal information": [
        ("an email address", r"[\w.+-]+@[\w-]+\.[\w.]+"),
        ("a phone number", r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"),
        ("an ID number", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("a street address", r"\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b"),
        ("a home address", r"\bmy (?:home )?address\b"),
    ],
    "passwords or secrets": [("a password", r"\b(?:password|passcode|pin code|api key|secret key)\b")],
    "rude or offensive language": [
        ("an insult", r"\b(?:stupid|idiot|dumb|moron|loser|shut up|hate you|crap|ugly)\b"),
    ],
    "violence or harm": [("violent language", r"\b(?:kill|hurt|weapon|gun|bomb|attack|punch|stab)\w*\b")],
    "prompt injection": [
        ("an attempt to override instructions",
         r"\b(?:ignore (?:all |any |the |your )?(?:previous |prior )?instructions|system prompt|pretend (?:you are|to be)|jailbreak)\b"),
    ],
    "medical advice": [("medical advice", r"\b(?:dose|dosage|diagnos\w*|medication|prescription|milligrams?|mg)\b")],
    "academic dishonesty": [("a request to cheat", r"\b(?:do my homework|write my essay|answers to the (?:test|quiz|exam))\b")],
}
_CONCEPTS = [
    (("personal", "private", "privacy", "pii", "contact", "address", "phone", "email"), "personal information"),
    (("password", "secret", "credential", "key"), "passwords or secrets"),
    (("profan", "rude", "insult", "offensive", "mean", "bully", "swear", "language", "toxic", "polite"), "rude or offensive language"),
    (("violen", "harm", "danger", "threat", "weapon", "unsafe"), "violence or harm"),
    (("injection", "instruction", "jailbreak", "override"), "prompt injection"),
    (("medical", "health", "doctor", "medicine"), "medical advice"),
    (("cheat", "homework", "academic", "plagiar"), "academic dishonesty"),
]


def _judge_task(task, schema):
    text = task["text"]
    checklist = task["checklist"]
    items = [checklist] if isinstance(checklist, str) else list(checklist)
    concepts = []
    for item in items:
        low = str(item).lower()
        matched = [concept for keys, concept in _CONCEPTS if any(k in low for k in keys)]
        concepts.extend(matched)
        if not matched:  # an item the rules don't know: look for its own words
            item_words = set(content_words(item)) - {"mention", "contain", "include", "text", "message"}
            hit = item_words & set(content_words(text))
            if hit:
                return {"problem": True, "reason": f"The text mentions {', '.join(sorted(hit))}, which the checklist rules out."}
    for concept in concepts or list(_DETECTORS):
        for what, pattern in _DETECTORS[concept]:
            if re.search(pattern, text, re.I):
                return {"problem": True, "reason": f"The text contains {what} ({concept})."}
    return {"problem": False, "reason": "No problems from the checklist were found."}


def _extract_task(task, schema):
    return fill(schema, task["text"], missing_blank=True)


_TASKS = {"classify": _classify_task, "judge": _judge_task, "extract": _extract_task}
