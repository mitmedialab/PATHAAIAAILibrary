"""ai.ask and ai.Chat: the two ways to talk to a model."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ._engine import complete


def ask(prompt, system="", temperature=None, max_tokens=300, examples=None):
    """Ask one question and get one answer back, as text.

    Nothing from an earlier call is sent along, so the model has no memory.

    Args:
        prompt: the question or instruction.
        system: standing instructions, the model's job description.
        temperature: 0 (repeatable) to 1 (adventurous). Haiku only.
        max_tokens: a ceiling on the length of the reply.
        examples: a list of (input, output) pairs for the model to imitate.
    """
    messages = []
    if system:
        messages.append(SystemMessage(system))
    for example_input, example_output in examples or []:
        messages.append(HumanMessage(str(example_input)))
        messages.append(AIMessage(str(example_output)))
    messages.append(HumanMessage(str(prompt)))
    reply = complete(messages, max_tokens=max_tokens, temperature=temperature,
                     examples=len(examples or []))
    return reply.text


def to_langchain(messages, system=""):
    """Turn a list of {"role", "content"} dicts into LangChain messages."""
    kinds = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    converted = [SystemMessage(system)] if system else []
    for message in messages:
        converted.append(kinds[message["role"]](message["content"]))
    return converted


class Chat:
    """A conversation that remembers.

    The memory is ``chat.messages``: a plain list of {"role", "content"}
    dictionaries that is sent again with every turn. There is no other memory.

        chat = ai.Chat(system="You are a friendly tutor.")
        chat.say("My name is Sam.")
        chat.say("What is my name?")
    """

    def __init__(self, system="", max_tokens=300):
        self.system = system
        self.max_tokens = max_tokens
        self.messages = []

    def say(self, text):
        """Send one message and get the reply, as text."""
        self.messages.append({"role": "user", "content": str(text)})
        try:
            reply = complete(to_langchain(self.messages, self.system), max_tokens=self.max_tokens)
        except Exception:
            self.messages.pop()  # the turn never happened
            raise
        self.messages.append({"role": "assistant", "content": reply.text})
        return reply.text

    @property
    def last(self):
        """The model's most recent reply ("" before the first one)."""
        for message in reversed(self.messages):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    def transcript(self):
        """The whole conversation as printable text."""
        lines = [f"system: {self.system}"] if self.system else []
        lines += [f"{m['role']}: {m['content']}" for m in self.messages]
        return "\n".join(lines)

    def reset(self):
        """Forget everything. The system prompt stays."""
        self.messages = []

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return f"<ai.Chat with {len(self)} messages>"
