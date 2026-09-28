"""ai.Agent: the think, act, observe loop."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_openai_messages
from langgraph.errors import GraphRecursionError

from ._engine import DevanModel
from .tools import Tool


class Agent:
    """A model given tools and run in a loop.

    Each round the model thinks, may ask for one of your tools, sees the result
    and thinks again, until it answers in plain text or ``max_steps`` is spent.

        agent = ai.Agent([price_lookup], system="You help students shop.")
        agent.run("How much is milk?")
        agent.steps   # the trace as data
    """

    def __init__(self, tools, system="", max_steps=6, verbose=True):
        self.tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        self.system = system
        self.max_steps = max_steps
        self.verbose = verbose
        self.steps = []
        self.answer = ""
        self._history = []
        self._graph = create_agent(
            DevanModel(),
            tools=[t.as_langchain() for t in self.tools],
            system_prompt=system or None,
        )

    def run(self, task):
        """Work on a task until the model answers. Returns the answer as text."""
        start = len(self._history)
        self._history.append(HumanMessage(str(task)))
        self.steps = []
        pending = {}
        # Each step is one model call plus the tool calls it asks for. Stopping
        # after a tool step keeps every tool call in the transcript answered.
        config = {"recursion_limit": 2 * self.max_steps}
        try:
            for update in self._graph.stream({"messages": list(self._history)}, config, stream_mode="updates"):
                for node_output in update.values():
                    for message in (node_output or {}).get("messages", []):
                        self._history.append(message)
                        self._record(message, pending)
        except GraphRecursionError:
            self._finish(f"I stopped after {self.max_steps} steps without reaching an answer.")
            self._history.append(AIMessage(self.answer))
        except Exception:
            del self._history[start:]  # the run never happened
            raise
        return self.answer

    def _record(self, message, pending):
        if isinstance(message, AIMessage) and message.tool_calls:
            thought = message.text.strip()
            if thought:
                self._print("Thought", thought)
            for call in message.tool_calls:
                step = {"thought": thought, "tool": call["name"], "input": call["args"], "observation": None}
                self.steps.append(step)
                pending[call["id"]] = step
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                self._print("Action", f"{call['name']}({args})")
        elif isinstance(message, ToolMessage):
            observation = message.text
            if message.tool_call_id in pending:
                pending[message.tool_call_id]["observation"] = observation
            self._print("Observation", observation)
        elif isinstance(message, AIMessage):
            self._finish(message.text)

    def _finish(self, answer):
        self.answer = answer
        self.steps.append({"answer": answer})
        self._print("Answer", answer)

    def _print(self, label, text):
        if self.verbose:
            print(f"{label}: {text}", flush=True)

    @property
    def messages(self):
        """The raw transcript as a list of dicts: the agent's working memory."""
        system = [{"role": "system", "content": self.system}] if self.system else []
        return system + convert_to_openai_messages(self._history)

    def reset(self):
        """Clear the transcript and the trace. Tools and system prompt stay."""
        self._history = []
        self.steps = []
        self.answer = ""

    def __repr__(self):
        return f"<ai.Agent with tools {[t.name for t in self.tools]}>"
