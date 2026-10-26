"""The LangGraph ReAct research agent, surfaced one step at a time.

`iter_steps` drives the prebuilt ReAct graph with `stream(stream_mode="updates")`
and yields one `AgentStep` per LLM turn, carrying the tool results that turn
produced. Yielding per step is what lets the caller score and, if needed, break
the loop between steps rather than only after the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

import config

try:  # LangGraph 1.x relocated the prebuilt ReAct graph into langchain.agents.
    from langchain.agents import create_agent as _create_react_graph

    _PROMPT_KWARG = "system_prompt"
except ImportError:  # pragma: no cover - LangGraph 0.x
    from langgraph.prebuilt import create_react_agent as _create_react_graph

    _PROMPT_KWARG = "prompt"


@dataclass
class ToolCallRecord:
    name: str
    query: str
    call_id: str = ""
    result_preview: str = ""


@dataclass
class AgentStep:
    """One LLM turn: its visible text, its tool calls, and what they returned."""

    index: int
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prev_step_used_tools: bool = False
    prev_step_had_tool_results: bool = False
    is_first_step: bool = False
    any_results_so_far: bool = False
    input_tokens: int = 0        # uncached input only
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class _SerialToolChat(ChatAnthropic):
    """ChatAnthropic that refuses to batch tool calls.

    The graph builder calls `bind_tools` itself, so forcing the flag here is the
    only place it can be applied without reaching inside the prebuilt graph.
    """

    def bind_tools(self, tools, **kwargs):
        kwargs.setdefault("parallel_tool_calls", False)
        return super().bind_tools(tools, **kwargs)


def build_agent():
    """Wire up Claude + Tavily into a prebuilt ReAct graph."""
    # Thinking stays off: REASONING_FIRST is measured on *visible* prose, and
    # extended thinking would move exactly that reasoning out of the response.
    chat_cls = _SerialToolChat if config.SERIAL_TOOL_CALLS else ChatAnthropic
    model_kwargs = {}
    if config.PROMPT_CACHING:
        # Top-level auto-caching: the API places the breakpoint on the last
        # cacheable block, so the system prompt, the tool schema and the
        # transcript prefix are all re-read from cache on the next step.
        model_kwargs["cache_control"] = {"type": "ephemeral"}
    llm = chat_cls(
        model=config.AGENT_MODEL,
        max_tokens=config.AGENT_MAX_TOKENS,
        api_key=config.ANTHROPIC_API_KEY,
        model_kwargs=model_kwargs,
    )
    search = TavilySearch(
        max_results=config.TAVILY_MAX_RESULTS,
        tavily_api_key=config.TAVILY_API_KEY,
    )
    return _create_react_graph(
        llm,
        tools=[search],
        **{_PROMPT_KWARG: config.build_system_prompt()},
    )


def _text_of(message: AIMessage) -> str:
    """Flatten an AIMessage's content to the visible text it showed the user."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


# langchain-anthropic reports cache writes under a TTL-specific key and zeroes
# the generic one, so a write count has to sum all three.
_CACHE_WRITE_KEYS = (
    "cache_creation",
    "ephemeral_5m_input_tokens",
    "ephemeral_1h_input_tokens",
)


def _cache_read(usage: dict) -> int:
    details = usage.get("input_token_details") or {}
    return int(details.get("cache_read", 0) or 0)


def _cache_write(usage: dict) -> int:
    details = usage.get("input_token_details") or {}
    return sum(int(details.get(k, 0) or 0) for k in _CACHE_WRITE_KEYS)


def _uncached_input(usage: dict) -> int:
    """Input tokens billed at the full rate.

    LangChain's `input_tokens` is the *total* - it adds cache reads and writes
    back onto Anthropic's figure, which excludes them - so both must come off
    again or they get billed twice.
    """
    total = int(usage.get("input_tokens", 0) or 0)
    return max(0, total - _cache_read(usage) - _cache_write(usage))


def _describe_call(call: dict) -> ToolCallRecord:
    args = call.get("args") or {}
    query = args.get("query") or args.get("input") or str(args)
    return ToolCallRecord(
        name=call.get("name", "tool"),
        query=str(query),
        call_id=call.get("id") or "",
    )


def iter_steps(agent, task: str, max_steps: int) -> Iterator[AgentStep]:
    """Yield one AgentStep per LLM turn until the agent stops or hits max_steps.

    Closing this generator (via `break` in the caller) abandons the run, which is
    how the circuit breaker aborts mid-research.
    """
    stream = agent.stream(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": max_steps * 2 + 4},
        stream_mode="updates",
    )

    index = 0
    pending: AgentStep | None = None
    prev_used_tools = False
    prev_had_results = False
    any_results = False

    for update in stream:
        for _node, payload in update.items():
            messages = (payload or {}).get("messages") or []
            ai_messages = [m for m in messages if isinstance(m, AIMessage)]
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

            if ai_messages:
                # A new LLM turn: the previous step is now complete.
                if pending is not None:
                    yield pending
                    prev_used_tools = pending.used_tools
                    prev_had_results = bool(pending.tool_calls)
                    if pending.tool_calls:
                        any_results = True
                    pending = None

                ai = ai_messages[-1]
                usage = getattr(ai, "usage_metadata", None) or {}
                index += 1
                pending = AgentStep(
                    index=index,
                    text=_text_of(ai),
                    tool_calls=[_describe_call(c) for c in (ai.tool_calls or [])],
                    prev_step_used_tools=prev_used_tools,
                    prev_step_had_tool_results=prev_had_results,
                    is_first_step=index == 1,
                    any_results_so_far=any_results,
                    input_tokens=_uncached_input(usage),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=_cache_read(usage),
                    cache_write_tokens=_cache_write(usage),
                )
                if index >= max_steps:
                    yield pending
                    pending = None
                    stream.close()
                    return

            if tool_messages and pending is not None:
                # Attach results to the step whose calls produced them.
                by_id = {m.tool_call_id: m for m in tool_messages if m.tool_call_id}
                for record in pending.tool_calls:
                    msg = by_id.get(record.call_id)
                    if msg is not None:
                        record.result_preview = str(msg.content)[:800]

    if pending is not None:
        yield pending
