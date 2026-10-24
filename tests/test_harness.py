import json
import pytest
from harness import (Agent, AgentConfig, ConversationMemory, Message, ToolRegistry, ToolCall, LLMResponse, tool)
from harness.llm import ScriptedLLM, TextToolAdapter
from harness.tools.builtin import safe_math


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.load_builtin()
    return r


def call(name, **args):
    return LLMResponse("", [ToolCall("c1", name, args)])


def make(registry, script, tools=("sympy_math",), **cfg):
    llm = ScriptedLLM(script)
    agent = Agent(AgentConfig("t", "sys", tools=list(tools), **cfg), llm, registry, root="/tmp/uah_test",
                  session_id=f"s{id(script)}")
    agent.memory.reset()
    return agent, llm


def test_tool_loop_end_to_end(registry):
    agent, _ = make(registry, [call("sympy_math", operation="solve", expression="x^2+5*x+6=0"),
                               LLMResponse("done: -3 and -2")])
    r = agent.run("solve")
    assert r.stopped == "done" and r.tool_calls == 1
    tool_msg = [m for m in agent.memory.messages if m.role == "tool"][0]
    assert "-3" in tool_msg.content and "-2" in tool_msg.content


@pytest.mark.parametrize("bad", ["__import__('os').system('id')", "().__class__", "open('/etc/passwd')",
                                 "x.__class__", "9**9**9", "2**1000000", "x**2000", "[1,2]", "lambda: 1"])
def test_safe_math_rejects_attacks(bad):
    with pytest.raises(ValueError):
        safe_math.check(bad)


def test_unknown_tool_and_bad_args_recover(registry):
    agent, _ = make(registry, [call("nope"), call("sympy_math", operation="bogus", expression="x"),
                               LLMResponse("ok")])
    r = agent.run("go")
    outs = [m.content for m in agent.memory.messages if m.role == "tool"]
    assert "Unknown tool" in outs[0] and "must be one of" in outs[1] and r.stopped == "done"


def test_tool_exception_does_not_crash(registry):
    agent, _ = make(registry, [call("sympy_math", operation="evaluate", expression="x+1"), LLMResponse("ok")])
    agent.run("go")
    assert "free symbols" in [m.content for m in agent.memory.messages if m.role == "tool"][0]


def test_max_steps_forces_final_answer(registry):
    script = [call("sympy_math", operation="evaluate", expression=f"{i}+1") for i in range(3)]
    agent, llm = make(registry, script + [LLMResponse("best effort")], max_steps=3)
    r = agent.run("loop")
    assert r.stopped == "max_steps" and r.final == "best effort"
    assert llm.calls[-1][1] is None  # final call had no tools


def test_repeat_call_guard(registry):
    same = lambda *_: call("sympy_math", operation="evaluate", expression="1+1")
    agent, _ = make(registry, [same] * 5 + [LLMResponse("x")], max_steps=6)
    agent.run("go")
    assert any("exact call" in m.content for m in agent.memory.messages if m.role == "tool")


def test_timeout(registry):
    import time

    @tool("slow", timeout=0.2)
    def slow() -> str:
        time.sleep(2)
        return "x"
    registry.register(slow)
    agent, _ = make(registry, [call("slow"), LLMResponse("ok")], tools=("slow",))
    agent.run("go")
    assert "Timed out" in [m.content for m in agent.memory.messages if m.role == "tool"][0]


def test_approval_gate(registry):
    @tool("danger", requires_approval=True)
    def danger() -> str:
        return "boom"
    registry.register(danger)
    agent, _ = make(registry, [call("danger"), LLMResponse("ok")], tools=("danger",))
    agent.run("go")
    assert "denied" in [m.content for m in agent.memory.messages if m.role == "tool"][0]


def test_window_never_splits_tool_pairs():
    m = ConversationMemory()
    for i in range(6):
        m.add(Message("user", "q" * 400))
        m.add(Message("assistant", "", [ToolCall(f"c{i}", "t", {})]))
        m.add(Message("tool", "r" * 400, tool_call_id=f"c{i}", name="t"))
        m.add(Message("assistant", "a" * 400))
    kept, dropped = m.window(budget=700)
    assert kept[0].role == "user" and dropped and len(kept) % 4 == 0
    assert kept + [] == m.messages[len(dropped):]


def test_text_adapter_parses_and_recovers():
    inner = ScriptedLLM([LLMResponse('hi <tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'),
                         LLMResponse("<tool_call>{bad json</tool_call>")])
    ad = TextToolAdapter(inner)
    tools = [{"type": "function", "function": {"name": "a", "description": "d", "parameters": {}}}]
    r = ad.chat([Message("system", "s"), Message("user", "u")], tools)
    assert r.tool_calls[0].name == "a" and r.tool_calls[0].arguments == {"x": 1} and r.content == "hi"
    assert "Tools:" in inner.calls[0][0][0].content
    assert ad.chat([Message("user", "u")], tools).tool_calls[0].name == "invalid_tool_call"


def test_notes_roundtrip_and_search(registry):
    agent, _ = make(registry, [call("add_note", text="Experiment A uses 5g NaCl", tag="a"),
                               call("search_notes", query="NaCl"), LLMResponse("ok")],
                    tools=("add_note", "search_notes"))
    agent.ctx.notes._notes.clear()
    agent.run("go")
    assert "NaCl" in [m.content for m in agent.memory.messages if m.role == "tool"][-1]


def test_plot_and_symbol_names(registry, tmp_path):
    agent, _ = make(registry, [call("plot", kind="function", expression="y = sqrt(x)", x_min=-4, x_max=4),
                               LLMResponse("ok")], tools=("plot",))
    agent.ctx.workdir = tmp_path
    agent.run("go")
    out = [m.content for m in agent.memory.messages if m.role == "tool"][0]
    assert out.startswith("Saved plot") and list(tmp_path.glob("*.png"))


def test_plugin_dir_and_config(registry, tmp_path):
    registry.load_dir("examples/custom_tools")
    assert "unit_convert" in registry.names()
    from harness.config import load_agent_config
    f = tmp_path / "bad.toml"
    f.write_text('system_prompt="x"\nbogus=1')
    with pytest.raises(ValueError):
        load_agent_config(f)


@tool("hang_forever", timeout=1.0, isolated=True)
def hang_forever() -> str:
    """Never returns."""
    import time
    while True:
        time.sleep(1)


@tool("iso_echo", isolated=True)
def iso_echo(text: str, ctx=None) -> str:
    """Echo from a child process."""
    import os
    print("noise on stdout")
    return f"{text}:{os.getpid()}:{ctx.session_id}"


@tool("iso_fail", isolated=True)
def iso_fail() -> str:
    """Raises."""
    raise RuntimeError("bad")


def _ctx(tmp_path):
    from harness import ToolContext
    return ToolContext(tmp_path, "sess1")


def test_isolated_tool_is_killed_on_timeout(tmp_path):
    import time
    t0 = time.perf_counter()
    res = hang_forever.run({}, _ctx(tmp_path))
    assert not res.ok and "Timed out" in res.output
    assert time.perf_counter() - t0 < 10


def test_isolated_tool_runs_in_child(tmp_path):
    import os
    res = iso_echo.run({"text": "hi"}, _ctx(tmp_path))
    assert res.ok, res.output
    text, pid, sess = res.output.split(":")
    assert text == "hi" and sess == "sess1" and int(pid) != os.getpid()
    bad = iso_fail.run({}, _ctx(tmp_path))
    assert not bad.ok and "RuntimeError: bad" in bad.output


def test_builtin_math_and_plot_are_isolated(registry):
    assert registry.get("sympy_math").isolated and registry.get("plot").isolated
