"""Offline tests for the HTTP clients, streaming and the CLI, using a local fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from harness import Agent, AgentConfig, Hooks, Message, ToolCall, ToolRegistry
from harness.llm import AnthropicClient, LLMError, OpenAICompatClient

TOOLS = [{"type": "function", "function": {"name": "add", "description": "Add numbers.",
                                          "parameters": {"type": "object", "properties": {"a": {"type": "integer"}},
                                                         "required": ["a"]}}}]


class Fake:
    """Serves queued replies. Each reply is (status, body, headers); body is a dict, str or list of SSE lines."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        fake = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                fake.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(raw)})
                status, body, headers = fake.replies.pop(0)
                if isinstance(body, list):
                    payload, ctype = "".join(f"data: {b}\n\n" for b in body).encode(), "text/event-stream"
                else:
                    payload, ctype = json.dumps(body).encode(), "application/json"
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    made = []

    def make(*replies):
        f = Fake(replies)
        made.append(f)
        return f
    yield make
    for f in made:
        f.close()


def oa_reply(content="", tool_calls=None, usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return (200, {"choices": [{"message": msg}], "usage": usage or {}}, None)


def oa_call(args, id="c1", name="add"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


# ---- OpenAICompatClient ----

def test_openai_success(fake):
    f = fake(oa_reply("hello", usage={"total_tokens": 7}))
    r = OpenAICompatClient("m", f.url, api_key="k").chat([Message("user", "hi")], TOOLS, 0.3)
    assert r.content == "hello" and r.usage == {"total_tokens": 7}
    req = f.requests[0]
    assert req["headers"]["user-agent"].startswith("universal-agent-harness/")
    assert req["path"] == "/chat/completions" and req["headers"]["authorization"] == "Bearer k"
    assert req["body"]["model"] == "m" and req["body"]["tools"] == TOOLS and req["body"]["temperature"] == 0.3


def test_openai_tool_calls_parsed_and_sent(fake):
    f = fake(oa_reply("", [oa_call('{"a": 2}')]), oa_reply("done"))
    c = OpenAICompatClient("m", f.url)
    r = c.chat([Message("user", "go")], TOOLS)
    assert r.tool_calls == [ToolCall("c1", "add", {"a": 2})]
    c.chat([Message("user", "go"), Message("assistant", "", r.tool_calls),
            Message("tool", "4", tool_call_id="c1", name="add")], TOOLS)
    sent = f.requests[1]["body"]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 2}'
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "4"}


def test_openai_malformed_arguments_become_raw(fake):
    f = fake(oa_reply("", [oa_call("{not json", "c1"), oa_call("[1]", "c2")]))
    r = OpenAICompatClient("m", f.url).chat([Message("user", "go")], TOOLS)
    assert r.tool_calls[0].arguments == {"__raw__": "{not json"}
    assert r.tool_calls[1].arguments == {"__raw__": "[1]"}


def test_openai_retries_429_with_retry_after(fake, monkeypatch):
    sleeps = []
    monkeypatch.setattr("harness.llm._http.time.sleep", sleeps.append)
    f = fake((429, {"error": "slow down"}, {"Retry-After": "2"}), oa_reply("ok"))
    assert OpenAICompatClient("m", f.url).chat([Message("user", "hi")]).content == "ok"
    assert sleeps == [2.0] and len(f.requests) == 2


def test_openai_400_raises_without_retry(fake):
    f = fake((400, {"error": "bad request"}, None), oa_reply("never"))
    with pytest.raises(LLMError, match="HTTP 400"):
        OpenAICompatClient("m", f.url).chat([Message("user", "hi")])
    assert len(f.requests) == 1


def test_gives_up_after_retries(fake, monkeypatch):
    monkeypatch.setattr("harness.llm._http.time.sleep", lambda s: None)
    f = fake(*[(503, {"error": "down"}, None)] * 3)
    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        OpenAICompatClient("m", f.url, max_retries=2).chat([Message("user", "hi")])


# ---- AnthropicClient ----

def an_reply(blocks, usage=None):
    return (200, {"type": "message", "role": "assistant", "content": blocks,
                  "usage": usage or {"input_tokens": 3, "output_tokens": 4}}, None)


def test_anthropic_request_and_response_mapping(fake):
    f = fake(an_reply([{"type": "text", "text": "Let me add."},
                       {"type": "tool_use", "id": "tu_1", "name": "add", "input": {"a": 2}}]))
    c = AnthropicClient("claude-x", f.url, api_key="secret", max_tokens=100)
    history = [Message("system", "be brief"), Message("user", "add 2"),
               Message("assistant", "thinking", [ToolCall("tu_0", "add", {"a": 1}), ToolCall("tu_00", "add", {"a": 5})]),
               Message("tool", "1", tool_call_id="tu_0", name="add"),
               Message("tool", "5", tool_call_id="tu_00", name="add")]
    r = c.chat(history, TOOLS, 0.1)
    assert r.content == "Let me add." and r.tool_calls == [ToolCall("tu_1", "add", {"a": 2})]
    assert r.usage == {"input_tokens": 3, "output_tokens": 4}

    req = f.requests[0]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["x-api-key"] == "secret" and req["headers"]["anthropic-version"] == "2023-06-01"
    body = req["body"]
    assert body["system"] == "be brief" and body["max_tokens"] == 100 and body["model"] == "claude-x"
    assert body["tools"] == [{"name": "add", "description": "Add numbers.",
                              "input_schema": TOOLS[0]["function"]["parameters"]}]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "thinking"},
        {"type": "tool_use", "id": "tu_0", "name": "add", "input": {"a": 1}},
        {"type": "tool_use", "id": "tu_00", "name": "add", "input": {"a": 5}}]
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_0", "content": "1"},
        {"type": "tool_result", "tool_use_id": "tu_00", "content": "5"}]


def test_anthropic_retry_and_errors(fake, monkeypatch):
    monkeypatch.setattr("harness.llm._http.time.sleep", lambda s: None)
    f = fake((500, {"error": "x"}, None), an_reply([{"type": "text", "text": "ok"}]),
             (400, {"error": "nope"}, None))
    c = AnthropicClient("m", f.url, api_key="k")
    assert c.chat([Message("user", "hi")]).content == "ok"
    with pytest.raises(LLMError, match="HTTP 400"):
        c.chat([Message("user", "hi")])


def test_anthropic_works_in_agent_loop(fake, tmp_path):
    from harness import tool

    @tool("add")
    def add(a: int) -> str:
        """Add."""
        return str(a + 1)
    reg = ToolRegistry()
    reg.register(add)
    f = fake(an_reply([{"type": "tool_use", "id": "tu_1", "name": "add", "input": {"a": 2}}]),
             an_reply([{"type": "text", "text": "answer 3"}]))
    agent = Agent(AgentConfig("t", "sys", tools=["add"]), AnthropicClient("m", f.url, api_key="k"), reg,
                  root=tmp_path)
    r = agent.run("add")
    assert r.final == "answer 3" and r.tool_calls == 1
    assert f.requests[1]["body"]["messages"][-1]["content"][0] == {
        "type": "tool_result", "tool_use_id": "tu_1", "content": "3"}


# ---- streaming ----

def chunk(delta, usage=None):
    d = {"choices": [{"delta": delta}]}
    if usage:
        d["usage"] = usage
    return json.dumps(d)


def sse_text():
    return [chunk({"role": "assistant"}), chunk({"content": "Hel"}), chunk({"content": "lo"}),
            json.dumps({"choices": [], "usage": {"total_tokens": 5}}), "[DONE]"]


def test_stream_yields_deltas_and_returns_response(fake):
    f = fake((200, sse_text(), None))
    gen = OpenAICompatClient("m", f.url).stream([Message("user", "hi")])
    deltas = []
    while True:
        try:
            deltas.append(next(gen))
        except StopIteration as s:
            final = s.value
            break
    assert deltas == ["Hel", "lo"] and final.content == "Hello" and final.usage == {"total_tokens": 5}
    assert f.requests[0]["body"]["stream"] is True


def test_stream_assembles_tool_calls(fake):
    f = fake((200, [chunk({"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "add", "arguments": ""}}]}),
                    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}),
                    chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' 4}'}}]}),
                    "[DONE]"], None))
    gen = OpenAICompatClient("m", f.url).stream([Message("user", "hi")], TOOLS)
    with pytest.raises(StopIteration) as s:
        next(gen)
    assert s.value.value.tool_calls == [ToolCall("c9", "add", {"a": 4})]


def test_agent_emits_token_events_and_falls_back(fake, tmp_path):
    f = fake((200, sse_text(), None))
    hooks, tokens = Hooks(), []
    hooks.on("token", lambda e, d: tokens.append(d["text"]))
    agent = Agent(AgentConfig("t", "sys"), OpenAICompatClient("m", f.url), ToolRegistry(), root=tmp_path,
                  hooks=hooks, stream=True)
    r = agent.run("hi")
    assert tokens == ["Hel", "lo"] and r.final == "Hello" and r.usage == {"total_tokens": 5}

    from harness.llm import ScriptedLLM
    from harness import LLMResponse
    hooks2, tokens2 = Hooks(), []
    hooks2.on("token", lambda e, d: tokens2.append(d))
    agent2 = Agent(AgentConfig("t", "sys"), ScriptedLLM([LLMResponse("plain")]), ToolRegistry(), root=tmp_path / "b",
                   hooks=hooks2, stream=True)
    assert agent2.run("hi").final == "plain" and [item["text"] for item in tokens2] == ["plain"]


def test_cli_stream_and_provider(fake, tmp_path, capsys):
    from harness.cli import main
    f = fake((200, sse_text(), None))
    code = main(["--model", "m", "--base-url", f.url, "--stream", "--root", str(tmp_path), "-p", "hi"])
    assert code == 0 and capsys.readouterr().out == "Hello\n"

    f2 = fake(an_reply([{"type": "text", "text": "from claude"}]))
    code = main(["--provider", "anthropic", "--model", "m", "--base-url", f2.url, "--root", str(tmp_path),
                 "-p", "hi"])
    assert code == 0 and capsys.readouterr().out == "from claude\n"


# ---- smoke script ----

def _load_smoke():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("smoke", Path(__file__).parent.parent / "scripts" / "smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_skips_without_model(monkeypatch, capsys):
    monkeypatch.delenv("HARNESS_MODEL", raising=False)
    assert _load_smoke().main([]) == 0
    assert "SKIP" in capsys.readouterr().out


def test_smoke_pass_and_fail(fake, monkeypatch, capsys):
    f = fake(oa_reply("", [oa_call(json.dumps({"operation": "solve", "expression": "x^2+5*x+6=0"}), name="sympy_math")]),
            oa_reply("The roots are -3 and -2."),
            oa_reply("", [oa_call(json.dumps({"text": "Experiment A uses 5g NaCl"}), name="add_note")]),
            oa_reply("Saved."),
            oa_reply("", [oa_call(json.dumps({"query": "Experiment A"}), name="search_notes")]),
            oa_reply("Experiment A uses 5g NaCl."))
    monkeypatch.setenv("HARNESS_MODEL", "m")
    monkeypatch.setenv("HARNESS_BASE_URL", f.url)
    assert _load_smoke().main([]) == 0
    out = capsys.readouterr().out
    assert "PASS math" in out and "PASS lab" in out

    f2 = fake(*[oa_reply("I did not use tools. -3 -2 NaCl")] * 4)  # no tool calls: both must FAIL
    monkeypatch.setenv("HARNESS_BASE_URL", f2.url)
    assert _load_smoke().main([]) == 1
    assert "FAIL math" in capsys.readouterr().out


def test_smoke_flags_and_runs(fake, capsys):
    f = fake(*[oa_reply("", [oa_call(json.dumps({"operation": "solve", "expression": "x^2+5*x+6=0"}), name="sympy_math")]),
               oa_reply("The roots are -3 and -2.")] * 2,
             *[oa_reply("", [oa_call(json.dumps({"text": "Experiment A uses 5g NaCl"}), name="add_note")]),
               oa_reply("Saved."),
               oa_reply("", [oa_call(json.dumps({"query": "Experiment A"}), name="search_notes")]),
               oa_reply("Experiment A uses 5g NaCl.")] * 2)
    assert _load_smoke().main(["--model", "m", "--base-url", f.url, "--runs", "2"]) == 0
    assert "pass rate: math 2/2, lab 2/2" in capsys.readouterr().out


# ---- fine-tuning data pipeline ----

def _load_script(name):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent.parent / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_data_keeps_only_clean_tool_runs(fake, tmp_path):
    collect = _load_script("collect_data")
    f = fake(oa_reply("", [oa_call(json.dumps({"operation": "solve", "expression": "x^2+5*x+6=0"}), name="sympy_math")]),
             oa_reply("The roots are -3 and -2."),
             oa_reply("I guessed: 4."),  # no tool call, must be dropped
             oa_reply("", [oa_call(json.dumps({"operation": "bogus", "expression": "x"}), name="sympy_math")]),
             oa_reply("done"))  # tool returned an error, must be dropped
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps({"agent": "math", "turns": [t]}) for t in ["a", "b", "c"]))
    out = tmp_path / "train.jsonl"
    assert collect.main(["--model", "m", "--base-url", f.url, "--prompts", str(prompts), "--out", str(out), "--workers", "1"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    roles = [m["role"] for m in rows[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert rows[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "sympy_math"
    assert rows[0]["tools"][0]["function"]["name"] == "sympy_math"


_FLAT_EX = {"tools": [{"type": "function", "function": {"name": "add", "description": "d", "parameters": {}}}],
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
                         {"role": "assistant", "content": "", "tool_calls": [
                             {"id": "1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": [2]}'}},
                             {"id": "2", "type": "function", "function": {"name": "add", "arguments": '{"a": 2}'}}]},
                         {"role": "tool", "tool_call_id": "1", "content": "2"},
                         {"role": "tool", "tool_call_id": "2", "content": "3"},
                         {"role": "assistant", "content": "answer"}]}


def test_qwen35_flatten_matches_hf_template():
    ft = _load_script("finetune_tinker")
    assert ft.default_tool_format("Qwen/Qwen3.5-4B") == "qwen35_xml"
    assert ft.default_tool_format("Qwen/Qwen3-8B") == "qwen3_json"
    out = ft.to_qwen_messages(_FLAT_EX, "qwen35_xml")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant"]
    assert out[0]["content"].startswith("# Tools\n\nYou have access to the following functions:\n\n<tools>\n{")
    assert out[0]["content"].endswith("</IMPORTANT>\n\nsys")
    assert out[2]["content"] == ("<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n"
                                 "<parameter=b>\n[2]\n</parameter>\n</function>\n</tool_call>\n"
                                 "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>")
    assert out[3]["content"] == "<tool_response>\n2\n</tool_response>\n<tool_response>\n3\n</tool_response>"


def test_qwen3_json_flatten():
    ft = _load_script("finetune_tinker")
    out = ft.to_qwen_messages(_FLAT_EX, "qwen3_json")
    assert "<tools>" in out[0]["content"] and '"name": "add"' in out[0]["content"]
    assert out[2]["content"].count("<tool_call>") == 2 and '"arguments": {"a": 1, "b": [2]}' in out[2]["content"]


def test_finetune_skips_without_key_and_ollama_modelfile(monkeypatch, capsys, tmp_path):
    ft = _load_script("finetune_tinker")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    assert ft.main(["--data", str(tmp_path / "none.jsonl")]) == 0
    assert "SKIP" in capsys.readouterr().out

    oi = _load_script("ollama_import")
    base = '# Modelfile generated by "ollama show"\nFROM /blobs/sha256-abc\nTEMPLATE """\n# Tools\n{{ .Prompt }}"""\nPARAMETER stop <|im_end|>\n'
    mf = oi.build_modelfile(base, tmp_path)
    assert mf.startswith(f"FROM {tmp_path.resolve()}\n") and "/blobs/" not in mf
    assert "# Tools" in mf and "PARAMETER stop <|im_end|>" in mf
