import json
import threading
import urllib.request

import pytest

from harness import ToolRegistry
from harness.server import App, ModelPreset, serve


@pytest.fixture
def api(tmp_path):
    reg = ToolRegistry()
    reg.load_builtin()
    from harness.cli import BUNDLED
    app = App(tmp_path, [BUNDLED], reg, [ModelPreset("mock", "Mock", "mock")])
    srv = serve(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r), r.headers


def chat(base, **body):
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        return [json.loads(line[6:]) for line in r.read().decode().splitlines() if line.startswith("data: ")]


def test_discovery_endpoints(api):
    base, _ = api
    agents, headers = get(base + "/api/agents")
    assert {"math", "lab", "general"} <= {a["name"] for a in agents}
    math = next(a for a in agents if a["name"] == "math")
    assert [t["name"] for t in math["tools"]] == ["sympy_math", "plot"] and math["tools"][0]["isolated"]
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    tools, _ = get(base + "/api/tools")
    assert "parameters" in next(t for t in tools if t["name"] == "plot")
    models, _ = get(base + "/api/models")
    assert models["default"] == "mock" and "api_key" not in json.dumps(models)


def test_chat_streams_tool_events_and_serves_plot(api):
    base, root = api
    events = chat(base, agent="math", message="plot sin(x) from -3 to 3", session="s1", model="mock")
    types = [e["type"] for e in events]
    assert types[0] == "session" and types[-1] == "done"
    assert types.index("tool_start") < types.index("tool_end") < types.index("llm_response", types.index("tool_end"))
    start = next(e for e in events if e["type"] == "tool_start")
    end = next(e for e in events if e["type"] == "tool_end")
    assert start["name"] == "plot" and start["id"] == end["id"] and end["ok"]
    assert end["files"] and end["files"][0].startswith("/files/s1/")
    png = urllib.request.urlopen(base + end["files"][0]).read()
    assert png[:4] == b"\x89PNG"
    hist, _ = get(base + "/api/sessions/s1")
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant", "tool", "assistant"]


def test_notes_and_errors(api):
    base, _ = api
    chat(base, agent="lab", message="remember: Experiment A uses 5g NaCl", session="s2", model="mock")
    notes, _ = get(base + "/api/notes?session=s2&agent=lab")
    assert notes and "NaCl" in notes[0]["text"]
    bad = chat(base, agent="nope", message="hi")
    assert bad[-1]["type"] == "error" and "unknown agent" in bad[-1]["error"]
    bad = chat(base, agent="math", message="hi", session="../etc")
    assert bad[-1]["type"] == "error"
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(base + "/files/s2/..%2f..%2fpasswd")
