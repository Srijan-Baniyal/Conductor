import { describe, expect, it } from "vitest";
import { createRun, liveText, reduceRun, type RunState } from "./run";
import type { RunEvent } from "./types";

function play(events: RunEvent[]): RunState {
  let state = createRun({
    id: "r1",
    prompt: "plot",
    agent: "math",
    model: "mock",
    startedAt: 0,
  });
  for (const event of events) {
    state = reduceRun(state, event);
  }
  return state;
}

const session: RunEvent = {
  type: "session",
  session: "abc",
  agent: "math",
  model: "mock",
};

describe("reduceRun", () => {
  it("records the session id", () => {
    expect(play([session]).session).toBe("abc");
  });

  it("builds steps and tool calls through the full lifecycle", () => {
    const run = play([
      session,
      { type: "llm_request", step: 1, messages: 2, tools: 2 },
      {
        type: "llm_response",
        step: 1,
        ms: 12,
        tool_calls: ["plot"],
        calls: [{ id: "c1", name: "plot", arguments: { x: 1 } }],
        usage: { prompt_tokens: 5, completion_tokens: 3 },
        text: "",
      },
      { type: "tool_start", t: 0.1, id: "c1", name: "plot", arguments: { x: 1 } },
    ]);
    expect(run.steps).toHaveLength(1);
    expect(run.steps[0].messages).toBe(2);
    expect(run.steps[0].calls[0].status).toBe("running");
    expect(run.steps[0].calls[0].startedAt).toBe(0.1);

    const ended = reduceRun(run, {
      type: "tool_end",
      id: "c1",
      name: "plot",
      ok: true,
      ms: 40,
      output: "Saved",
      files: ["/files/abc/p.png"],
    });
    const call = ended.steps[0].calls[0];
    expect(call.status).toBe("ok");
    expect(call.ms).toBe(40);
    expect(call.files).toEqual(["/files/abc/p.png"]);
  });

  it("matches parallel tool calls by id, not order", () => {
    const run = play([
      session,
      { type: "llm_request", step: 1 },
      {
        type: "llm_response",
        step: 1,
        calls: [
          { id: "a", name: "t1", arguments: {} },
          { id: "b", name: "t2", arguments: {} },
        ],
      },
      { type: "tool_start", id: "a", name: "t1", arguments: {} },
      { type: "tool_start", id: "b", name: "t2", arguments: {} },
      { type: "tool_end", id: "b", name: "t2", ok: false, ms: 5, output: "boom" },
    ]);
    const [a, b] = run.steps[0].calls;
    expect(a.status).toBe("running");
    expect(b.status).toBe("error");
    expect(b.output).toBe("boom");
  });

  it("accumulates streamed tokens and exposes them as live text", () => {
    const run = play([
      session,
      { type: "llm_request", step: 1 },
      { type: "token", step: 1, text: "Hel" },
      { type: "token", step: 1, text: "lo" },
    ]);
    expect(liveText(run)).toBe("Hello");
    // Consecutive tokens are merged in the raw event log.
    expect(run.events.filter((e) => e.type === "token")).toHaveLength(1);

    const responded = reduceRun(run, {
      type: "llm_response",
      step: 1,
      text: "Hello",
    });
    expect(liveText(responded)).toBe("");
    expect(responded.steps[0].text).toBe("Hello");
  });

  it("finishes with totals and the final answer on done", () => {
    const run = play([
      session,
      { type: "llm_request", step: 1 },
      { type: "llm_response", step: 1, text: "42" },
      { type: "final", stopped: "done", steps: 1, tool_calls: 0, usage: {} },
      {
        type: "done",
        stopped: "done",
        steps: 1,
        tool_calls: 0,
        usage: { total_tokens: 9 },
        final: "42",
        ms: 100,
      },
    ]);
    expect(run.status).toBe("done");
    expect(run.final).toBe("42");
    expect(run.totals).toEqual({
      steps: 1,
      toolCalls: 0,
      usage: { total_tokens: 9 },
      ms: 100,
    });
  });

  it("marks the run as errored when the server stops with error", () => {
    const run = play([
      session,
      { type: "error", where: "llm", error: "bad key" },
      { type: "done", stopped: "error", steps: 1, tool_calls: 0, final: "" },
    ]);
    expect(run.status).toBe("error");
    expect(run.errors).toEqual([{ where: "llm", error: "bad key" }]);
  });

  it("cancels open tool calls on abort", () => {
    const run = play([
      session,
      { type: "llm_request", step: 1 },
      {
        type: "llm_response",
        step: 1,
        calls: [{ id: "c", name: "plot", arguments: {} }],
      },
      { type: "tool_start", id: "c", name: "plot", arguments: {} },
      { type: "client_abort" },
    ]);
    expect(run.status).toBe("aborted");
    expect(run.steps[0].calls[0].status).toBe("cancelled");
  });

  it("infers a plot image from the output when files is empty", () => {
    const run = play([
      session,
      { type: "llm_request", step: 1 },
      { type: "tool_start", id: "c", name: "plot", arguments: {} },
      {
        type: "tool_end",
        id: "c",
        name: "plot",
        ok: true,
        output: "Saved plot to .harness/workspace/abc/plot_1.png",
        files: [],
      },
    ]);
    expect(run.steps[0].calls[0].files).toEqual(["/files/abc/plot_1.png"]);
  });

  it("does not mutate the previous state", () => {
    const before = play([session, { type: "llm_request", step: 1 }]);
    const snapshot = JSON.stringify(before);
    reduceRun(before, { type: "token", step: 1, text: "x" });
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});
