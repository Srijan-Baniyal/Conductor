import { describe, expect, it } from "vitest";
import { parseSseBuffer, readEvents } from "./sse";

describe("parseSseBuffer", () => {
  it("parses complete frames and keeps no remainder", () => {
    const { events, rest } = parseSseBuffer(
      'data: {"type":"session","session":"a"}\n\ndata: {"type":"done"}\n\n'
    );
    expect(events).toEqual([{ type: "session", session: "a" }, { type: "done" }]);
    expect(rest).toBe("");
  });

  it("returns an incomplete trailing frame as rest", () => {
    const first = parseSseBuffer('data: {"type":"token","text":"he');
    expect(first.events).toEqual([]);
    const second = parseSseBuffer(`${first.rest}llo"}\n\n`);
    expect(second.events).toEqual([{ type: "token", text: "hello" }]);
    expect(second.rest).toBe("");
  });

  it("handles a frame split exactly between the two newlines", () => {
    const first = parseSseBuffer('data: {"type":"a"}\n');
    expect(first.events).toEqual([]);
    const second = parseSseBuffer(`${first.rest}\ndata: {"type":"b"}\n\n`);
    expect(second.events).toEqual([{ type: "a" }, { type: "b" }]);
  });

  it("normalises CRLF line endings", () => {
    const { events } = parseSseBuffer('data: {"type":"x"}\r\n\r\n');
    expect(events).toEqual([{ type: "x" }]);
  });

  it("ignores comments, invalid JSON and frames without data", () => {
    const { events } = parseSseBuffer(
      ': keepalive\n\ndata: not json\n\nevent: ping\n\ndata: {"type":"ok"}\n\n'
    );
    expect(events).toEqual([{ type: "ok" }]);
  });

  it("joins multi-line data fields", () => {
    const { events } = parseSseBuffer('data: {"type":\ndata: "multi"}\n\n');
    expect(events).toEqual([{ type: "multi" }]);
  });
});

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

describe("readEvents", () => {
  it("reassembles frames split across arbitrary chunk boundaries", async () => {
    const payload = 'data: {"type":"token","text":"é"}\n\ndata: {"type":"done"}\n\n';
    const chunks: string[] = [];
    for (let i = 0; i < payload.length; i += 7) {
      chunks.push(payload.slice(i, i + 7));
    }
    const seen: string[] = [];
    for await (const event of readEvents(streamOf(chunks))) {
      seen.push(event.type);
    }
    expect(seen).toEqual(["token", "done"]);
  });

  it("flushes a last frame that has no terminating blank line", async () => {
    const seen: string[] = [];
    for await (const event of readEvents(streamOf(['data: {"type":"done"}']))) {
      seen.push(event.type);
    }
    expect(seen).toEqual(["done"]);
  });
});
