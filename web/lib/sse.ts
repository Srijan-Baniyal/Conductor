import type { HarnessEvent } from "./types";

export interface ParsedChunk {
  /** Complete, JSON-decoded frames found in the buffer. */
  events: unknown[];
  /** Trailing text that is not yet a complete frame. */
  rest: string;
}

const FRAME_BREAK = /\n\n/;
const LINE_BREAK = /\n/;
const CRLF = /\r\n?/g;

function decodeFrame(frame: string): unknown {
  const dataLines: string[] = [];
  for (const line of frame.split(LINE_BREAK)) {
    if (line.startsWith("data:")) {
      const value = line.slice(5);
      dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
    }
  }
  if (dataLines.length === 0) {
    return undefined;
  }
  try {
    return JSON.parse(dataLines.join("\n"));
  } catch {
    return undefined;
  }
}

/**
 * Pure SSE parser. Feed it the text received so far (previous `rest` plus the
 * new chunk). It returns every complete `data: {json}` frame and the unfinished
 * remainder, so frames split across network chunks are handled by the caller
 * simply passing `rest + nextChunk` back in.
 */
export function parseSseBuffer(buffer: string): ParsedChunk {
  const text = buffer.replace(CRLF, "\n");
  const parts = text.split(FRAME_BREAK);
  const rest = parts.pop() ?? "";
  const events: unknown[] = [];
  for (const part of parts) {
    const decoded = decodeFrame(part);
    if (decoded !== undefined) {
      events.push(decoded);
    }
  }
  return { events, rest };
}

export function isHarnessEvent(value: unknown): value is HarnessEvent {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { type?: unknown }).type === "string"
  );
}

/** Read a fetch() response body as a stream of harness events. */
export async function* readEvents(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<HarnessEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let rest = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      const parsed = parseSseBuffer(rest + decoder.decode(value, { stream: true }));
      rest = parsed.rest;
      for (const event of parsed.events) {
        if (isHarnessEvent(event)) {
          yield event;
        }
      }
    }
    // Flush a final frame the server did not terminate with a blank line.
    const tail = parseSseBuffer(`${rest}\n\n`);
    for (const event of tail.events) {
      if (isHarnessEvent(event)) {
        yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
