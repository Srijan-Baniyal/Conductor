import type {
  CountOrList,
  RunEvent,
  StopReason,
  ToolCallRequest,
  ToolEndEvent,
  Usage,
} from "./types";

export type RunStatus = "running" | "done" | "error" | "aborted";
export type ToolStatus = "pending" | "running" | "ok" | "error" | "cancelled";
export type StepStatus = "waiting" | "streaming" | "responded";

export interface ToolCallState {
  id: string;
  name: string;
  arguments: unknown;
  status: ToolStatus;
  /** Seconds since the request started, when the tool began. */
  startedAt?: number;
  ms?: number;
  output?: string;
  files: string[];
}

export interface StepState {
  step: number;
  status: StepStatus;
  messages?: number;
  tools?: number;
  ms?: number;
  usage?: Usage;
  text: string;
  requested: string[];
  calls: ToolCallState[];
}

export interface RunError {
  where?: string;
  error: string;
}

export interface RunTotals {
  steps: number;
  toolCalls: number;
  usage: Usage;
  ms?: number;
}

export interface RunState {
  id: string;
  prompt: string;
  agent: string;
  model: string;
  startedAt: number;
  status: RunStatus;
  session?: string;
  steps: StepState[];
  final?: string;
  stopped?: StopReason;
  totals?: RunTotals;
  errors: RunError[];
  events: RunEvent[];
}

export function createRun(args: {
  id: string;
  prompt: string;
  agent: string;
  model: string;
  startedAt: number;
}): RunState {
  return { ...args, status: "running", steps: [], errors: [], events: [] };
}

const IMAGE_NAME = /([^\s/\\]+\.(?:png|jpe?g|gif|webp|svg))\b/i;

function count(value: CountOrList | undefined): number | undefined {
  if (typeof value === "number") {
    return value;
  }
  return Array.isArray(value) ? value.length : undefined;
}

function emptyStep(step: number): StepState {
  return { step, status: "waiting", text: "", requested: [], calls: [] };
}

function newCall(call: ToolCallRequest): ToolCallState {
  return {
    id: call.id,
    name: call.name,
    arguments: call.arguments,
    status: "pending",
    files: [],
  };
}

function updateStep(
  steps: StepState[],
  step: number,
  update: (current: StepState) => StepState
): StepState[] {
  const index = steps.findIndex((s) => s.step === step);
  if (index === -1) {
    return [...steps, update(emptyStep(step))];
  }
  return steps.map((s, i) => (i === index ? update(s) : s));
}

function hasCall(steps: StepState[], id: string): boolean {
  return steps.some((s) => s.calls.some((c) => c.id === id));
}

function updateCall(
  steps: StepState[],
  id: string,
  update: (current: ToolCallState) => ToolCallState
): StepState[] {
  return steps.map((s) =>
    s.calls.some((c) => c.id === id)
      ? { ...s, calls: s.calls.map((c) => (c.id === id ? update(c) : c)) }
      : s
  );
}

/** Attach an unknown tool call to the newest step, creating step 1 if needed. */
function ensureCall(
  steps: StepState[],
  call: ToolCallRequest
): StepState[] {
  if (hasCall(steps, call.id)) {
    return steps;
  }
  if (steps.length === 0) {
    return [{ ...emptyStep(1), calls: [newCall(call)] }];
  }
  const last = steps.at(-1)?.step ?? 1;
  return updateStep(steps, last, (s) => ({
    ...s,
    calls: [...s.calls, newCall(call)],
  }));
}

/**
 * Older servers return no `files` array. Plot tools print the saved path, so
 * recover the image from the output and the session id as a best effort.
 */
function inferFiles(event: ToolEndEvent, session?: string): string[] {
  if (event.files && event.files.length > 0) {
    return event.files;
  }
  if (!(event.ok && session && event.output)) {
    return [];
  }
  const match = IMAGE_NAME.exec(event.output);
  return match ? [`/files/${session}/${match[1]}`] : [];
}

function appendEvent(events: RunEvent[], event: RunEvent): RunEvent[] {
  const last = events.at(-1);
  if (
    event.type === "token" &&
    last?.type === "token" &&
    last.step === event.step
  ) {
    return [
      ...events.slice(0, -1),
      { ...last, text: last.text + event.text },
    ];
  }
  return [...events, event];
}

function finish(
  state: RunState,
  event: { stopped: StopReason; steps: number; tool_calls: number; usage?: Usage },
  extra: { ms?: number; final?: string }
): RunState {
  const status: RunStatus = event.stopped === "error" ? "error" : "done";
  return {
    ...state,
    status,
    stopped: event.stopped,
    final: extra.final ?? state.final,
    totals: {
      steps: event.steps,
      toolCalls: event.tool_calls,
      usage: event.usage ?? {},
      ms: extra.ms ?? state.totals?.ms,
    },
  };
}

function settleOpenCalls(steps: StepState[]): StepState[] {
  return steps.map((s) => ({
    ...s,
    status: "responded" as const,
    calls: s.calls.map((c) =>
      c.status === "pending" || c.status === "running"
        ? { ...c, status: "cancelled" as const }
        : c
    ),
  }));
}

function applyEvent(state: RunState, event: RunEvent): RunState {
  switch (event.type) {
    case "session":
      return { ...state, session: event.session };
    case "llm_request":
      return {
        ...state,
        steps: updateStep(state.steps, event.step, (s) => ({
          ...s,
          messages: count(event.messages),
          tools: count(event.tools),
        })),
      };
    case "token":
      return {
        ...state,
        steps: updateStep(state.steps, event.step, (s) => ({
          ...s,
          status: "streaming",
          text: s.text + event.text,
        })),
      };
    case "llm_response":
      return {
        ...state,
        steps: updateStep(state.steps, event.step, (s) => {
          const known = new Set(s.calls.map((c) => c.id));
          const added = (event.calls ?? [])
            .filter((c) => !known.has(c.id))
            .map(newCall);
          return {
            ...s,
            status: "responded",
            ms: event.ms,
            usage: event.usage,
            text: event.text ?? s.text,
            requested: event.tool_calls ?? (event.calls ?? []).map((c) => c.name),
            calls: [...s.calls, ...added],
          };
        }),
      };
    case "tool_start": {
      const steps = ensureCall(state.steps, event);
      return {
        ...state,
        steps: updateCall(steps, event.id, (c) => ({
          ...c,
          arguments: event.arguments,
          status: "running",
          startedAt: event.t,
        })),
      };
    }
    case "tool_end": {
      const steps = ensureCall(state.steps, {
        id: event.id,
        name: event.name,
        arguments: {},
      });
      return {
        ...state,
        steps: updateCall(steps, event.id, (c) => ({
          ...c,
          status: event.ok ? "ok" : "error",
          ms: event.ms,
          output: event.output ?? "",
          files: inferFiles(event, state.session),
        })),
      };
    }
    case "error":
      return {
        ...state,
        errors: [...state.errors, { where: event.where, error: event.error }],
      };
    case "final":
      return finish(state, event, {});
    case "done":
      return finish(state, event, { ms: event.ms, final: event.final });
    case "client_abort":
      return {
        ...state,
        status: "aborted",
        steps: settleOpenCalls(state.steps),
      };
    case "client_error":
      return {
        ...state,
        status: "error",
        steps: settleOpenCalls(state.steps),
        errors: [...state.errors, { where: "client", error: event.error }],
      };
    default:
      return state;
  }
}

/** Pure reducer: the whole run view is derived from the event stream. */
export function reduceRun(state: RunState, event: RunEvent): RunState {
  const next = applyEvent(state, event);
  return { ...next, events: appendEvent(state.events, event) };
}

/** Streamed text that has not yet been confirmed by an `llm_response`. */
export function liveText(run: RunState): string {
  const last = run.steps.at(-1);
  return last && last.status === "streaming" ? last.text : "";
}

/** Drop token deltas before saving; the final text already contains them. */
export function compactRun(run: RunState): RunState {
  return { ...run, events: run.events.filter((e) => e.type !== "token") };
}

export function sumUsage(usage: Usage | undefined): {
  input?: number;
  output?: number;
  total?: number;
} {
  if (!usage) {
    return {};
  }
  const pick = (...keys: string[]): number | undefined => {
    for (const key of keys) {
      const value = usage[key];
      if (typeof value === "number") {
        return value;
      }
    }
    return undefined;
  };
  const input = pick("prompt_tokens", "input_tokens");
  const output = pick("completion_tokens", "output_tokens");
  const total =
    pick("total_tokens") ??
    (input !== undefined || output !== undefined
      ? (input ?? 0) + (output ?? 0)
      : undefined);
  return { input, output, total };
}
