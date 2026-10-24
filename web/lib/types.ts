export interface ToolBrief {
  name: string;
  description: string;
  isolated: boolean;
}

export interface AgentInfo {
  name: string;
  description: string;
  system_prompt: string;
  tools: ToolBrief[];
  max_steps: number;
  tool_mode: string;
  notes_scope: string;
  temperature: number;
}

export interface JsonSchema {
  type?: string | string[];
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: unknown[];
  items?: JsonSchema;
  default?: unknown;
  [key: string]: unknown;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: JsonSchema;
  timeout: number;
  isolated: boolean;
  requires_approval: boolean;
}

export interface ModelPreset {
  id: string;
  label: string;
  provider: string;
  model: string;
  base_url: string | null;
  extra_body: Record<string, unknown>;
}

export interface ModelsResponse {
  default: string;
  models: ModelPreset[];
}

export interface NoteItem {
  id: string | number;
  tag?: string | null;
  text: string;
  [key: string]: unknown;
}

export interface ChatMessage {
  role: string;
  content: string | null;
  tool_calls?: { id: string; name: string; arguments: unknown }[];
  tool_call_id?: string | null;
  name?: string | null;
}

export interface SessionResponse {
  session: string;
  messages: ChatMessage[];
}

export type Usage = Record<string, unknown>;

export interface ToolCallRequest {
  id: string;
  name: string;
  arguments: unknown;
}

/** A count (current server) or the full list (older contract). */
export type CountOrList = number | unknown[];

export interface SessionEvent {
  type: "session";
  session: string;
  agent: string;
  model: string;
  t?: number;
}

export interface LlmRequestEvent {
  type: "llm_request";
  t?: number;
  step: number;
  messages?: CountOrList;
  tools?: CountOrList;
}

export interface TokenEvent {
  type: "token";
  t?: number;
  step: number;
  text: string;
}

export interface LlmResponseEvent {
  type: "llm_response";
  t?: number;
  step: number;
  ms?: number;
  tool_calls?: string[];
  calls?: ToolCallRequest[];
  usage?: Usage;
  text?: string;
}

export interface ToolStartEvent {
  type: "tool_start";
  t?: number;
  id: string;
  name: string;
  arguments: unknown;
}

export interface ToolEndEvent {
  type: "tool_end";
  t?: number;
  id: string;
  name: string;
  ok: boolean;
  ms?: number;
  output?: string;
  files?: string[];
}

export interface ErrorEvent {
  type: "error";
  t?: number;
  where?: string;
  error: string;
}

export type StopReason = "done" | "max_steps" | "error";

export interface FinalEvent {
  type: "final";
  t?: number;
  stopped: StopReason;
  steps: number;
  tool_calls: number;
  usage?: Usage;
}

export interface DoneEvent {
  type: "done";
  t?: number;
  stopped: StopReason;
  steps: number;
  tool_calls: number;
  usage?: Usage;
  final?: string;
  ms?: number;
}

export type HarnessEvent =
  | SessionEvent
  | LlmRequestEvent
  | TokenEvent
  | LlmResponseEvent
  | ToolStartEvent
  | ToolEndEvent
  | ErrorEvent
  | FinalEvent
  | DoneEvent;

/** Events the browser adds itself, so the reducer can model every outcome. */
export interface ClientAbortEvent {
  type: "client_abort";
}

export interface ClientErrorEvent {
  type: "client_error";
  error: string;
}

export type RunEvent = HarnessEvent | ClientAbortEvent | ClientErrorEvent;

export interface ChatRequest {
  agent: string;
  message: string;
  session?: string;
  model?: string;
}
