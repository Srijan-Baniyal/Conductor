import { readEvents } from "./sse";
import type {
  AgentInfo,
  ChatRequest,
  HarnessEvent,
  ModelsResponse,
  NoteItem,
  SessionResponse,
  ToolSpec,
} from "./types";

const TRAILING_SLASHES = /\/+$/;

export const API_BASE = (
  process.env.NEXT_PUBLIC_HARNESS_URL ?? "http://localhost:8765"
).replace(TRAILING_SLASHES, "");

export const START_COMMAND =
  "uv run python -m harness --serve --mock";

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(`Cannot reach the harness API at ${API_BASE}`);
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(
      `${response.status} ${response.statusText} ${detail}`.trim(),
      response.status
    );
  }
  return response;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await request(path, { signal });
  return (await response.json()) as T;
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  const body = await getJson<{ ok?: boolean }>("/api/health", signal);
  return body.ok === true;
}

export function listAgents(signal?: AbortSignal): Promise<AgentInfo[]> {
  return getJson<AgentInfo[]>("/api/agents", signal);
}

export function listTools(signal?: AbortSignal): Promise<ToolSpec[]> {
  return getJson<ToolSpec[]>("/api/tools", signal);
}

export function listModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return getJson<ModelsResponse>("/api/models", signal);
}

export function getSession(
  id: string,
  signal?: AbortSignal
): Promise<SessionResponse> {
  return getJson<SessionResponse>(
    `/api/sessions/${encodeURIComponent(id)}`,
    signal
  );
}

export async function deleteSession(id: string): Promise<void> {
  await request(`/api/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function listNotes(
  params: { session?: string; agent?: string },
  signal?: AbortSignal
): Promise<NoteItem[]> {
  const query = new URLSearchParams();
  if (params.session) {
    query.set("session", params.session);
  }
  if (params.agent) {
    query.set("agent", params.agent);
  }
  return getJson<NoteItem[]>(`/api/notes?${query.toString()}`, signal);
}

/** Server file paths are relative ("/files/<session>/<name>"); make them absolute. */
export function fileUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

/** POST /api/chat and yield each Server-Sent Event as it arrives. */
export async function* streamChat(
  body: ChatRequest,
  signal?: AbortSignal
): AsyncGenerator<HarnessEvent> {
  const response = await request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.body) {
    throw new ApiError("The server returned no response body");
  }
  yield* readEvents(response.body);
}
