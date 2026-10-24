"use client";

import Image from "next/image";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Streamdown } from "streamdown";
import {
  Activity,
  ArrowUpRight,
  Check,
  Clock3,
  Copy,
  FileCode2,
  FlaskConical,
  Gauge,
  Layers3,
  MessageSquareText,
  Network,
  Paperclip,
  Play,
  RotateCcw,
  Square,
  Terminal,
  Wrench,
  X,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  checkHealth,
  fileUrl,
  listAgents,
  listModels,
  listNotes,
  listTools,
  streamChat,
} from "@/lib/api";
import {
  createRun,
  reduceRun,
  sumUsage,
  type RunState,
} from "@/lib/run";
import type {
  AgentInfo,
  ModelsResponse,
  NoteItem,
  ToolSpec,
} from "@/lib/types";

const SAMPLE_PROMPTS: Record<string, readonly [string, string, string][]> = {
  general: [
    ["Summarize clearly", "Summarize the key decisions in this project in five concise bullets.", "A crisp overview for a busy teammate."],
    ["Plan a feature", "Create a practical implementation plan for adding saved chat sessions.", "Turn a fuzzy idea into next steps."],
    ["Draft a response", "Write a thoughtful, concise reply to a customer reporting a confusing error.", "Useful wording with a calm tone."],
  ],
  lab: [
    ["Save a protocol", "Remember this: Experiment A uses 5g NaCl in 100ml water.", "Store a reusable lab note."],
    ["Recall a protocol", "What does Experiment A use? Check your notes.", "Search session memory for a fact."],
    ["Compare conditions", "Compare the recorded conditions for Experiment A and identify anything missing.", "Inspect notes before making a conclusion."],
  ],
  math: [
    ["Solve an equation", "Solve x^2 + 5*x + 6 = 0 and explain the roots.", "Use symbolic math, then explain it."],
    ["Make a plot", "Plot y = sin(x) from -2*pi to 2*pi.", "Generate a visual artifact with the plot tool."],
    ["Simplify an expression", "Factor x^3 - 4*x^2 + x + 6 and verify the result.", "Show the symbolic steps and check the answer."],
  ],
};
const CHAT_HISTORY_KEY = "universal-agent-chat-history";
const MAX_SAVED_CHATS = 12;

function samplePromptsFor(agent: string) {
  return SAMPLE_PROMPTS[agent] ?? SAMPLE_PROMPTS.general;
}

function formatMs(ms?: number) {
  return ms === undefined
    ? "--"
    : ms < 1000
      ? `${Math.round(ms)}ms`
      : `${(ms / 1000).toFixed(1)}s`;
}
function runLabel(run: RunState | null) {
  if (!run) return "Ready";
  if (run.status === "running") return "Working";
  if (run.status === "aborted") return "Stopped";
  if (run.status === "error") return "Needs attention";
  return "Complete";
}
function statusVariant(run: RunState | null) {
  return run?.status === "error"
    ? "destructive"
    : run?.status === "running"
      ? "secondary"
      : "outline";
}
function chatTitle(prompt: string) {
  return prompt.length > 42 ? `${prompt.slice(0, 42).trimEnd()}...` : prompt;
}
function chatDate(timestamp: number) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(timestamp);
}

export default function AgentConsole() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [tools, setTools] = useState<ToolSpec[]>([]);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [notes, setNotes] = useState<NoteItem[]>([]);
  const [agent, setAgent] = useState("general");
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [run, setRun] = useState<RunState | null>(null);
  const [chatHistory, setChatHistory] = useState<RunState[]>([]);
  const [previousRun, setPreviousRun] = useState<RunState | null>(null);
  const [health, setHealth] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(CHAT_HISTORY_KEY);
      if (!stored) return;
      const parsed: unknown = JSON.parse(stored);
      if (Array.isArray(parsed)) {
        setChatHistory(parsed.filter((item): item is RunState => (
          typeof item === "object" &&
          item !== null &&
          typeof (item as RunState).id === "string" &&
          typeof (item as RunState).prompt === "string"
        )).slice(0, MAX_SAVED_CHATS));
      }
    } catch {
      window.localStorage.removeItem(CHAT_HISTORY_KEY);
    }
  }, []);

  useEffect(() => {
    if (!run || run.status === "running") return;
    setChatHistory((current) => {
      const next = [run, ...current.filter((item) => item.id !== run.id)].slice(0, MAX_SAVED_CHATS);
      window.localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(next));
      return next;
    });
  }, [run]);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      listAgents(controller.signal),
      listTools(controller.signal),
      listModels(controller.signal),
      checkHealth(controller.signal),
    ])
      .then(([loadedAgents, loadedTools, loadedModels, isHealthy]) => {
        setAgents(loadedAgents);
        setTools(loadedTools);
        setModels(loadedModels);
        setAgent(
          loadedAgents.find((item) => item.name === "lab")?.name ??
            loadedAgents[0]?.name ??
            "general",
        );
        setModel(loadedModels.default || loadedModels.models[0]?.id || "");
        setHealth(isHealthy);
      })
      .catch(() => {
        setHealth(false);
        setModels(null);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!run?.session) return;
    listNotes({ session: run.session, agent })
      .then(setNotes)
      .catch(() => setNotes([]));
  }, [run?.session, agent]);

  async function submit(message = prompt) {
    const text = message.trim();
    if (!text || !model || run?.status === "running") return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const nextRun = createRun({
      id: crypto.randomUUID(),
      prompt: text,
      agent,
      model,
      startedAt: Date.now(),
    });
    if (run?.final) setPreviousRun(run);
    setPrompt("");
    setRun(nextRun);
    try {
      for await (const event of streamChat(
        { agent, message: text, model, session: run?.session },
        controller.signal,
      )) {
        const updated = reduceRun(nextRun, event);
        Object.assign(nextRun, updated);
        setRun(updated);
      }
    } catch (error) {
      setRun(
        reduceRun(
          nextRun,
          error instanceof DOMException && error.name === "AbortError"
            ? { type: "client_abort" }
            : {
                type: "client_error",
                error:
                  error instanceof Error
                    ? error.message
                    : "The run failed unexpectedly",
              },
        ),
      );
    } finally {
      abortRef.current = null;
    }
  }

  async function copyAnswer() {
    if (!run?.final) return;
    await navigator.clipboard.writeText(run.final);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  const selectedAgent = agents.find((item) => item.name === agent);
  const samplePrompts = samplePromptsFor(agent);
  const usage = sumUsage(run?.totals?.usage);
  const hasModel = Boolean(model && models?.models.length);
  const startNewChat = () => {
    abortRef.current?.abort();
    setRun(null);
    setPreviousRun(null);
    setPrompt("");
  };

  return (
    <main className="min-h-screen min-w-0 overflow-x-hidden bg-[radial-gradient(circle_at_top,#f8fafc_0%,#f4f5f7_42%,#eef0f3_100%)] text-foreground">
      <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-black/[0.07] bg-background/85 px-4 shadow-[0_1px_12px_rgba(15,23,42,0.04)] backdrop-blur-xl md:px-6">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
            <FlaskConical className="size-4" />
          </div>
          <div>
            <p className="text-sm font-semibold leading-none">
              Universal Agent
            </p>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Harness console
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={`size-2 rounded-full ${health === true ? "bg-emerald-500" : health === false ? "bg-destructive" : "bg-muted-foreground"}`}
          />
          {health === true
            ? "API connected"
            : health === false
              ? "API unavailable"
              : "Connecting"}
          <Badge variant="outline" className="hidden sm:inline-flex">
            LOCAL
          </Badge>
        </div>
      </header>
      <div className="mx-auto grid min-h-[calc(100vh-4rem)] w-full max-w-[1800px] grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)] xl:grid-cols-[260px_minmax(0,1fr)_340px]">
        <aside className="hidden border-r border-black/6 bg-white/45 p-5 lg:flex lg:flex-col">
          <div className="mb-7 flex items-center justify-between">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                Workspace
              </p>
              <p className="mt-1 text-xs text-muted-foreground/80">Choose how to run your task</p>
            </div>
            <Button
              variant="ghost"
              size="icon-xs"
              title="Reset run"
              onClick={startNewChat}
            >
              <RotateCcw />
            </Button>
          </div>
          <p className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Run setup</p>
          <FieldLabel>Agent</FieldLabel>
          <Select
            value={agent}
            onValueChange={(value) => {
              if (value) setAgent(value);
            }}
            disabled={loading || !hasModel || run?.status === "running"}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {agents.map((item) => (
                <SelectItem key={item.name} value={item.name}>
                  {item.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="mt-2 min-h-10 rounded-lg bg-background/80 p-3 text-xs leading-4 text-muted-foreground ring-1 ring-border/60">{selectedAgent?.description ?? "Choose a specialist to run."}</div>
          <FieldLabel>Model preset</FieldLabel>
          <Select
            value={model}
            onValueChange={(value) => {
              if (value) setModel(value);
            }}
            disabled={loading || !hasModel || run?.status === "running"}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {models?.models.map((item) => (
                <SelectItem key={item.id} value={item.id}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
            <Separator className="my-6" />
            <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Quick starts
            </p>
            <Zap className="size-3.5 text-amber-500" />
          </div>
          <div className="grid gap-1">
            {samplePrompts.map(([label, value]) => (
              <Button
                key={label}
                variant="ghost"
                className="h-auto justify-between rounded-lg border border-transparent px-3 py-2.5 text-left text-xs font-normal text-muted-foreground hover:border-border hover:bg-background hover:text-foreground"
                onClick={() => setPrompt(value)}
              >
                {label}
                <ArrowUpRight className="size-3.5" />
              </Button>
            ))}
          </div>
          <Separator className="my-6" />
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Recent chats
            </p>
            {chatHistory.length > 0 && (
              <Badge variant="secondary" className="text-[10px]">{chatHistory.length}</Badge>
            )}
          </div>
          <div className="max-h-52 space-y-1 overflow-y-auto pr-1">
            {chatHistory.length === 0 ? (
              <p className="px-2 text-[11px] leading-4 text-muted-foreground">
                Completed chats will appear here.
              </p>
            ) : (
              chatHistory.map((chat) => (
                <button
                  key={chat.id}
                  type="button"
                  className={`w-full rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-background ${run?.id === chat.id ? "bg-background ring-1 ring-border" : ""}`}
                  onClick={() => {
                    setPreviousRun(null);
                    setRun(chat);
                    setPrompt("");
                  }}
                >
                  <span className="block truncate text-xs font-medium text-foreground">{chatTitle(chat.prompt)}</span>
                  <span className="mt-1 block text-[10px] text-muted-foreground">{chatDate(chat.startedAt)}</span>
                </button>
              ))
            )}
          </div>
          <div className="mt-auto grid gap-2 border-t pt-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-2">
              <Network className="size-3.5 text-primary" />
              {agents.length} agents
            </span>
            <span className="flex items-center gap-2">
              <Wrench className="size-3.5 text-primary" />
              {tools.length} tools
            </span>
          </div>
        </aside>
        <section className="flex min-w-0 flex-col">
          <div className="border-b border-black/[0.07] bg-background/75 px-4 py-3 backdrop-blur-sm md:px-8 lg:hidden">
            <div className="flex items-center gap-2">
              <Select value={agent} onValueChange={(value) => value && setAgent(value)} disabled={loading || run?.status === "running"}>
                <SelectTrigger className="h-9 min-w-0 flex-1 bg-background text-xs">
                  <SelectValue placeholder="Agent" />
                </SelectTrigger>
                <SelectContent>
                  {agents.map((item) => <SelectItem key={item.name} value={item.name}>{item.name}</SelectItem>)}
                </SelectContent>
              </Select>
              <Select value={model} onValueChange={(value) => value && setModel(value)} disabled={loading || run?.status === "running"}>
                <SelectTrigger className="h-9 min-w-0 flex-1 bg-background text-xs">
                  <SelectValue placeholder="Model" />
                </SelectTrigger>
                <SelectContent>
                  {models?.models.map((item) => <SelectItem key={item.id} value={item.id}>{item.label}</SelectItem>)}
                </SelectContent>
              </Select>
              <Button variant="outline" size="icon" className="size-9 shrink-0 bg-background" title="New chat" onClick={startNewChat}>
                <RotateCcw className="size-3.5" />
              </Button>
            </div>
          </div>
          <div className="flex min-h-20 items-center justify-between gap-4 border-b border-black/[0.07] bg-background/80 px-4 py-4 backdrop-blur-sm md:px-8">
            <div className="min-w-0">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                Live workspace
              </p>
              <h1 className="truncate text-base font-semibold md:text-lg">
                {run?.prompt ?? "Ask the harness to do something"}
              </h1>
            </div>
            <Badge variant={statusVariant(run)} className="shrink-0 gap-1.5">
              <span
                className={`size-1.5 rounded-full ${run?.status === "running" ? "animate-pulse bg-amber-500" : run?.status === "error" ? "bg-destructive" : "bg-emerald-500"}`}
              />
              {runLabel(run)}
            </Badge>
          </div>
          <ScrollArea className="flex-1">
            <div className="mx-auto flex min-h-full w-full max-w-4xl min-w-0 flex-col space-y-6 p-4 md:p-8 xl:px-10">
              {!run ? (
                <EmptyState prompts={samplePrompts} onSelect={setPrompt} />
              ) : (
                <>
                  <div className="space-y-7">
                    {run.steps.map((step) => (
                      <Step key={step.step} step={step} />
                    ))}
                  </div>
                  {run.status === "running" && <ActivityFeed events={run.events} />}
                  {run.final && (
                    <div className="space-y-3">
                      <Card className="border-emerald-500/30 bg-emerald-500/5">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                          <CardTitle className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-700">
                            <Check className="size-4" />
                            Final answer
                          </CardTitle>
                          <Button variant="ghost" size="sm" onClick={copyAnswer}>
                            {copied ? <Check /> : <Copy />}
                            {copied ? "Copied" : "Copy"}
                          </Button>
                        </CardHeader>
                        <CardContent>
                          <MarkdownContent content={run.final} />
                        </CardContent>
                      </Card>
                      {previousRun?.final && (
                        <Card>
                          <CardHeader className="pb-3">
                            <CardTitle className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider">
                              <Layers3 className="size-4 text-primary" />
                              Side-by-side comparison
                              <Badge variant="secondary" className="ml-auto">previous vs current</Badge>
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid gap-5 md:grid-cols-2">
                              <div className="min-w-0 space-y-2 md:border-r md:pr-5">
                                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Previous answer</p>
                                <MarkdownContent content={previousRun.final} muted />
                              </div>
                              <div className="min-w-0 space-y-2">
                                <p className="text-[10px] font-semibold uppercase tracking-wider text-primary">Current answer</p>
                                <MarkdownContent content={run.final} />
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  )}
                  {run.errors.map((error, index) => (
                    <Card
                      key={`${error.error}-${index}`}
                      className="border-destructive/30 bg-destructive/5"
                    >
                      <CardContent className="flex gap-2 p-3 text-sm text-destructive">
                        <X className="size-4 shrink-0" />
                        {error.where ? `${error.where}: ` : ""}
                        {error.error}
                      </CardContent>
                    </Card>
                  ))}
                </>
              )}
            </div>
          </ScrollArea>
          <div className="border-t border-black/[0.07] bg-background/80 p-3 backdrop-blur-md sm:p-4 md:px-8">
            <div className="mx-auto max-w-4xl">
              <Card className="gap-0 overflow-hidden border-black/10 bg-background py-0 shadow-[0_8px_30px_rgba(15,23,42,0.07)]">
                <Textarea
                  className="min-h-20 resize-none rounded-b-none border-0 shadow-none focus-visible:ring-0"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      (event.metaKey || event.ctrlKey)
                    ) {
                      event.preventDefault();
                      void submit();
                    }
                  }}
                  placeholder={`Message ${agent || "agent"}...`}
                  disabled={run?.status === "running"}
                />
                <div className="flex items-center justify-between border-t bg-muted/30 px-3 py-2">
                  <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <Paperclip className="size-3.5" />
                    context-aware
                  </span>
                  {run?.status === "running" ? (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => abortRef.current?.abort()}
                    >
                      <Square className="size-3.5" />
                      Stop
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => void submit()}
                      disabled={!prompt.trim() || !model}
                    >
                      <Play className="size-3.5 fill-current" />
                      Run
                    </Button>
                  )}
                </div>
              </Card>
              <p className="mt-2 text-[10px] text-muted-foreground">
                ⌘ Enter to run <span className="px-1">·</span> Sessions persist
                automatically
              </p>
            </div>
          </div>
        </section>
        <aside className="hidden border-l border-black/6 bg-white/45 p-3 xl:block">
          <Tabs defaultValue="run" className="h-full">
            <TabsList
              variant="line"
              className="grid h-12 w-full grid-cols-3 rounded-lg border bg-background p-1"
            >
              <TabsTrigger value="run">
                <Terminal />
                Run
              </TabsTrigger>
              <TabsTrigger value="tools">
                <Wrench />
                Tools{" "}
                <Badge
                  variant="secondary"
                  className="h-4 min-w-4 px-1 text-[10px]"
                >
                  {tools.length}
                </Badge>
              </TabsTrigger>
              <TabsTrigger value="notes">
                <FileCode2 />
                Notes{" "}
                <Badge
                  variant="secondary"
                  className="h-4 min-w-4 px-1 text-[10px]"
                >
                  {notes.length}
                </Badge>
              </TabsTrigger>
            </TabsList>
            <TabsContent value="run" className="mt-2 space-y-1 rounded-lg border bg-background p-4">
              <InfoRow
                icon={<Terminal />}
                label="Session"
                value={run?.session ?? "New session"}
                mono
              />
              <InfoRow
                icon={<Gauge />}
                label="Steps"
                value={
                  run ? String(run.totals?.steps ?? run.steps.length) : "0"
                }
              />
              <InfoRow
                icon={<Wrench />}
                label="Tool calls"
                value={
                  run
                    ? String(
                        run.totals?.toolCalls ??
                          run.steps.reduce(
                            (sum, step) => sum + step.calls.length,
                            0,
                          ),
                      )
                    : "0"
                }
              />
              <InfoRow
                icon={<Clock3 />}
                label="Duration"
                value={formatMs(run?.totals?.ms)}
              />
              <Separator className="my-5" />
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Token usage
              </p>
              <div className="grid grid-cols-3 gap-1">
                <Metric label="input" value={usage.input} />
                <Metric label="output" value={usage.output} />
                <Metric label="total" value={usage.total} />
              </div>
            </TabsContent>
            <TabsContent value="tools" className="mt-2 space-y-2 rounded-lg border bg-background p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Available tools
              </p>
              {tools.map((tool) => (
                <div
                  className="flex items-start justify-between gap-2 border-b py-2.5 text-xs"
                  key={tool.name}
                >
                  <div>
                    <p className="font-mono font-medium">{tool.name}</p>
                    <p className="mt-1 text-muted-foreground">
                      {tool.description}
                    </p>
                  </div>
                  <Badge
                    variant={tool.isolated ? "secondary" : "outline"}
                    className="shrink-0 text-[10px]"
                  >
                    {tool.isolated ? "isolated" : "ready"}
                  </Badge>
                </div>
              ))}
            </TabsContent>
            <TabsContent value="notes" className="mt-2 space-y-2 rounded-lg border bg-background p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Session memory
              </p>
              {notes.length === 0 ? (
                <div className="flex min-h-32 flex-col items-center justify-center gap-2 text-center text-xs text-muted-foreground">
                  <FileCode2 className="size-5" />
                  <p>No notes in this session yet.</p>
                </div>
              ) : (
                notes.map((note) => (
                  <div className="border-b py-2.5" key={String(note.id)}>
                    <Badge variant="outline" className="text-[10px]">
                      {note.tag ?? "note"}
                    </Badge>
                    <p className="mt-1 text-xs leading-5 text-muted-foreground">
                      {note.text}
                    </p>
                  </div>
                ))
              )}
            </TabsContent>
          </Tabs>
        </aside>
      </div>
    </main>
  );
}

function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <p className="mb-2 mt-4 text-xs font-medium text-muted-foreground">
      {children}
    </p>
  );
}
function EmptyState({ prompts, onSelect }: { prompts: readonly [string, string, string][]; onSelect: (value: string) => void }) {
  return (
    <div className="flex min-h-[55vh] flex-col items-center justify-center py-8 text-center">
      <div className="mb-5 flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary">
        <MessageSquareText className="size-6" />
      </div>
      <h2 className="text-xl font-semibold tracking-tight">
        Ready when you are.
      </h2>
      <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
        Send a task to an agent and watch its reasoning, tools, and artifacts
        arrive in real time.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-4 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Gauge className="size-3.5 text-primary" />
          streamed responses
        </span>
        <span className="flex items-center gap-1.5">
          <Layers3 className="size-3.5 text-primary" />
          session memory
        </span>
        <span className="flex items-center gap-1.5">
          <Activity className="size-3.5 text-primary" />
          tool traces
        </span>
      </div>
      <div className="mt-10 w-full max-w-2xl text-left">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Try a sample</p>
          <Zap className="size-3.5 text-amber-500" />
        </div>
        <div className="grid gap-2 md:grid-cols-3">
          {prompts.map(([label, value, description]) => (
            <Button key={label} variant="outline" className="h-auto min-h-24 items-start justify-between whitespace-normal p-3 text-left" onClick={() => onSelect(value)}>
              <span><span className="block text-xs font-semibold">{label}</span><span className="mt-1 block text-[11px] font-normal leading-4 text-muted-foreground">{description}</span></span>
              <ArrowUpRight className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
function Step({ step }: { step: RunState["steps"][number] }) {
  return (
    <article className="flex min-w-0 gap-3 sm:gap-4">
      <div className="flex flex-col items-center">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-lg border bg-muted font-mono text-[10px] font-semibold text-muted-foreground">
          {step.step.toString().padStart(2, "0")}
        </span>
        <span className="mt-2 w-px flex-1 bg-border" />
      </div>
      <div className="min-w-0 max-w-full flex-1 overflow-hidden pb-1">
        <div className="mb-3 flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
          <span>Agent step {step.step}</span>
          {step.ms !== undefined && (
            <span className="flex items-center gap-1">
              <Clock3 className="size-3" />
              {formatMs(step.ms)}
            </span>
          )}
          {step.calls.length > 0 && (
            <span className="flex items-center gap-1">
              <Wrench className="size-3" />
              {step.calls.length} tool{step.calls.length === 1 ? "" : "s"}
            </span>
          )}
        </div>
        {step.text && <MarkdownContent content={step.text} streaming={step.status === "streaming"} />}
        {step.calls.map((call) => (
          <ToolCall key={call.id} call={call} />
        ))}
      </div>
    </article>
  );
}

function ActivityFeed({ events }: { events: RunState["events"] }) {
  const visible = events.filter((event) => event.type !== "token").slice(-8);
  if (visible.length === 0) return null;

  return (
    <div className="rounded-xl border border-primary/15 bg-primary/[0.035] p-3">
      <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-primary">
        <span className="size-1.5 animate-pulse rounded-full bg-primary" />
        Live harness activity
      </div>
      <div className="grid gap-1.5">
        {visible.map((event, index) => {
          const label =
            event.type === "llm_request"
              ? `Reasoning step ${event.step} started`
              : event.type === "llm_response"
                ? `Model response received for step ${event.step}`
                : event.type === "tool_start"
                  ? `Running ${event.name}`
                  : event.type === "tool_end"
                    ? `${event.name} ${event.ok ? "completed" : "failed"}`
                    : event.type === "error"
                      ? `${event.where ?? "Harness"} error`
                      : event.type === "session"
                        ? "Session connected"
                        : event.type;
          return (
            <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground" key={`${event.type}-${index}`}>
              <span className={`size-1.5 shrink-0 rounded-full ${event.type === "error" || (event.type === "tool_end" && !event.ok) ? "bg-destructive" : "bg-emerald-500"}`} />
              <span className="truncate">{label}</span>
              {"t" in event && event.t !== undefined && (
                <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground/70">
                  {formatMs(event.t * 1000)}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ToolCall({
  call,
}: {
  call: RunState["steps"][number]["calls"][number];
}) {
  return (
    <Card size="sm" className="mt-3 min-w-0 max-w-full gap-2 overflow-hidden bg-muted/30">
      <CardHeader className="flex flex-row items-center gap-2 space-y-0 py-2.5">
        <span className="flex size-6 items-center justify-center rounded-md bg-primary/10 text-primary">
          <Wrench className="size-3.5" />
        </span>
        <CardTitle className="font-mono text-xs">{call.name}</CardTitle>
        <Badge
          variant={
            call.status === "error"
              ? "destructive"
              : call.status === "ok"
                ? "secondary"
                : "outline"
          }

          className="ml-auto text-[10px]"
        >
          {call.status}
        </Badge>
        {call.ms !== undefined && (
          <span className="text-[10px] text-muted-foreground">
            {formatMs(call.ms)}
          </span>
        )}
      </CardHeader>
      <CardContent className="space-y-2 py-0 pb-3">
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">View arguments</summary>
          <pre className="mt-2 overflow-auto rounded-md bg-background p-2 font-mono text-[10px]">
            {JSON.stringify(call.arguments, null, 2)}
          </pre>
        </details>
        {call.output && (
          <p className="whitespace-pre-wrap font-mono text-[11px] leading-5 text-muted-foreground">
            {call.output}
          </p>
        )}
        {call.files.map((file) => (
          <Image
            key={file}
            className="rounded-md border"
            src={fileUrl(file)}
            alt={`Artifact from ${call.name}`}
            width={640}
            height={420}
            unoptimized
          />
        ))}
      </CardContent>
    </Card>
  );
}

function MarkdownContent({
  content,
  streaming = false,
  muted = false,
}: {
  content: string;
  streaming?: boolean;
  muted?: boolean;
}) {
  return (
    <div className={`streamdown-content ${muted ? "text-muted-foreground" : ""}`}>
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        isAnimating={streaming}
        animated={streaming}
        caret={streaming ? "block" : undefined}
        parseIncompleteMarkdown={streaming}
      >
        {content}
      </Streamdown>
    </div>
  );
}

function InfoRow({
  icon,
  label,
  value,
  mono = false,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex min-h-9 items-center justify-between gap-3 border-b py-2 text-xs">
      <span className="flex items-center gap-2 text-muted-foreground">
        {icon}
        {label}
      </span>
      <strong
        className={`max-w-32 truncate font-medium ${mono ? "font-mono text-[10px]" : ""}`}
      >
        {value}
      </strong>
    </div>
  );
}
function Metric({ label, value }: { label: string; value?: number }) {
  return (
    <div className="rounded-md border bg-muted/20 p-2 text-center">
      <strong className="block font-mono text-xs">{value ?? "--"}</strong>
      <span className="text-[10px] text-muted-foreground">{label}</span>
    </div>
  );
}
