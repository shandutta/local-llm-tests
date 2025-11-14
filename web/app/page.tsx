"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import useSWR from "swr";

type ModelManifest = {
  defaults?: Record<string, unknown>;
  models: Record<
    string,
    {
      description?: string;
      port?: number;
      relative_path?: string;
      arguments?: string[];
    }
  >;
};

type StatusResponse = {
  containers: Array<{
    name?: string;
    state?: string;
    published_ports?: string[];
  }>;
};

type ChatMessage = {
  role: string;
  content: string;
};

type ReasoningEffort = "low" | "medium" | "high";

const apiBase = process.env.NEXT_PUBLIC_API_BASE;

const fetcher = async (path: string) => {
  const res = await fetch(`${apiBase}${path}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}`);
  }
  return res.json();
};

export default function Home() {
  const { data: manifest } = useSWR<ModelManifest>("/models", fetcher);
  const { data: status } = useSWR<StatusResponse>("/status", fetcher, {
    refreshInterval: 5000,
  });

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isModelBusy, setIsModelBusy] = useState(false);
  const [activeModelName, setActiveModelName] = useState<string | null>(null);
  const [pendingModel, setPendingModel] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<"start" | "stop" | null>(null);
  const [reasoning, setReasoning] = useState<ReasoningEffort>("medium");
  const [progress, setProgress] = useState(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streamingReply, setStreamingReply] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!selectedModel && manifest) {
      const first = Object.keys(manifest.models ?? {})[0];
      if (first) setSelectedModel(first);
    }
  }, [manifest, selectedModel]);

  const runningModels = useMemo(() => {
    const models = new Set<string>();
    if (!manifest || !status?.containers) return models;
    Object.entries(manifest.models ?? {}).forEach(([name]) => {
      const expectedName = `llm-${name}`;
      const matched = status.containers?.some(
        (container) =>
          container.name?.includes(expectedName) &&
          (container.state?.toLowerCase().includes("up") ||
            container.state?.toLowerCase().includes("running")),
      );
      if (matched) models.add(name);
    });
    return models;
  }, [manifest, status]);

  useEffect(() => {
    const first = runningModels.values().next().value ?? null;
    setActiveModelName(first ?? null);
  }, [runningModels]);

  useEffect(() => {
    if (!pendingAction) return;
    if (pendingAction === "start" && pendingModel && pendingModel === activeModelName) {
      setProgress(100);
      setTimeout(() => setProgress(0), 800);
      setIsModelBusy(false);
      setPendingAction(null);
      setPendingModel(null);
    } else if (pendingAction === "stop" && !activeModelName) {
      setProgress(0);
      setIsModelBusy(false);
      setPendingAction(null);
      setPendingModel(null);
    }
  }, [pendingAction, pendingModel, activeModelName]);

  useEffect(() => {
    if (!pendingAction || !isModelBusy) {
      setProgress((prev) => (prev === 100 ? prev : 0));
      return;
    }
    const target =
      pendingAction === "start"
        ? { initial: 25, increment: 10, ceiling: 90 }
        : { initial: 20, increment: 15, ceiling: 95 };
    setProgress((prev) => (prev < target.initial ? target.initial : prev));
    const timer = setInterval(() => {
      setProgress((prev) => Math.min(prev + target.increment, target.ceiling));
    }, 800);
    return () => clearInterval(timer);
  }, [pendingAction, isModelBusy]);

  const sendMessage = async () => {
    if (!selectedModel || !input.trim()) return;
    const outbound: ChatMessage = { role: "user", content: input.trim() };
    const nextHistory = [...messages, outbound];
    setMessages(nextHistory);
    setInput("");
    setStreamingReply("");
    setIsStreaming(true);

    try {
      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          reasoning_effort: reasoning,
          messages: nextHistory,
        }),
      });

      if (!response.ok || !response.body) {
        const errorText = await response.text();
        throw new Error(errorText || "Chat request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const event of events) {
          const line = event.trim();
          if (!line || line === "data: [DONE]") continue;
          const payloadRaw = line.replace(/^data:\s*/, "");
          try {
            const payload = JSON.parse(payloadRaw);
            const delta =
              payload?.choices?.[0]?.delta?.content ??
              payload?.choices?.[0]?.text ??
              payload?.token?.text ??
              "";
            assistantText += delta;
            setStreamingReply(assistantText);
          } catch {
            continue;
          }
        }
      }

      setMessages((prev) => [...prev, { role: "assistant", content: assistantText }]);
      setStreamingReply("");
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${
            err instanceof Error ? err.message : "chat service unavailable"
          }`,
        },
      ]);
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-12">
        <header>
          <h1 className="text-3xl font-bold">Local LLM Control Center</h1>
          <p className="text-sm text-zinc-500">
            Backend API: {apiBase ?? "not configured"} •{" "}
            {isModelBusy
              ? "Switching models…"
              : `Active model: ${activeModelName ?? "none"}`}
          </p>
          {isModelBusy && (
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-200">
              <div
                className="h-full animate-pulse bg-gradient-to-r from-sky-500 via-indigo-500 to-sky-500 transition-[width]"
                style={{ width: `${progress || 40}%` }}
              />
            </div>
          )}
        </header>

        <section className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold">Models</h2>
          {!manifest ? (
            <p className="text-zinc-500">Loading manifest…</p>
          ) : (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {Object.entries(manifest.models ?? {}).map(([name, model]) => {
                const isRunning = runningModels.has(name);
                const isPendingStart =
                  isModelBusy &&
                  pendingAction === "start" &&
                  pendingModel === name;
                const isPendingStop =
                  isModelBusy &&
                  pendingAction === "stop" &&
                  pendingModel === name;
                return (
                  <button
                    key={name}
                    onClick={() => setSelectedModel(name)}
                    className={`rounded-xl border p-4 text-left transition ${
                      selectedModel === name
                        ? "border-sky-500 bg-sky-50"
                        : "border-zinc-200 bg-zinc-50 hover:border-zinc-300"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-lg font-semibold">{name}</p>
                        {model.port && (
                          <p className="text-xs text-zinc-500">Port {model.port}</p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        {isRunning && (
                          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">
                            Active
                          </span>
                        )}
                        <span
                          className={`h-3 w-3 rounded-full ${
                            isRunning ? "bg-emerald-500" : "bg-zinc-300"
                          }`}
                        />
                      </div>
                    </div>
                    <p className="mt-2 text-sm text-zinc-600">
                      {model.description ?? "No description provided."}
                    </p>
                  <div className="mt-4 flex gap-2 text-sm text-zinc-500">
                      <button
                        className="rounded-full bg-zinc-900 px-3 py-1 text-white disabled:opacity-50"
                        disabled={
                          isModelBusy && !isPendingStart && pendingAction !== null
                        }
                        onClick={async (e) => {
                          e.stopPropagation();
                          if (isModelBusy && !isPendingStart) return;
                          setIsModelBusy(true);
                          setPendingModel(name);
                          setPendingAction("start");
                          await fetch(`${apiBase}/start`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ model: name }),
                          }).catch(() => {
                            setIsModelBusy(false);
                            setPendingAction(null);
                            setPendingModel(null);
                          });
                        }}
                      >
                        {isPendingStart
                          ? "Starting…"
                          : isRunning
                          ? "Restart"
                          : "Start"}
                      </button>
                      <button
                        className="rounded-full border border-zinc-300 px-3 py-1 disabled:opacity-50"
                        disabled={
                          isPendingStart ||
                          (isModelBusy && !isPendingStop) ||
                          !isRunning
                        }
                        onClick={async (e) => {
                          e.stopPropagation();
                          if ((!isPendingStop && isModelBusy) || !isRunning) return;
                          setIsModelBusy(true);
                          setPendingModel(name);
                          setPendingAction("stop");
                          await fetch(`${apiBase}/stop`, { method: "POST" }).catch(() => {
                            setIsModelBusy(false);
                            setPendingAction(null);
                            setPendingModel(null);
                          });
                        }}
                      >
                        {isPendingStop ? "Stopping…" : "Stop"}
                      </button>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-xl font-semibold">Chat Console</h2>
              <p className="text-sm text-zinc-500">
                Streams are proxied through Harmony for GPT-OSS.
              </p>
            </div>
            <div className="flex gap-3">
              <label className="text-sm text-zinc-600">
                Reasoning effort
                <select
                  value={reasoning}
                  onChange={(e) => setReasoning(e.target.value as ReasoningEffort)}
                  className="ml-2 rounded-lg border border-zinc-300 px-2 py-1 text-sm"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </label>
            </div>
          </div>

          <div className="mt-6 space-y-4 rounded-xl border border-dashed border-zinc-200 bg-zinc-50 p-4">
            {!selectedModel ? (
              <p className="text-sm text-zinc-500">Select a model to begin.</p>
            ) : selectedModel !== activeModelName ? (
              <p className="text-sm text-zinc-500">
                {isModelBusy
                  ? "Starting the model…"
                  : "Model not running. Start it to send messages."}
              </p>
            ) : messages.length === 0 && !streamingReply ? (
              <p className="text-sm text-zinc-500">
                Chat is ready. Harmony formatting enables reasoning-effort controls on GPT-OSS.
              </p>
            ) : (
              <>
                {messages.map((msg, idx) => {
                  const prettyRole =
                    msg.role.charAt(0).toUpperCase() + msg.role.slice(1);
                  return (
                    <div key={`${msg.role}-${idx}`} className="text-sm leading-6">
                      <p className="font-semibold text-zinc-700">{prettyRole}</p>
                      <div className="prose prose-sm max-w-none text-zinc-800">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  );
                })}
                {streamingReply && (
                  <div className="text-sm leading-6">
                    <p className="font-semibold text-zinc-700">Assistant (streaming)</p>
                    <div className="prose prose-sm max-w-none text-zinc-800">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {streamingReply}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="mt-4 flex flex-col gap-3 md:flex-row">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question or give an instruction…"
              className="h-24 flex-1 rounded-xl border border-zinc-200 p-3 text-sm"
              onKeyDown={handleTextareaKeyDown}
              disabled={
                !selectedModel ||
                isStreaming ||
                selectedModel !== activeModelName
              }
            />
            <button
              disabled={
                !selectedModel ||
                isStreaming ||
                selectedModel !== activeModelName
              }
              onClick={sendMessage}
              className="rounded-xl bg-zinc-900 px-6 py-3 text-white disabled:opacity-50"
            >
              {isStreaming ? "Streaming…" : "Send"}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
