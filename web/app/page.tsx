"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import useSWR from "swr";
import "katex/dist/katex.min.css";

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

const MATH_MACRO_PATTERN =
  /\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|frac|tfrac|sqrt|sum|prod|int|oint|log|ln|sin|cos|tan|csc|sec|cot|Re|Im|text|mathrm|mathbf|mathbb|mathcal|operatorname|partial|nabla|infty|cdot|pm|leq|geq|neq|approx|sim)/i;

const GPT_OSS_MODELS = new Set(["gpt-oss-120b", "gpt-oss-120b-uncensored"]);

const isGptOssModel = (model: string | null) =>
  model ? GPT_OSS_MODELS.has(model.toLowerCase()) : false;

const cleanGptOssText = (input: string) => {
  if (!input) return input;
  const finalMatch = input.match(
    /<\|channel\|>final(?:<\|message\|>)?([\s\S]*?)(?:<\|end\|>|$)/i,
  );
  if (finalMatch?.[1]) return finalMatch[1].trim();
  return input
    .replace(/<\|start\|>.*?(?=<\|channel\|>final)/gis, " ")
    .replace(/<\|start\|>|<\|end\|>|<\|message\|>/gi, " ")
    .replace(/<\|channel\|>[a-zA-Z0-9_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
};

const normalizeMathDelimiters = (text: string) => {
  if (!text) return text;
  const convertSegment = (
    value: string,
    pattern: RegExp,
    wrap: (expr: string) => string,
  ) =>
    value.replace(pattern, (full, expr) => {
      if (!expr || !expr.includes("\\")) return full;
      if (!MATH_MACRO_PATTERN.test(expr) && !/[{}_^]/.test(expr)) return full;
      return wrap(expr.trim());
    });

  let normalized = text;
  normalized = convertSegment(
    normalized,
    /\[\s*([\s\S]*?)\s*\]/g,
    (expr) => `$$${expr}$$`,
  );
  normalized = convertSegment(
    normalized,
    /\(\s*([\s\S]*?)\s*\)/g,
    (expr) => `$${expr}$`,
  );
  return normalized;
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
  const [chatStatus, setChatStatus] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [streamingReply, setStreamingReply] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  const isModelReady = selectedModel && selectedModel === activeModelName;
  const supportsReasoningEffort = isGptOssModel(selectedModel);

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
      const matched = status.containers?.some((container) => {
        const state = container.state?.toLowerCase() ?? "";
        return (
          container.name === expectedName &&
          (state.includes("up") || state.includes("running"))
        );
      });
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

  const handleCardKeydown = (
    e: React.KeyboardEvent<HTMLDivElement>,
    modelName: string,
  ) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedModel(modelName);
    }
  };

  const parseGptOssContent = (content: string) => {
    const analysisMatch = content.match(
      /<\|channel\|>analysis(?:<\|message\|>)?([\s\S]*?)(?:<\|end\|>|$)/i,
    );
    const finalMatch = content.match(
      /<\|channel\|>final(?:<\|message\|>)?([\s\S]*?)(?:<\|end\|>|$)/i,
    );
    return {
      analysis: analysisMatch?.[1]?.trim(),
      final: finalMatch?.[1]?.trim() ?? cleanGptOssText(content),
    };
  };

  const sendMessage = async () => {
    if (!selectedModel || !input.trim()) return;
    const outbound: ChatMessage = { role: "user", content: input.trim() };
    const nextHistory = [...messages, outbound];
    setMessages(nextHistory);
    setInput("");
    setStreamingReply("");
    setIsStreaming(true);

    let retries = 0;
    const maxRetries = 6;
    let response: Response | null = null;

    try {
      while (retries <= maxRetries) {
        response = await fetch(`${apiBase}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedModel,
            reasoning_effort: reasoning,
            messages: nextHistory,
          }),
        });

        if (response.status === 503) {
          retries += 1;
          if (retries > maxRetries) break;
          setChatStatus(
            `Model is still loading… retrying (${retries}/${maxRetries})`,
          );
          const delay = Math.min(4000 * retries, 15000);
          await new Promise((resolve) => setTimeout(resolve, delay));
          continue;
        }

        break;
      }

      setChatStatus(null);

      if (!response) {
        throw new Error("Chat request failed: no response");
      }

      if (response.status === 503) {
        setChatStatus(
          "Model is still loading after several attempts. Please try again in a moment.",
        );
        setIsStreaming(false);
        return;
      }

      if (!response.ok || !response.body) {
        const errorText = await response.text();
        let message = errorText || "Chat request failed";
        try {
          const parsed = JSON.parse(errorText);
          message =
            parsed?.detail?.error?.message ||
            parsed?.error?.message ||
            message;
        } catch {
          // ignore JSON parse errors
        }
        throw new Error(message);
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
          if (!event.trim()) continue;
          const lines = event.split("\n").map((line) => line.trim());
          const dataLine = lines.find((line) => line.startsWith("data:"));
          if (!dataLine) continue;
          if (dataLine === "data: [DONE]") {
            buffer = "";
            break;
          }
          const payloadRaw = dataLine.replace(/^data:\s*/, "");
          try {
            const payload = JSON.parse(payloadRaw);
            const delta =
              payload?.choices?.[0]?.delta?.content ??
              payload?.choices?.[0]?.text ??
              payload?.token?.text ??
              payload?.content ??
              "";
            if (!delta) continue;
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
      setChatStatus(
        err instanceof Error ? err.message : "Chat service unavailable",
      );
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

  const handleTextareaKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      sendMessage();
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
                  <div
                    key={name}
                    className={`rounded-xl border p-4 text-left transition ${
                      selectedModel === name
                        ? "border-sky-500 bg-sky-50"
                        : "border-zinc-200 bg-zinc-50 hover:border-zinc-300"
                    }`}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedModel(name)}
                    onKeyDown={(e) => handleCardKeydown(e, name)}
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
                  </div>
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
                  disabled={!supportsReasoningEffort}
                  className={`ml-2 rounded-lg border px-2 py-1 text-sm ${
                    supportsReasoningEffort
                      ? "border-zinc-300"
                      : "cursor-not-allowed border-zinc-200 bg-zinc-100 text-zinc-400"
                  }`}
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
                {!supportsReasoningEffort && (
                  <span className="ml-2 text-xs text-zinc-400">
                    (Only available for GPT-OSS)
                  </span>
                )}
              </label>
            </div>
          </div>

          <div className="mt-6 space-y-4 rounded-xl border border-dashed border-zinc-200 bg-zinc-50 p-4">
            {chatStatus && (
              <p className="rounded-lg bg-amber-100 px-3 py-2 text-sm text-amber-800">
                {chatStatus}
              </p>
            )}
            {!selectedModel ? (
              <p className="text-sm text-zinc-500">Select a model to begin.</p>
            ) : !isModelReady ? (
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
                  const isGptOss = isGptOssModel(selectedModel);
                  const { analysis, final } =
                    isGptOss && msg.content.includes("<|channel|>")
                      ? parseGptOssContent(msg.content)
                      : { analysis: undefined, final: msg.content };
                  return (
                    <div key={`${msg.role}-${idx}`} className="text-sm leading-6">
                      <p className="font-semibold text-zinc-700">{prettyRole}</p>
                      {analysis && (
                        <details className="mb-2 rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-600">
                          <summary className="cursor-pointer select-none text-zinc-700">
                            Reasoning trace
                          </summary>
                          <div className="prose prose-sm max-w-none text-zinc-800">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                          >
                            {normalizeMathDelimiters(analysis)}
                          </ReactMarkdown>
                        </div>
                      </details>
                    )}
                    <div className="prose prose-sm max-w-none text-zinc-800">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {normalizeMathDelimiters(final)}
                      </ReactMarkdown>
                    </div>
                  </div>
                );
              })}
                {streamingReply && (
                  <div className="text-sm leading-6">
                    <p className="font-semibold text-zinc-700">Assistant (streaming)</p>
                    <div className="prose prose-sm max-w-none text-zinc-800">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {normalizeMathDelimiters(
                          isGptOssModel(selectedModel)
                            ? cleanGptOssText(streamingReply)
                            : streamingReply,
                        )}
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
                !isModelReady
              }
            />
            <button
              disabled={
                !selectedModel ||
                isStreaming ||
                !isModelReady
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
