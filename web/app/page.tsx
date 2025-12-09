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
  meta?: {
    model?: string;
    durationMs?: number;
    tokens?: number;
    raw?: string;
  };
};

type RegisterProgress = {
  status: string;
  message?: string | null;
  downloaded_bytes?: number;
  total_bytes?: number;
  file?: string | null;
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

const GPT_OSS_MODELS = new Set(["gpt-oss-120b"]);

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

const collapseWhitespace = (text: string) => {
  if (!text) return text;
  return (
    text
      .split("\n")
      .map((line) => line.replace(/\s+$/, ""))
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim()
  );
};

const formatAssistantContent = (text: string, model: string | null) => {
  const cleaned = isGptOssModel(model) ? cleanGptOssText(text) : text;
  return collapseWhitespace(cleaned);
};

const stripEndTokens = (text: string) =>
  text.replace(/<\|im_end\|>|<\|end\|>|\s*<\/s>\s*/gi, " ").trim();

const simplifyPunctuation = (text: string) =>
  text
    .replace(/\u2026/g, "...")
    .replace(/([.?!]){3,}/g, "$1")
    .replace(/[•·]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();

const dropPunctuationSpam = (text: string) => {
  const lines = text.split("\n");
  const kept: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const nonAlpha = trimmed.replace(/[A-Za-z0-9]/g, "").length;
    const ratio = nonAlpha / Math.max(1, trimmed.length);
    if (ratio > 0.9 && trimmed.length > 10) continue;
    kept.push(trimmed);
  }
  return kept.join("\n");
};

const dedupeAdjacentLines = (text: string) => {
  const lines = text.split("\n");
  const deduped: string[] = [];
  let last = "";
  for (const line of lines) {
    if (line.trim() && line.trim() === last) continue;
    deduped.push(line);
    last = line.trim();
  }
  return deduped.join("\n");
};

const dedupeParagraphs = (text: string) => {
  const paras = text.split(/\n\s*\n/);
  const seen = new Set<string>();
  const kept: string[] = [];
  for (const para of paras) {
    const key = para.trim().slice(0, 120).toLowerCase();
    if (key && seen.has(key)) continue;
    if (key) seen.add(key);
    kept.push(para);
  }
  return kept.join("\n\n");
};

const dedupeSentences = (text: string) => {
  const sentences = text.split(/(?<=[.?!])\s+/);
  const seen = new Set<string>();
  const kept: string[] = [];
  for (const sentence of sentences) {
    const key = sentence.trim().toLowerCase();
    if (!key) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push(sentence.trim());
  }
  return kept.join(" ");
};

const applyAggressiveClean = (text: string, model: string | null) => {
  const base = simplifyPunctuation(formatAssistantContent(text, model));
  const cleaned = dedupeAdjacentLines(dropPunctuationSpam(stripEndTokens(base)));
  return dedupeSentences(dedupeParagraphs(cleaned));
};

const PREVIEW_CHAR_LIMIT = 1500;

const buildPreview = (content: string) => {
  const paragraphs = content.split(/\n\s*\n/).filter(Boolean);
  const firstPara = paragraphs[0] ?? content;
  const previewBase = firstPara.slice(0, PREVIEW_CHAR_LIMIT);
  const truncated =
    content.length > PREVIEW_CHAR_LIMIT ||
    firstPara.length < content.length;
  return { preview: previewBase, truncated };
};

const SYSTEM_GUARDRAIL =
  "You are a concise assistant. Default to brief, complete answers, but if the user explicitly asks for long-form (stories, lists, code), provide the requested length. Do not repeat acknowledgments or ask for clarification unless essential. Avoid filler, hedging, and restating the prompt. If greeted, greet once, then respond directly.";

const estimateTokens = (text: string) =>
  Math.max(1, Math.round(text.length / 4));

export default function Home() {
  const { data: manifest, mutate: mutateManifest } = useSWR<ModelManifest>(
    "/models",
    fetcher,
  );
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
  const [modelUrl, setModelUrl] = useState("");
  const [modelName, setModelName] = useState("");
  const [modelDescription, setModelDescription] = useState("");
  const [registerStatus, setRegisterStatus] = useState<string | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [expandedMessages, setExpandedMessages] = useState<Set<number>>(new Set());
  const [showRawMessages, setShowRawMessages] = useState<Set<number>>(new Set());

  const { data: registerProgress } = useSWR<RegisterProgress>(
    isRegistering ? "/models/register/status" : null,
    fetcher,
    { refreshInterval: 1000 },
  );

  const markdownComponents = useMemo(
    () => ({
      code({
        inline,
        className,
        children,
        ...props
      }: {
        inline?: boolean;
        className?: string;
        children?: React.ReactNode;
      }) {
        const text = String(children ?? "");
        if (inline) {
          return (
            <code className={className} {...props}>
              {text}
            </code>
          );
        }
        const handleCopy = () => {
          if (typeof navigator !== "undefined" && navigator.clipboard) {
            navigator.clipboard.writeText(text.trimEnd()).catch(() => {});
          }
        };
        return (
          <div className="group relative">
            <pre className="overflow-x-auto rounded-lg border border-zinc-200 bg-white p-3 text-xs leading-5">
              <code className={className} {...props}>
                {text}
              </code>
            </pre>
            <button
              type="button"
              onClick={handleCopy}
              className="absolute right-2 top-2 rounded-full bg-white px-2 py-1 text-[10px] font-semibold text-zinc-600 shadow-sm ring-1 ring-zinc-200 opacity-0 transition group-hover:opacity-100"
            >
              Copy
            </button>
          </div>
        );
      },
    }),
    [],
  );

  const renderMarkdown = (content: string) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={markdownComponents}
    >
      {normalizeMathDelimiters(content)}
    </ReactMarkdown>
  );

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

  const handleRegisterModel = async () => {
    if (!modelUrl.trim()) {
      setRegisterStatus("Paste a Hugging Face link to a GGUF file.");
      return;
    }
    setIsRegistering(true);
    setRegisterStatus("Starting download… this can take a while for large files.");
    try {
      const res = await fetch(`${apiBase}/models/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: modelUrl.trim(),
          name: modelName.trim() || undefined,
          description: modelDescription.trim() || undefined,
        }),
      });
      const payload = await res.json();
      if (!res.ok) {
        const message =
          payload?.detail?.detail ||
          payload?.detail?.message ||
          payload?.detail ||
          payload?.message ||
          "Model registration failed";
        throw new Error(message);
      }
      await mutateManifest();
      setRegisterStatus(
        `Added ${payload.model} on port ${payload.port}. Start it to chat.`,
      );
      setSelectedModel(payload.model);
      setModelUrl("");
      setModelName("");
      setModelDescription("");
    } catch (err) {
      setRegisterStatus(
        err instanceof Error ? err.message : "Failed to register model",
      );
    } finally {
      setIsRegistering(false);
    }
  };

  const handleCardKeydown = (
    e: React.KeyboardEvent<HTMLDivElement>,
    modelName: string,
  ) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedModel(modelName);
    }
  };

  const toggleExpandMessage = (idx: number) => {
    setExpandedMessages((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const toggleRawMessage = (idx: number) => {
    setShowRawMessages((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const handleDeleteModel = async (name: string) => {
    const confirmed = window.confirm(
      `Delete model '${name}' from the list? (This will not delete the file on disk.)`,
    );
    if (!confirmed) return;
    await fetch(`${apiBase}/models/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });
    await mutateManifest();
    if (selectedModel === name) {
      const remaining = Object.keys((manifest?.models ?? {})).filter((m) => m !== name);
      setSelectedModel(remaining[0] ?? null);
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
    const startTime = performance.now();
    const modelForMeta = selectedModel;
    const rawCollector: string[] = [];

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
            messages: [
              { role: "system", content: SYSTEM_GUARDRAIL },
              ...nextHistory,
            ],
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
            rawCollector.push(delta);
            assistantText += delta;
            setStreamingReply(
              applyAggressiveClean(assistantText, selectedModel),
            );
          } catch {
            continue;
          }
        }
      }

      const finalContent = applyAggressiveClean(assistantText, selectedModel);
      const durationMs = Math.round(performance.now() - startTime);
      const tokens = estimateTokens(finalContent);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: finalContent,
          meta: {
            model: modelForMeta ?? undefined,
            durationMs,
            tokens,
            raw: rawCollector.join(""),
          },
        },
      ]);
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
            <>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-4">
                  <p className="text-sm font-semibold text-zinc-800">
                    Add a model from Hugging Face
                  </p>
                  <p className="mt-1 text-xs text-zinc-500">
                    Paste a direct GGUF link (https://huggingface.co/.../resolve/main/*.gguf).
                  </p>
                  <div className="mt-3 space-y-2">
                    <input
                      value={modelUrl}
                      onChange={(e) => setModelUrl(e.target.value)}
                      className="w-full rounded-lg border border-zinc-200 px-3 py-2 text-sm"
                      placeholder="https://huggingface.co/..."
                    />
                    <div className="grid gap-2 md:grid-cols-2">
                      <input
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        className="w-full rounded-lg border border-zinc-200 px-3 py-2 text-sm"
                        placeholder="Name (optional)"
                      />
                      <input
                        value={modelDescription}
                        onChange={(e) => setModelDescription(e.target.value)}
                        className="w-full rounded-lg border border-zinc-200 px-3 py-2 text-sm"
                        placeholder="Description (optional)"
                      />
                    </div>
                    <button
                      className="w-full rounded-lg bg-zinc-900 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
                      disabled={isRegistering}
                      onClick={handleRegisterModel}
                    >
                      {isRegistering ? "Downloading…" : "Download & Register"}
                    </button>
                    {registerProgress && registerProgress.status !== "idle" && (
                      <div className="space-y-1 text-xs text-zinc-600">
                        <div className="flex items-center justify-between">
                          <span>
                            {registerProgress.message ??
                              (registerProgress.status === "completed"
                                ? "Download complete"
                                : registerProgress.status)}
                          </span>
                          {registerProgress.total_bytes ? (
                            <span>
                              {Math.round(
                                ((registerProgress.downloaded_bytes ?? 0) /
                                  registerProgress.total_bytes) *
                                  100,
                              )}
                              %
                            </span>
                          ) : null}
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200">
                          <div
                            className="h-full rounded-full bg-zinc-900 transition-[width]"
                            style={{
                              width: registerProgress.total_bytes
                                ? `${Math.min(
                                    100,
                                    Math.round(
                                      ((registerProgress.downloaded_bytes ?? 0) /
                                        registerProgress.total_bytes) *
                                        100,
                                    ),
                                  )}%`
                                : "35%",
                            }}
                          />
                        </div>
                      </div>
                    )}
                    {registerStatus && (
                      <p className="text-xs text-zinc-600">{registerStatus}</p>
                    )}
                  </div>
                </div>

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
                          <button
                            className="text-xs font-semibold text-rose-600 underline"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteModel(name);
                            }}
                            disabled={isRunning}
                            title={
                              isRunning
                                ? "Stop the model before deleting the entry"
                                : "Remove from manifest"
                            }
                          >
                            Delete
                          </button>
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
            </>
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
                  const raw = msg.meta?.raw;
                  const cleaned = applyAggressiveClean(
                    final ?? msg.content,
                    selectedModel,
                  );
                  const contentToRender = showRawMessages.has(idx)
                    ? raw || final || msg.content
                    : cleaned || final || msg.content;
                  const expanded = expandedMessages.has(idx);
                  const { preview, truncated } = buildPreview(contentToRender ?? "");
                  const showContent =
                    !expanded && !showRawMessages.has(idx) && truncated ? preview : contentToRender;
                  const isAssistant = msg.role === "assistant";
                  const meta = msg.meta ?? {};
                  const isLong = truncated || (showContent?.length ?? 0) > 1200;
                  return (
                    <div
                      key={`${msg.role}-${idx}`}
                      className={`rounded-xl border p-4 text-sm leading-6 ${
                        isAssistant ? "bg-zinc-50" : "bg-white"
                      }`}
                    >
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span
                          className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                            isAssistant
                              ? "bg-sky-100 text-sky-700"
                              : "bg-zinc-100 text-zinc-700"
                          }`}
                        >
                          {prettyRole}
                        </span>
                        {isAssistant && (
                          <div className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
                            {meta.model && (
                              <span className="rounded-full bg-white px-2 py-0.5 ring-1 ring-zinc-200">
                                {meta.model}
                              </span>
                            )}
                            {typeof meta.durationMs === "number" && (
                              <span className="rounded-full bg-white px-2 py-0.5 ring-1 ring-zinc-200">
                                {meta.durationMs} ms
                              </span>
                            )}
                            {typeof meta.tokens === "number" && (
                              <span className="rounded-full bg-white px-2 py-0.5 ring-1 ring-zinc-200">
                                ~{meta.tokens} tokens
                              </span>
                            )}
                            {raw && (
                              <button
                                className="rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-sky-700 ring-1 ring-sky-200"
                                onClick={() => toggleRawMessage(idx)}
                              >
                                {showRawMessages.has(idx) ? "View cleaned" : "View raw"}
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                      {analysis && (
                        <details className="mb-2 rounded-lg border border-zinc-200 bg-white p-3 text-xs text-zinc-700">
                          <summary className="cursor-pointer select-none text-zinc-700">
                            Reasoning trace
                          </summary>
                          <div className="prose prose-xs max-w-none text-zinc-800">
                            {renderMarkdown(analysis)}
                          </div>
                        </details>
                      )}
                      <div
                        className={`prose prose-sm max-w-none text-zinc-800 ${
                          !expanded && isLong ? "relative max-h-64 overflow-hidden" : ""
                        }`}
                      >
                        {renderMarkdown(showContent)}
                        {!expanded && isLong && (
                          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-zinc-50 to-transparent" />
                        )}
                      </div>
                      {isLong && (
                        <button
                          className="mt-2 text-xs font-semibold text-sky-700"
                          onClick={() => toggleExpandMessage(idx)}
                        >
                          {expanded ? "Show less" : "Show more"}
                        </button>
                      )}
                    </div>
                  );
                })}
                {streamingReply && (
                  <div className="rounded-xl border border-dashed border-zinc-300 bg-white p-4 text-sm leading-6">
                    <div className="mb-2 flex items-center gap-2">
                      <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-semibold text-sky-700">
                        Assistant
                      </span>
                      <span className="flex items-center gap-1 text-[11px] font-semibold text-sky-700">
                        <span className="h-2 w-2 animate-pulse rounded-full bg-sky-500" />
                        Streaming…
                      </span>
                    </div>
                    <div className="prose prose-sm max-w-none text-zinc-800">
                      {renderMarkdown(
                        formatAssistantContent(streamingReply, selectedModel),
                      )}
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
