"use client";

import { FormEvent, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ChatResponse = {
  response: string;
  detected_language: string;
  sentiment: string;
  sentiment_confidence: number;
  intent: string;
  intent_route: string;
  intent_confidence: number;
  priority_flag: boolean;
  handling: string;
  retrieved_chunks: {
    score: number;
    instruction: string;
    response: string;
    intent: string;
    category: string;
  }[];
};

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debug, setDebug] = useState<ChatResponse | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    setTimeout(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);
  };

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setError(null);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    setDebug(null);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Server returned ${res.status}: ${detail}`);
      }

      const data: ChatResponse = await res.json();
      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.response },
      ]);
      setDebug(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong connecting to the chatbot."
      );
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }

  return (
    <div className="flex h-screen flex-col bg-zinc-50">
      <header className="border-b border-zinc-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-zinc-900">
          E-commerce Support Chatbot
        </h1>
        <p className="text-sm text-zinc-500">
          RAG-powered assistant — local models + Groq generation
        </p>
      </header>

      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center text-zinc-400">
              Ask me anything about your orders, shipping, returns, or policies.
            </div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${
                m.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[75%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "border border-zinc-200 bg-white text-zinc-800"
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-500">
                Thinking...
              </div>
            </div>
          )}
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {debug && (
          <div className="border-t border-zinc-200 bg-zinc-900 px-6 py-3 text-xs text-zinc-300">
            <div className="mb-1 flex flex-wrap gap-2">
              <span className="rounded bg-zinc-700 px-2 py-0.5">
                lang: {debug.detected_language}
              </span>
              <span className="rounded bg-zinc-700 px-2 py-0.5">
                sentiment: {debug.sentiment} ({debug.sentiment_confidence.toFixed(2)})
              </span>
              <span className="rounded bg-zinc-700 px-2 py-0.5">
                intent: {debug.intent} → {debug.intent_route}
              </span>
              <span className="rounded bg-zinc-700 px-2 py-0.5">
                handling: {debug.handling}
              </span>
              <span
                className={`rounded px-2 py-0.5 ${
                  debug.priority_flag ? "bg-red-700" : "bg-emerald-800"
                }`}
              >
                priority: {debug.priority_flag ? "high" : "normal"}
              </span>
            </div>
            <details className="mt-1">
              <summary className="cursor-pointer">
                Retrieved chunks ({debug.retrieved_chunks?.length ?? 0})
              </summary>
              {debug.retrieved_chunks?.map((c, i) => (
                <div key={i} className="mt-2 rounded bg-zinc-800 p-2">
                  <div className="text-zinc-400">
                    [{c.category}] {c.intent} — score {c.score.toFixed(3)}
                  </div>
                  <div className="mt-1">{c.instruction}</div>
                </div>
              ))}
            </details>
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          className="flex gap-2 border-t border-zinc-200 bg-white px-6 py-4"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 rounded-full border border-zinc-300 px-4 py-2 text-sm outline-none focus:border-blue-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-full bg-blue-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
