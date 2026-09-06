import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Terminal, Sparkles, ArrowRight, CornerDownLeft, Loader2, AlertCircle, User, Clock, Shield } from 'lucide-react';
import { playHapticSound } from '../../lib/utils';
import { INITIAL_EVENTS, SEATS_DATA } from '../../data/demoData';

interface SurveillanceAIQueryConsoleProps {
  onSelectCandidate?: (seatId: string) => void;
}

interface AIMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ParsedMatch {
  seatId: string;
  candidateName: string;
  severity?: string;
  detail: string;
}

// Parse the AI response to extract seat/candidate references for quick-jump buttons
function extractMatches(text: string): ParsedMatch[] {
  const matches: ParsedMatch[] = [];
  const seen = new Set<string>();
  // Match patterns like "A06", "Seat A06", "Candidate #6", "B04", etc.
  const seatPattern = /\b([AB]\d{2})\b/g;
  let m: RegExpExecArray | null;
  while ((m = seatPattern.exec(text)) !== null) {
    const seatId = m[1];
    if (!seen.has(seatId)) {
      seen.add(seatId);
      const seat = SEATS_DATA.find(s => s.id === seatId);
      const event = INITIAL_EVENTS.find(e => e.seatId === seatId);
      matches.push({
        seatId,
        candidateName: seat?.candidateName ?? `Seat ${seatId}`,
        severity: event?.severity,
        detail: event?.behaviorDisplayTitle ?? 'Behavioural signal detected',
      });
    }
  }
  return matches.slice(0, 4);
}

// Build a compact context string from demo data for the LLM system prompt
function buildContext(): string {
  const eventSummaries = INITIAL_EVENTS.slice(0, 10).map(e =>
    `[${e.id}] Seat ${e.seatId} (${e.candidateName}): ${e.behaviorDisplayTitle} — severity=${e.severity}, confidence=${(e.aiConfidence * 100).toFixed(0)}%, cam=${e.cameraId}, duration=${e.persistenceSec}s, priority=${e.reviewPriority}`
  ).join('\n');

  const seatSummaries = SEATS_DATA.slice(0, 20).map(s =>
    `Seat ${s.id}: ${s.candidateName}, state=${s.state}, headEvidence=${s.telemetry.headEvidence}, hands=${s.telemetry.handsEvidence}`
  ).join('\n');

  return `You are the NETRA-4D Surveillance AI assistant for an exam computer lab with 20 terminals (A01–A10, B01–B10) monitored by 4 cameras.\n\nLive Event Log (latest 10 events):\n${eventSummaries}\n\nCandidate Seat Status:\n${seatSummaries}\n\nAnswer natural language queries about these candidates and events concisely. When relevant, mention specific seat IDs like A06 or B04. Be analytical, precise, and professional. Keep answers under 200 words.`;
}

async function callGroq(messages: AIMessage[]): Promise<string> {
  // 1. Primary: Query via FastAPI backend endpoint (proxied automatically via /api)
  try {
    const response = await fetch('/api/telemetry/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages }),
    });

    if (response.ok) {
      const data = await response.json();
      if (data.response) return data.response;
      if (data.choices?.[0]?.message?.content) return data.choices[0].message.content;
    } else {
      const errDetail = await response.text();
      console.warn('Backend query endpoint error:', response.status, errDetail);
    }
  } catch (backendErr) {
    console.warn('Backend query endpoint unreachable, checking client key:', backendErr);
  }

  // 2. Secondary fallback if VITE_GROQ_API_KEY is configured in frontend env
  const apiKey = import.meta.env.VITE_GROQ_API_KEY as string | undefined;
  if (!apiKey) {
    throw new Error('Surveillance AI Query service is unavailable. Please ensure the backend is running on port 8000.');
  }

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'qwen/qwen3.8-27b',
      messages: [
        { role: 'system', content: buildContext() },
        ...messages,
      ],
      max_tokens: 300,
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Groq API error ${response.status}: ${err}`);
  }

  const data = await response.json() as { choices: { message: { content: string } }[] };
  return data.choices[0]?.message?.content ?? '(no response)';
}

const QUICK_PROMPTS = [
  'Which candidates have the highest risk right now?',
  'Show seats with escalated severity events',
  'List candidates with hand evidence issues',
  'Any peer interaction incidents detected?',
  'Who has the lowest AI confidence score?',
];

export const SurveillanceAIQueryConsole: React.FC<SurveillanceAIQueryConsoleProps> = ({
  onSelectCandidate,
}) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (text: string) => {
    const q = text.trim();
    if (!q || loading) return;
    playHapticSound('snap');
    setQuery('');
    setError(null);

    const userMsg: AIMessage = { role: 'user', content: q };
    const history = [...messages, userMsg];
    setMessages(history);
    setLoading(true);

    try {
      const reply = await callGroq(history);
      setMessages([...history, { role: 'assistant', content: reply }]);
    } catch (e: unknown) {
      setError((e as Error).message ?? 'Unknown error');
      // Remove the optimistic user message on failure
      setMessages(messages);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div className="rounded-[2rem] bg-white border border-neutral-200/80 shadow-[0_20px_40px_rgba(0,0,0,0.035)] flex flex-col font-sans overflow-hidden">

      {/* Header */}
      <div className="px-7 pt-7 pb-5 border-b border-neutral-200/80 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-amber-50 text-amber-700 border border-amber-100">
            <Sparkles size={20} />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-lg text-neutral-950 tracking-tight">
              Natural Language Surveillance AI
            </h3>
            <p className="text-xs text-neutral-500 mt-0.5">
              Ask anything about candidates, events, or camera coverage
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono font-bold px-2.5 py-1 rounded-full bg-neutral-900 text-white tracking-widest">
          GROQ · LLAMA-3.3-70B
        </span>
      </div>

      {/* Conversation area */}
      <div className="flex-1 overflow-y-auto px-7 py-5 space-y-4 min-h-[260px] max-h-[420px]">
        {messages.length === 0 && !loading && (
          <div className="text-center py-8 space-y-2">
            <Shield size={32} className="mx-auto text-neutral-200" />
            <p className="text-sm font-semibold text-neutral-400">Ask the AI anything about your exam session</p>
            <p className="text-xs text-neutral-300">Queries are grounded in live candidate and event data</p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-xl bg-amber-50 border border-amber-100 flex items-center justify-center shrink-0 mt-0.5">
                  <Sparkles size={13} className="text-amber-600" />
                </div>
              )}
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-xs leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-neutral-900 text-white rounded-tr-md font-mono'
                  : 'bg-[#F5F4F0] text-neutral-800 rounded-tl-md'
              }`}>
                {msg.role === 'assistant' ? (
                  <>
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                    {/* Quick-jump buttons for referenced seats */}
                    {(() => {
                      const matches = extractMatches(msg.content);
                      if (matches.length === 0) return null;
                      return (
                        <div className="mt-3 pt-3 border-t border-neutral-200/60 flex flex-wrap gap-2">
                          {matches.map(m => (
                            <button
                              key={m.seatId}
                              onClick={() => { playHapticSound('click'); onSelectCandidate?.(m.seatId); }}
                              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-neutral-200 text-[10px] font-semibold text-neutral-700 hover:bg-neutral-50 transition-colors cursor-pointer"
                            >
                              <User size={9} />
                              {m.candidateName}
                              <ArrowRight size={9} className="text-neutral-400" />
                            </button>
                          ))}
                        </div>
                      );
                    })()}
                  </>
                ) : (
                  <span>{msg.content}</span>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-xl bg-neutral-900 flex items-center justify-center shrink-0 mt-0.5">
                  <User size={13} className="text-white" />
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
            <div className="w-7 h-7 rounded-xl bg-amber-50 border border-amber-100 flex items-center justify-center shrink-0">
              <Loader2 size={13} className="text-amber-600 animate-spin" />
            </div>
            <div className="bg-[#F5F4F0] rounded-2xl rounded-tl-md px-4 py-3 text-xs text-neutral-400 font-mono flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </motion.div>
        )}

        {error && (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Quick prompts */}
      <div className="px-7 py-3 border-t border-neutral-100 flex items-center gap-2 flex-wrap">
        <span className="text-[10px] font-mono font-bold text-neutral-400 uppercase tracking-wider shrink-0">Try:</span>
        {QUICK_PROMPTS.slice(0, 3).map((p, i) => (
          <button
            key={i}
            onClick={() => handleSend(p)}
            disabled={loading}
            className="px-2.5 py-1 rounded-lg bg-neutral-100 hover:bg-neutral-200 text-neutral-600 text-[11px] font-semibold transition-colors cursor-pointer disabled:opacity-40"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Input bar */}
      <div className="px-7 pb-7 pt-3">
        <div className="relative flex items-center gap-2">
          <Terminal size={14} className="absolute left-4 text-neutral-400 pointer-events-none" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && query) handleSend(query); }}
            disabled={loading}
            placeholder="Ask the AI about candidates, patterns, or events…"
            className="flex-1 bg-[#F5F4F0] border border-neutral-200 rounded-2xl pl-10 pr-4 py-3 text-xs text-neutral-900 placeholder-neutral-400 font-mono focus:outline-none focus:border-neutral-900 focus:bg-white transition-all shadow-inner disabled:opacity-50"
          />
          <button
            onClick={() => handleSend(query)}
            disabled={!query.trim() || loading}
            className="px-4 py-3 rounded-2xl bg-neutral-900 text-white text-xs font-mono font-bold flex items-center gap-2 hover:bg-black transition-colors disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed shrink-0"
          >
            {loading ? <Loader2 size={13} className="animate-spin" /> : <CornerDownLeft size={13} />}
            Execute
          </button>
        </div>
      </div>
    </div>
  );
};
