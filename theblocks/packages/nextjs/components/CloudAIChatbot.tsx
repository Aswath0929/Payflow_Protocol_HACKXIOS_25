"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║  PAYFLOW CLOUD AI CHATBOT - SECURITY ISOLATED WEB2 LAYER                              ║
 * ╠══════════════════════════════════════════════════════════════════════════════════════╣
 * ║  Built for Hackxios 2K25 - PayPal & Visa Track                                        ║
 * ║                                                                                       ║
 * ║  🌐 WEB2 LAYER: Cloud AI API (Internet-facing, isolated from blockchain)             ║
 * ║  🔒 WEB3 LAYER: Local Qwen3:8B for Fraud Detection (Air-gapped, secure)              ║
 * ║                                                                                       ║
 * ║  SECURITY ARCHITECTURE:                                                               ║
 * ║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
 * ║  │  ✅ Chatbot uses CLOUD API → No blockchain data exposed                         │  ║
 * ║  │  ✅ Fraud detection runs LOCALLY → All transaction data stays on-device         │  ║
 * ║  │  ✅ Full 8GB VRAM for Qwen3:8B → Maximum GPU performance                         │  ║
 * ║  │  ✅ Complete security isolation → No attack vector between layers               │  ║
 * ║  └────────────────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                                       ║
 * ║  SUPPORTED PROVIDERS:                                                                 ║
 * ║  • OpenRouter (FREE Gemini, Llama, Qwen models)                                      ║
 * ║  • Google Gemini (Fast & Multimodal)                                                  ║
 * ║  • Anthropic Claude (Best reasoning)                                                  ║
 * ║  • Groq (Ultra-fast inference)                                                        ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 */

import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowPathIcon,
  ChatBubbleLeftRightIcon,
  CheckCircleIcon,
  ClipboardIcon,
  CloudIcon,
  Cog6ToothIcon,
  DocumentArrowDownIcon,
  ExclamationTriangleIcon,
  GlobeAltIcon,
  HandThumbDownIcon,
  HandThumbUpIcon,
  MicrophoneIcon,
  PaperAirplaneIcon,
  ShieldCheckIcon,
  SparklesIcon,
  StopIcon,
  TrashIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import {
  CheckCircleIcon as CheckCircleSolid,
  HandThumbDownIcon as ThumbDownSolid,
  HandThumbUpIcon as ThumbUpSolid,
} from "@heroicons/react/24/solid";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  responseTime?: number;
  reaction?: "like" | "dislike" | null;
  isError?: boolean;
  suggestions?: string[];
  provider?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

type AIProvider = "openrouter" | "google" | "openai" | "anthropic" | "groq";

interface ProviderConfig {
  name: string;
  baseUrl: string;
  defaultModel: string;
  freeModels?: string[];
  requiresKey: boolean;
  icon: string;
}

interface CloudChatbotProps {
  walletAddress?: string;
  onActionRequest?: (action: string, params: Record<string, unknown>) => void;
  theme?: "light" | "dark" | "auto";
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS & CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const STORAGE_KEY = "payflow_cloud_chat_sessions";
const API_KEY_STORAGE = "payflow_cloud_api_key";
const MAX_HISTORY_MESSAGES = 10;
const MAX_SESSIONS = 20;

const PROVIDERS: Record<AIProvider, ProviderConfig> = {
  openrouter: {
    name: "OpenRouter (Free Models)",
    baseUrl: "https://openrouter.ai/api/v1",
    defaultModel: "google/gemini-2.0-flash-exp:free",
    freeModels: [
      "google/gemini-2.0-flash-exp:free",
      "meta-llama/llama-3.2-3b-instruct:free",
      "qwen/qwen-2.5-72b-instruct:free",
      "mistralai/mistral-7b-instruct:free",
    ],
    requiresKey: true,
    icon: "🌐",
  },
  google: {
    name: "Google Gemini",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    defaultModel: "gemini-1.5-flash",
    requiresKey: true,
    icon: "💎",
  },
  groq: {
    name: "Groq (Ultra Fast)",
    baseUrl: "https://api.groq.com/openai/v1",
    defaultModel: "llama-3.3-70b-versatile",
    requiresKey: true,
    icon: "⚡",
  },
  openai: {
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o-mini",
    requiresKey: true,
    icon: "🤖",
  },
  anthropic: {
    name: "Anthropic Claude",
    baseUrl: "https://api.anthropic.com/v1",
    defaultModel: "claude-3-haiku-20240307",
    requiresKey: true,
    icon: "🧠",
  },
};

// PayFlow System Prompt
const SYSTEM_PROMPT = `You are PayFlow AI Assistant, a SPECIALIZED expert ONLY on the PayFlow Protocol.

⚠️ CRITICAL RULES:
1. ONLY answer questions about PayFlow Protocol
2. If a question is NOT about PayFlow, respond: "I specialize in PayFlow Protocol. I can help with payments, compliance, escrow, oracles, fraud detection, and gasless transfers. What would you like to know?"
3. Be concise but thorough. Use markdown formatting.
4. Reference specific contract functions when applicable.

PAYFLOW KNOWLEDGE BASE:

**Smart Contracts:**
- PayFlowCore.sol: createPayment(), executePayment(), settleWithFX(), cancelPayment()
- ComplianceEngine.sol: 5-tier KYC (NONE→BASIC→STANDARD→ENHANCED→INSTITUTIONAL)
- SmartEscrow.sol: TIME_BASED, APPROVAL, ORACLE, MULTI_SIG release types
- OracleAggregator.sol: Chainlink (60%) + Pyth (40%) weighted consensus
- FraudOracle.sol: AI risk scoring 0-100, ECDSA-signed verdicts
- PayFlowPaymaster.sol: ERC-4337 gasless transactions

**Compliance Tiers:**
- NONE: $1K/day, $5K/month | BASIC: $10K/day, $50K/month
- STANDARD: $100K/day, $500K/month | ENHANCED: $1M/day, $5M/month
- INSTITUTIONAL: Unlimited

**AI Fraud Detection:**
- Expert AI Oracle v3.0 with 5-Model Neural Ensemble
- Models: Deep MLP, Gradient Boost, Graph Attention, Temporal LSTM, Isolation Forest
- Risk Levels: SAFE (0-20), LOW (21-40), MEDIUM (41-60), HIGH (61-80), CRITICAL (81-100)
- Runs on LOCAL GPU (air-gapped from this chat for security)

**Tokens:** PYUSD, USDC, DAI, USDT`;

// Utility functions
const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// ═══════════════════════════════════════════════════════════════════════════════
// STORAGE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

const loadSessions = (): ChatSession[] => {
  if (typeof window === "undefined") return [];
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];
    const sessions = JSON.parse(data) as ChatSession[];
    return sessions.map(s => ({
      ...s,
      createdAt: new Date(s.createdAt),
      updatedAt: new Date(s.updatedAt),
      messages: s.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) })),
    }));
  } catch {
    return [];
  }
};

const saveSessions = (sessions: ChatSession[]): void => {
  if (typeof window === "undefined") return;
  try {
    const trimmed = sessions.slice(0, MAX_SESSIONS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch {
    console.warn("Failed to save chat sessions");
  }
};

const loadApiKey = (): { provider: AIProvider; key: string } | null => {
  if (typeof window === "undefined") return null;
  try {
    const data = localStorage.getItem(API_KEY_STORAGE);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
};

const saveApiKey = (provider: AIProvider, key: string): void => {
  if (typeof window === "undefined") return;
  localStorage.setItem(API_KEY_STORAGE, JSON.stringify({ provider, key }));
};

// ═══════════════════════════════════════════════════════════════════════════════
// MARKDOWN RENDERER
// ═══════════════════════════════════════════════════════════════════════════════

const MarkdownRenderer = memo(({ content }: { content: string }) => {
  const renderMarkdown = useCallback((text: string): string => {
    return (
      text
        .replace(
          /```(\w+)?\n([\s\S]*?)```/g,
          (_, lang, code) =>
            `<pre class="bg-base-300 rounded-lg p-3 my-2 overflow-x-auto text-sm"><code class="language-${lang || "text"}">${code.trim()}</code></pre>`,
        )
        .replace(/`([^`]+)`/g, '<code class="bg-base-300 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>')
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/^### (.+)$/gm, '<h4 class="font-bold text-base mt-3 mb-1">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 class="font-bold text-lg mt-3 mb-1">$1</h3>')
        .replace(/^[•\-\*] (.+)$/gm, '<li class="ml-4">$1</li>')
        .replace(/^(\d+)\. (.+)$/gm, '<li class="ml-4"><span class="font-mono text-primary">$1.</span> $2</li>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-primary underline" target="_blank">$1</a>')
        .replace(/\n/g, "<br />")
    );
  }, []);

  return <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />;
});
MarkdownRenderer.displayName = "MarkdownRenderer";

// ═══════════════════════════════════════════════════════════════════════════════
// MESSAGE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface MessageProps {
  message: Message;
  onReaction: (id: string, reaction: "like" | "dislike" | null) => void;
  onCopy: (content: string) => void;
  isLatest: boolean;
}

const MessageComponent = memo(({ message, onReaction, onCopy, isLatest }: MessageProps) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  const handleCopy = () => {
    onCopy(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""} group`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? "bg-primary text-primary-content" : "bg-gradient-to-br from-blue-500 to-purple-600 text-white"
        }`}
      >
        {isUser ? "You" : <CloudIcon className="w-4 h-4" />}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block max-w-[85%] rounded-2xl px-4 py-3 ${
            isUser ? "bg-primary text-primary-content rounded-tr-none" : "bg-base-200 rounded-tl-none"
          } ${message.isError ? "border-2 border-error" : ""}`}
        >
          {message.isStreaming ? (
            <div className="flex items-center gap-2">
              <span className="loading loading-dots loading-sm"></span>
              <span className="text-sm opacity-70">Thinking...</span>
            </div>
          ) : (
            <MarkdownRenderer content={message.content} />
          )}
        </div>

        {/* Meta & Actions */}
        {isAssistant && !message.isStreaming && (
          <div className="flex items-center gap-2 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-xs opacity-50">
              {message.provider && <span className="mr-1">via {message.provider}</span>}
              {message.responseTime && `${(message.responseTime / 1000).toFixed(1)}s`}
            </span>

            {/* Reactions */}
            <button
              onClick={() => onReaction(message.id, message.reaction === "like" ? null : "like")}
              className={`btn btn-ghost btn-xs ${message.reaction === "like" ? "text-success" : ""}`}
            >
              {message.reaction === "like" ? <ThumbUpSolid className="w-3 h-3" /> : <HandThumbUpIcon className="w-3 h-3" />}
            </button>
            <button
              onClick={() => onReaction(message.id, message.reaction === "dislike" ? null : "dislike")}
              className={`btn btn-ghost btn-xs ${message.reaction === "dislike" ? "text-error" : ""}`}
            >
              {message.reaction === "dislike" ? (
                <ThumbDownSolid className="w-3 h-3" />
              ) : (
                <HandThumbDownIcon className="w-3 h-3" />
              )}
            </button>

            {/* Copy */}
            <button onClick={handleCopy} className="btn btn-ghost btn-xs">
              {copied ? <CheckCircleIcon className="w-3 h-3 text-success" /> : <ClipboardIcon className="w-3 h-3" />}
            </button>
          </div>
        )}

        {/* Suggestions */}
        {isAssistant && isLatest && message.suggestions && message.suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {message.suggestions.map((s, i) => (
              <button
                key={i}
                className="btn btn-xs btn-outline rounded-full"
                onClick={() => {
                  /* Will be handled by parent */
                }}
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});
MessageComponent.displayName = "MessageComponent";

// ═══════════════════════════════════════════════════════════════════════════════
// API KEY SETUP COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface ApiKeySetupProps {
  onComplete: (provider: AIProvider, key: string) => void;
}

const ApiKeySetup = ({ onComplete }: ApiKeySetupProps) => {
  const [selectedProvider, setSelectedProvider] = useState<AIProvider>("openrouter");
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");
  const [testing, setTesting] = useState(false);

  const testConnection = async () => {
    setTesting(true);
    setError("");

    try {
      const provider = PROVIDERS[selectedProvider];

      if (selectedProvider === "openrouter") {
        const response = await fetch(`${provider.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
            "HTTP-Referer": "https://payflow.protocol",
            "X-Title": "PayFlow Protocol",
          },
          body: JSON.stringify({
            model: provider.defaultModel,
            messages: [{ role: "user", content: "Hello" }],
            max_tokens: 10,
          }),
        });

        if (!response.ok) throw new Error("Invalid API key");
      } else if (selectedProvider === "google") {
        const response = await fetch(
          `${provider.baseUrl}/models/${provider.defaultModel}:generateContent?key=${apiKey}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [{ parts: [{ text: "Hello" }] }],
              generationConfig: { maxOutputTokens: 10 },
            }),
          },
        );

        if (!response.ok) throw new Error("Invalid API key");
      } else if (selectedProvider === "groq") {
        const response = await fetch(`${provider.baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            model: provider.defaultModel,
            messages: [{ role: "user", content: "Hello" }],
            max_tokens: 10,
          }),
        });

        if (!response.ok) throw new Error("Invalid API key");
      }

      saveApiKey(selectedProvider, apiKey);
      onComplete(selectedProvider, apiKey);
    } catch (err) {
      setError("Connection failed. Please check your API key.");
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="text-center">
        <CloudIcon className="w-16 h-16 mx-auto text-primary mb-4" />
        <h3 className="text-xl font-bold">Setup Cloud AI</h3>
        <p className="text-sm opacity-70 mt-2">
          The chatbot uses a cloud AI API, keeping it <strong>isolated from blockchain data</strong> for security.
        </p>
      </div>

      <div className="bg-base-200 rounded-lg p-4">
        <div className="flex items-center gap-2 text-success mb-2">
          <ShieldCheckIcon className="w-5 h-5" />
          <span className="font-semibold">Security Architecture</span>
        </div>
        <ul className="text-sm space-y-1 opacity-80">
          <li>• Chatbot → Cloud API (no blockchain access)</li>
          <li>• Fraud Detection → Local GPU (air-gapped)</li>
          <li>• Full 8GB VRAM for Qwen3:8B inference</li>
        </ul>
      </div>

      <div className="space-y-4">
        <div>
          <label className="label">
            <span className="label-text font-medium">Select Provider</span>
          </label>
          <div className="grid grid-cols-2 gap-2">
            {(Object.keys(PROVIDERS) as AIProvider[]).map(p => (
              <button
                key={p}
                onClick={() => setSelectedProvider(p)}
                className={`btn btn-sm ${selectedProvider === p ? "btn-primary" : "btn-outline"}`}
              >
                <span>{PROVIDERS[p].icon}</span>
                <span className="text-xs">{PROVIDERS[p].name.split(" ")[0]}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="label">
            <span className="label-text font-medium">API Key</span>
            {selectedProvider === "openrouter" && (
              <a
                href="https://openrouter.ai/keys"
                target="_blank"
                rel="noopener noreferrer"
                className="label-text-alt link link-primary"
              >
                Get free key →
              </a>
            )}
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder={`Enter your ${PROVIDERS[selectedProvider].name} API key`}
            className="input input-bordered w-full"
          />
        </div>

        {error && (
          <div className="alert alert-error text-sm">
            <ExclamationTriangleIcon className="w-4 h-4" />
            <span>{error}</span>
          </div>
        )}

        <button
          onClick={testConnection}
          disabled={!apiKey || testing}
          className="btn btn-primary w-full"
        >
          {testing ? (
            <>
              <span className="loading loading-spinner loading-sm"></span>
              Testing...
            </>
          ) : (
            <>
              <GlobeAltIcon className="w-4 h-4" />
              Connect
            </>
          )}
        </button>
      </div>

      <div className="text-xs text-center opacity-50">
        {selectedProvider === "openrouter" && "OpenRouter offers FREE models like Gemini, Llama, and Qwen"}
        {selectedProvider === "groq" && "Groq provides ultra-fast inference (100+ tokens/sec)"}
        {selectedProvider === "google" && "Gemini 1.5 Flash is fast and capable"}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN CLOUD CHATBOT COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const CloudAIChatbot: React.FC<CloudChatbotProps> = ({
  walletAddress,
  onActionRequest,
  theme = "dark",
}) => {
  // ═══════════════════════════════════════════════════════════════════════════
  // STATE
  // ═══════════════════════════════════════════════════════════════════════════
  const [isOpen, setIsOpen] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [apiConfig, setApiConfig] = useState<{ provider: AIProvider; key: string } | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Current session
  const currentSession = useMemo(
    () => sessions.find(s => s.id === currentSessionId),
    [sessions, currentSessionId],
  );
  const messages = currentSession?.messages || [];

  // ═══════════════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    // Load API config
    const savedConfig = loadApiKey();
    if (savedConfig) {
      setApiConfig(savedConfig);
    }

    // Load sessions
    const loaded = loadSessions();
    if (loaded.length > 0) {
      setSessions(loaded);
      setCurrentSessionId(loaded[0].id);
    }
  }, []);

  // Save sessions
  useEffect(() => {
    if (sessions.length > 0) {
      saveSessions(sessions);
    }
  }, [sessions]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // Initial greeting
  useEffect(() => {
    if (isOpen && apiConfig && currentSession && currentSession.messages.length === 0) {
      const greeting: Message = {
        id: "greeting",
        role: "assistant",
        content: `👋 Welcome to **PayFlow AI Assistant**!

🌐 **Powered by:** ${PROVIDERS[apiConfig.provider].name}
🔒 **Security:** This chatbot is isolated from blockchain data

✅ **Why Cloud AI?**
- Chatbot uses cloud → No access to transaction data
- Fraud detection runs locally → Full GPU performance
- Complete security isolation between layers

I can help you with:
• 💸 **Payments** - Create, execute, track transfers
• 🛡️ **Compliance** - KYC tiers and limits
• 🔒 **Smart Escrow** - Conditional payments
• 🔮 **Oracles** - Price feeds and FX rates
• 🧠 **Fraud Detection** - AI risk analysis
• ⛽ **Gasless Transfers** - Send without ETH

What would you like to know about PayFlow?`,
        timestamp: new Date(),
        suggestions: ["How do gasless transfers work?", "What are the KYC tiers?", "Explain fraud detection"],
        provider: apiConfig.provider,
      };

      setSessions(prev =>
        prev.map(s => {
          if (s.id !== currentSessionId) return s;
          return { ...s, messages: [greeting], updatedAt: new Date() };
        }),
      );
    }
  }, [isOpen, apiConfig, currentSession?.id, currentSession?.messages.length, currentSessionId]);

  // ═══════════════════════════════════════════════════════════════════════════
  // SESSION MANAGEMENT
  // ═══════════════════════════════════════════════════════════════════════════

  const createNewSession = useCallback(() => {
    const newSession: ChatSession = {
      id: generateId(),
      title: "New Chat",
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    setShowHistory(false);
  }, []);

  const updateSession = useCallback((sessionId: string, newMessages: Message[]) => {
    setSessions(prev =>
      prev.map(s => {
        if (s.id !== sessionId) return s;
        const firstUserMsg = newMessages.find(m => m.role === "user");
        const newTitle =
          s.title === "New Chat" && firstUserMsg
            ? firstUserMsg.content.slice(0, 40) + (firstUserMsg.content.length > 40 ? "..." : "")
            : s.title;
        return { ...s, title: newTitle, messages: newMessages, updatedAt: new Date() };
      }),
    );
  }, []);

  const deleteSession = useCallback(
    (sessionId: string) => {
      setSessions(prev => {
        const filtered = prev.filter(s => s.id !== sessionId);
        if (currentSessionId === sessionId) {
          setCurrentSessionId(filtered.length > 0 ? filtered[0].id : null);
        }
        return filtered;
      });
    },
    [currentSessionId],
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // CLOUD AI STREAMING
  // ═══════════════════════════════════════════════════════════════════════════

  const callCloudAI = useCallback(
    async (userInput: string, history: Message[]): Promise<{ response: string; responseTime: number }> => {
      if (!apiConfig) throw new Error("No API configured");

      const startTime = Date.now();
      const provider = PROVIDERS[apiConfig.provider];
      const controller = new AbortController();
      setAbortController(controller);

      // Build messages array with history
      const historyMessages = history
        .filter(m => m.role !== "system" && !m.isStreaming)
        .slice(-MAX_HISTORY_MESSAGES)
        .map(m => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        }));

      let fullResponse = "";

      try {
        if (apiConfig.provider === "openrouter" || apiConfig.provider === "openai" || apiConfig.provider === "groq") {
          // OpenAI-compatible API
          const response = await fetch(`${provider.baseUrl}/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${apiConfig.key}`,
              ...(apiConfig.provider === "openrouter" && {
                "HTTP-Referer": "https://payflow.protocol",
                "X-Title": "PayFlow Protocol",
              }),
            },
            body: JSON.stringify({
              model: provider.defaultModel,
              messages: [{ role: "system", content: SYSTEM_PROMPT }, ...historyMessages, { role: "user", content: userInput }],
              max_tokens: 1024,
              temperature: 0.7,
              stream: true,
            }),
            signal: controller.signal,
          });

          if (!response.ok) throw new Error(`API Error: ${response.status}`);

          const reader = response.body?.getReader();
          if (!reader) throw new Error("No response body");

          const decoder = new TextDecoder();

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split("\n").filter(line => line.startsWith("data: "));

            for (const line of lines) {
              const data = line.replace("data: ", "").trim();
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                const token = parsed.choices[0]?.delta?.content || "";
                if (token) {
                  fullResponse += token;
                  setStreamingText(fullResponse);
                }
              } catch {
                // Skip malformed
              }
            }
          }
        } else if (apiConfig.provider === "google") {
          // Google Gemini API
          const url = `${provider.baseUrl}/models/${provider.defaultModel}:streamGenerateContent?alt=sse&key=${apiConfig.key}`;

          const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [
                { role: "user", parts: [{ text: SYSTEM_PROMPT }] },
                { role: "model", parts: [{ text: "I understand. I am PayFlow AI Assistant." }] },
                ...historyMessages.map(m => ({
                  role: m.role === "assistant" ? "model" : "user",
                  parts: [{ text: m.content }],
                })),
                { role: "user", parts: [{ text: userInput }] },
              ],
              generationConfig: { maxOutputTokens: 1024, temperature: 0.7 },
            }),
            signal: controller.signal,
          });

          if (!response.ok) throw new Error(`Gemini Error: ${response.status}`);

          const reader = response.body?.getReader();
          if (!reader) throw new Error("No response body");

          const decoder = new TextDecoder();

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split("\n").filter(line => line.startsWith("data: "));

            for (const line of lines) {
              try {
                const data = JSON.parse(line.replace("data: ", ""));
                const token = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
                if (token) {
                  fullResponse += token;
                  setStreamingText(fullResponse);
                }
              } catch {
                // Skip malformed
              }
            }
          }
        }

        return { response: fullResponse, responseTime: Date.now() - startTime };
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          return { response: fullResponse || "Response cancelled.", responseTime: Date.now() - startTime };
        }
        throw error;
      }
    },
    [apiConfig],
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // SEND MESSAGE
  // ═══════════════════════════════════════════════════════════════════════════

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading || !currentSessionId || !apiConfig) return;

    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    const assistantPlaceholder: Message = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
      provider: apiConfig.provider,
    };

    const updatedMessages = [...messages, userMessage, assistantPlaceholder];
    updateSession(currentSessionId, updatedMessages);
    setInput("");
    setIsLoading(true);
    setStreamingText("");

    try {
      const { response, responseTime } = await callCloudAI(userMessage.content, messages);

      // Parse suggestions
      let finalResponse = response;
      let suggestions: string[] = [];

      const suggestionMatch = response.match(/---\s*\n?(.*?)$/s);
      if (suggestionMatch) {
        finalResponse = response.replace(/---\s*\n?.*$/s, "").trim();
        suggestions = suggestionMatch[1]
          .split("|")
          .map(s => s.trim().replace(/^\[|\]$/g, ""))
          .filter(Boolean);
      }

      const finalMessage: Message = {
        ...assistantPlaceholder,
        content: finalResponse || response,
        isStreaming: false,
        responseTime,
        suggestions: suggestions.length > 0 ? suggestions : undefined,
      };

      const finalMessages = updatedMessages.map(m => (m.id === assistantPlaceholder.id ? finalMessage : m));
      updateSession(currentSessionId, finalMessages);
    } catch (error) {
      const errorMessage: Message = {
        ...assistantPlaceholder,
        content: `Error: ${(error as Error).message}. Please try again.`,
        isStreaming: false,
        isError: true,
      };

      const finalMessages = updatedMessages.map(m => (m.id === assistantPlaceholder.id ? errorMessage : m));
      updateSession(currentSessionId, finalMessages);
    } finally {
      setIsLoading(false);
      setStreamingText("");
      setAbortController(null);
    }
  }, [input, isLoading, currentSessionId, apiConfig, messages, updateSession, callCloudAI]);

  // Stop generation
  const stopGeneration = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
  }, [abortController]);

  // Reactions
  const handleReaction = useCallback(
    (id: string, reaction: "like" | "dislike" | null) => {
      if (!currentSessionId) return;
      const updatedMessages = messages.map(m => (m.id === id ? { ...m, reaction } : m));
      updateSession(currentSessionId, updatedMessages);
    },
    [currentSessionId, messages, updateSession],
  );

  // Copy
  const handleCopy = useCallback((content: string) => {
    navigator.clipboard.writeText(content);
  }, []);

  // Handle API setup complete
  const handleApiSetupComplete = useCallback(
    (provider: AIProvider, key: string) => {
      setApiConfig({ provider, key });
      setShowSettings(false);

      // Create initial session if none exists
      if (sessions.length === 0) {
        createNewSession();
      }
    },
    [sessions.length, createNewSession],
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        sendMessage();
      } else if (e.key === "Escape" && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sendMessage, isOpen]);

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => {
          setIsOpen(true);
          if (sessions.length === 0) createNewSession();
        }}
        className="fixed bottom-4 right-4 btn btn-circle btn-lg btn-primary shadow-2xl hover:scale-110 transition-transform z-50"
      >
        <CloudIcon className="w-6 h-6" />
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-20 right-4 w-[420px] h-[600px] bg-base-100 rounded-2xl shadow-2xl border border-base-300 flex flex-col overflow-hidden z-50">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
            <div className="flex items-center gap-2">
              <CloudIcon className="w-6 h-6" />
              <div>
                <h3 className="font-bold">PayFlow AI</h3>
                <span className="text-xs opacity-80">
                  {apiConfig ? `via ${PROVIDERS[apiConfig.provider].icon} ${PROVIDERS[apiConfig.provider].name.split(" ")[0]}` : "Cloud AI"}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={() => setShowSettings(true)} className="btn btn-ghost btn-sm btn-circle text-white">
                <Cog6ToothIcon className="w-5 h-5" />
              </button>
              <button onClick={() => setIsOpen(false)} className="btn btn-ghost btn-sm btn-circle text-white">
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Security Badge */}
          <div className="px-3 py-1.5 bg-success/10 border-b border-success/20 flex items-center gap-2 text-xs">
            <ShieldCheckIcon className="w-4 h-4 text-success" />
            <span className="text-success">Isolated from blockchain data • Cloud API • Fraud detection runs locally</span>
          </div>

          {/* Main Content */}
          {!apiConfig || showSettings ? (
            <ApiKeySetup onComplete={handleApiSetupComplete} />
          ) : (
            <>
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, i) => (
                  <MessageComponent
                    key={msg.id}
                    message={{ ...msg, content: msg.isStreaming ? streamingText || "Thinking..." : msg.content }}
                    onReaction={handleReaction}
                    onCopy={handleCopy}
                    isLatest={i === messages.length - 1}
                  />
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-3 border-t border-base-300">
                <div className="flex gap-2 items-end">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder="Ask about PayFlow..."
                    className="textarea textarea-bordered flex-1 min-h-[44px] max-h-[120px] resize-none"
                    rows={1}
                    onKeyDown={e => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                  />
                  {isLoading ? (
                    <button onClick={stopGeneration} className="btn btn-circle btn-error">
                      <StopIcon className="w-5 h-5" />
                    </button>
                  ) : (
                    <button onClick={sendMessage} disabled={!input.trim()} className="btn btn-circle btn-primary">
                      <PaperAirplaneIcon className="w-5 h-5" />
                    </button>
                  )}
                </div>
                <div className="text-xs text-center mt-2 opacity-50">
                  Ctrl+Enter to send • Powered by Cloud AI
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};

export default CloudAIChatbot;
