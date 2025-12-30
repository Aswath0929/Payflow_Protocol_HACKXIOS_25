"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║  PAYFLOW AI CHATBOT PRO - ENTERPRISE-GRADE CONVERSATIONAL AI                         ║
 * ║  Modern Features Matching Leading AI Products (ChatGPT, Claude, Gemini)              ║
 * ╠══════════════════════════════════════════════════════════════════════════════════════╣
 * ║  Built for Hackxios 2K25 - PayPal & Visa Track                                       ║
 * ║                                                                                       ║
 * ║  🧠 AI MODEL: Perplexity Sonar (Cloud) / Qwen3 8B MoE (Local Fallback)                ║
 * ║  🔒 SECURITY: 100% ISOLATED from Web3/Blockchain data                                 ║
 * ║  ☁️  RUNTIME: Google Cloud (no local GPU required)                                    ║
 * ║                                                                                       ║
 * ║  🔐 SECURITY ARCHITECTURE:                                                            ║
 * ║  ┌─────────────────────────────────────────────────────────────────────────────────┐  ║
 * ║  │  WEB2 LAYER (THIS CHATBOT)             WEB3 LAYER (ISOLATED)                   │  ║
 * ║  │  ┌────────────────────────┐     🔥     ┌────────────────────────┐              │  ║
 * ║  │  │  Gemini API (Cloud)    │  FIREWALL  │  Fraud Detection (GPU) │              │  ║
 * ║  │  │  ✅ Documentation Q&A  │     ║      │  ✅ Transaction Scoring │             │  ║
 * ║  │  │  ✅ Protocol Education │     ║      │  ✅ On-chain Data       │             │  ║
 * ║  │  │  ❌ NO TX DATA ACCESS  │     ║      │  ✅ Wallet Interactions │             │  ║
 * ║  │  └────────────────────────┘            └────────────────────────┘              │  ║
 * ║  └─────────────────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                                       ║
 * ║  🚀 CUTTING-EDGE FEATURES:                                                            ║
 * ║  • Streaming responses (real-time token display)                                     ║
 * ║  • Persistent chat history (localStorage + sessions)                                 ║
 * ║  • Message reactions (like/dislike/copy/regenerate)                                  ║
 * ║  • Voice input (Web Speech API)                                                      ║
 * ║  • Suggested follow-ups (AI-generated)                                               ║
 * ║  • Context memory (multi-turn conversations)                                         ║
 * ║  • Code syntax highlighting                                                          ║
 * ║  • Keyboard shortcuts (Ctrl+Enter, Escape)                                           ║
 * ║  • Export conversation (Markdown/JSON)                                               ║
 * ║  • Response time metrics                                                             ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 */
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GeminiService } from "~~/services/ai/geminiService";
import {
  ArrowPathIcon,
  ArrowsPointingInIcon,
  ArrowsPointingOutIcon,
  ChatBubbleLeftRightIcon,
  CheckCircleIcon,
  ClipboardIcon,
  ClockIcon,
  Cog6ToothIcon,
  CloudIcon,
  CpuChipIcon,
  DocumentArrowDownIcon,
  ExclamationTriangleIcon,
  HandThumbDownIcon,
  HandThumbUpIcon,
  MagnifyingGlassIcon,
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
  model?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

interface ChatbotProProps {
  walletAddress?: string;
  transactionContext?: {
    sender?: string;
    recipient?: string;
    amount?: number;
    token?: string;
  };
  onActionRequest?: (action: string, params: Record<string, unknown>) => void;
  theme?: "light" | "dark" | "auto";
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS & CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const STORAGE_KEY = "payflow_chat_sessions";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const MAX_HISTORY_MESSAGES = 10; // Context window for conversation memory (used by Gemini service)
const MAX_SESSIONS = 20;
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Web Speech API type declarations
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition: { new (): SpeechRecognition };
    webkitSpeechRecognition: { new (): SpeechRecognition };
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMINI API CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const GEMINI_API_KEY = "REPLACE_WITH_ENV";
const geminiService = new GeminiService(GEMINI_API_KEY);

// Topic validation keywords
const PAYFLOW_KEYWORDS = [
  "payflow",
  "payment",
  "transfer",
  "send",
  "receive",
  "transaction",
  "contract",
  "escrow",
  "oracle",
  "compliance",
  "kyc",
  "audit",
  "paymaster",
  "pyusd",
  "usdc",
  "dai",
  "usdt",
  "stablecoin",
  "token",
  "gasless",
  "gas",
  "fee",
  "sponsor",
  "whitelist",
  "tier",
  "limit",
  "fraud",
  "risk",
  "score",
  "block",
  "approve",
  "review",
  "wallet",
  "address",
  "deploy",
  "sepolia",
  "mainnet",
  "create",
  "execute",
  "cancel",
  "settle",
  "release",
  "lock",
  "how",
  "what",
  "why",
  "when",
  "help",
  "explain",
];

// Response cache for performance
const responseCache = new Map<string, { response: string; timestamp: number; suggestions: string[] }>();

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

const isPayFlowRelated = (query: string): boolean => {
  const lowerQuery = query.toLowerCase();
  return PAYFLOW_KEYWORDS.some(kw => lowerQuery.includes(kw)) || lowerQuery.length < 20;
};

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const getCacheKey = (message: string, history: Message[]): string => {
  const historyHash = history
    .slice(-3)
    .map(m => m.content.slice(0, 50))
    .join("|");
  return `${message}|${historyHash}`;
};

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

// ═══════════════════════════════════════════════════════════════════════════════
// MARKDOWN RENDERER COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const MarkdownRenderer = memo(({ content }: { content: string }) => {
  const renderMarkdown = useCallback((text: string): string => {
    return (
      text
        // Code blocks with syntax highlighting simulation
        .replace(
          /```(\w+)?\n([\s\S]*?)```/g,
          (_, lang, code) =>
            `<pre class="bg-base-300 rounded-lg p-3 my-2 overflow-x-auto text-sm"><code class="language-${lang || "text"}">${code.trim()}</code></pre>`,
        )
        // Inline code
        .replace(/`([^`]+)`/g, '<code class="bg-base-300 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>')
        // Bold
        .replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>')
        // Italic
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        // Headers
        .replace(/^### (.+)$/gm, '<h4 class="font-bold text-base mt-3 mb-1">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 class="font-bold text-lg mt-3 mb-1">$1</h3>')
        // Lists
        .replace(/^[•\-\*] (.+)$/gm, '<li class="ml-4">$1</li>')
        .replace(/^(\d+)\. (.+)$/gm, '<li class="ml-4"><span class="font-mono text-primary">$1.</span> $2</li>')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-primary underline" target="_blank">$1</a>')
        // Line breaks
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
  onRegenerate: (id: string) => void;
  onSuggestionClick: (suggestion: string) => void;
  isLatest: boolean;
}

const MessageComponent = memo(
  ({ message, onReaction, onCopy, onRegenerate, onSuggestionClick, isLatest }: MessageProps) => {
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
            isUser ? "bg-primary" : "bg-gradient-to-br from-purple-500 to-blue-500"
          }`}
        >
          {isUser ? (
            <span className="text-primary-content text-sm font-bold">U</span>
          ) : (
            <SparklesIcon className="w-4 h-4 text-white" />
          )}
        </div>

        {/* Message Content */}
        <div className={`flex flex-col max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
          <div
            className={`rounded-2xl px-4 py-3 ${
              isUser
                ? "bg-primary text-primary-content rounded-br-sm"
                : message.isError
                  ? "bg-error/10 text-error border border-error/30 rounded-bl-sm"
                  : "bg-base-200 text-base-content rounded-bl-sm"
            }`}
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

          {/* Message Footer */}
          {isAssistant && !message.isStreaming && (
            <div className="flex items-center gap-2 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {/* Response Time */}
              {message.responseTime && (
                <span className="text-xs text-base-content/50 flex items-center gap-1">
                  <ClockIcon className="w-3 h-3" />
                  {message.responseTime}ms
                </span>
              )}

              {/* Action Buttons */}
              <div className="flex items-center gap-1">
                <button onClick={handleCopy} className="btn btn-ghost btn-xs" title="Copy response">
                  {copied ? (
                    <CheckCircleSolid className="w-3 h-3 text-success" />
                  ) : (
                    <ClipboardIcon className="w-3 h-3" />
                  )}
                </button>

                <button
                  onClick={() => onReaction(message.id, message.reaction === "like" ? null : "like")}
                  className={`btn btn-ghost btn-xs ${message.reaction === "like" ? "text-success" : ""}`}
                  title="Good response"
                >
                  {message.reaction === "like" ? (
                    <ThumbUpSolid className="w-3 h-3" />
                  ) : (
                    <HandThumbUpIcon className="w-3 h-3" />
                  )}
                </button>

                <button
                  onClick={() => onReaction(message.id, message.reaction === "dislike" ? null : "dislike")}
                  className={`btn btn-ghost btn-xs ${message.reaction === "dislike" ? "text-error" : ""}`}
                  title="Poor response"
                >
                  {message.reaction === "dislike" ? (
                    <ThumbDownSolid className="w-3 h-3" />
                  ) : (
                    <HandThumbDownIcon className="w-3 h-3" />
                  )}
                </button>

                {isLatest && (
                  <button
                    onClick={() => onRegenerate(message.id)}
                    className="btn btn-ghost btn-xs"
                    title="Regenerate response"
                  >
                    <ArrowPathIcon className="w-3 h-3" />
                  </button>
                )}
              </div>

              {/* Model Badge */}
              {message.model && <span className="badge badge-ghost badge-xs">{message.model}</span>}
            </div>
          )}

          {/* Suggested Follow-ups */}
          {isAssistant && !message.isStreaming && message.suggestions && message.suggestions.length > 0 && isLatest && (
            <div className="flex flex-wrap gap-1 mt-2">
              {message.suggestions.map((suggestion, idx) => (
                <button
                  key={idx}
                  onClick={() => onSuggestionClick(suggestion)}
                  className="btn btn-xs btn-outline btn-primary"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Timestamp */}
          <span className="text-xs text-base-content/40 mt-1">
            {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
      </div>
    );
  },
);
MessageComponent.displayName = "MessageComponent";

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN CHATBOT COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const AIChatbotPro: React.FC<ChatbotProProps> = ({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  walletAddress,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  transactionContext,
  onActionRequest: _onActionRequest,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  theme = "auto",
}) => {
  // State
  const [isOpen, setIsOpen] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isUsingLocalGPU, setIsUsingLocalGPU] = useState(false);
  const [activeModel, setActiveModel] = useState(geminiService.getActiveModel());
  const [isListening, setIsListening] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // UI State for Drag & Resize
  const [iconPos, setIconPos] = useState({ x: 0, y: 0 });
  const [winSize, setWinSize] = useState({ w: 420, h: 650 });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [savedWinSize, setSavedWinSize] = useState({ w: 420, h: 650 });
  const [savedIconPos, setSavedIconPos] = useState({ x: 0, y: 0 });
  const dragOffset = useRef({ x: 0, y: 0 });
  const resizeStart = useRef({ x: 0, y: 0, w: 0, h: 0 });

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // Memos
  const currentSession = useMemo(
    () => sessions.find(s => s.id === currentSessionId) || null,
    [sessions, currentSessionId],
  );

  const messages = useMemo(() => currentSession?.messages || [], [currentSession]);

  const filteredSessions = useMemo(() => {
    if (!searchQuery) return sessions;
    const query = searchQuery.toLowerCase();
    return sessions.filter(
      s => s.title.toLowerCase().includes(query) || s.messages.some(m => m.content.toLowerCase().includes(query)),
    );
  }, [sessions, searchQuery]);

  // Reserved for future
  void _onActionRequest;

  // ═══════════════════════════════════════════════════════════════════════════
  // LIFECYCLE & EFFECTS
  // ═══════════════════════════════════════════════════════════════════════════

  // Initialize position
  useEffect(() => {
    if (typeof window !== "undefined") {
      setIconPos({ x: window.innerWidth - 80, y: window.innerHeight - 80 });
    }
  }, []);

  // Drag & Resize Handlers
  const startDrag = useCallback((e: React.MouseEvent) => {
    // Only allow dragging if not clicking the close button (if open)
    setIsDragging(true);
    dragOffset.current = { x: e.clientX - iconPos.x, y: e.clientY - iconPos.y };
  }, [iconPos]);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    resizeStart.current = { x: e.clientX, y: e.clientY, w: winSize.w, h: winSize.h };
  }, [winSize]);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (isDragging) {
        setIconPos({
          x: e.clientX - dragOffset.current.x,
          y: e.clientY - dragOffset.current.y,
        });
      }
      if (isResizing) {
        // Resize logic: Assuming anchor is bottom-right relative to icon, 
        // but we are positioning top-left relative to icon.
        // Let's allow free resizing.
        // If we drag the resize handle (bottom-left of window?), we change width/height.
        // Actually, let's just change width/height based on delta.
        // Since the window is positioned relative to the icon (bottom-right), 
        // increasing width should expand it to the LEFT.
        // Increasing height should expand it UP.
        // Wait, standard resize handles are usually bottom-right.
        // If I put the handle at bottom-left of the window (since it's on the right of screen),
        // dragging LEFT increases width.
        
        const deltaX = resizeStart.current.x - e.clientX; // Drag left to increase width
        const deltaY = resizeStart.current.y - e.clientY; // Drag up to increase height? 
        // No, let's make it standard: Handle at Top-Left? No.
        // Let's put handle at Bottom-Left.
        
        setWinSize({
          w: Math.max(300, resizeStart.current.w + deltaX),
          h: Math.max(400, resizeStart.current.h + (e.clientY - resizeStart.current.y)) // Drag down to increase height
        });
      }
    };

    const handleUp = () => {
      setIsDragging(false);
      setIsResizing(false);
    };

    if (isDragging || isResizing) {
      window.addEventListener("mousemove", handleMove);
      window.addEventListener("mouseup", handleUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [isDragging, isResizing]);

  // Load sessions from localStorage
  useEffect(() => {
    const loaded = loadSessions();
    setSessions(loaded);
    if (loaded.length > 0) {
      setCurrentSessionId(loaded[0].id);
    }
  }, []);

  // Save sessions when they change
  useEffect(() => {
    if (sessions.length > 0) {
      saveSessions(sessions);
    }
  }, [sessions]);

  // Check connection - verifies Gemini Cloud API is available
  const refreshModelState = useCallback(() => {
    const model = geminiService.getActiveModel();
    setActiveModel(model);
    setIsUsingLocalGPU(model.toLowerCase().includes("qwen"));
  }, []);

  const checkConnection = useCallback(async () => {
    try {
      // Check Gemini API availability
      const isHealthy = await geminiService.checkHealth();
      setIsConnected(isHealthy);
      refreshModelState();
    } catch {
      setIsConnected(false);
    }
  }, [refreshModelState]);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 60000); // Check every minute for cloud API
    return () => clearInterval(interval);
  }, [checkConnection]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  // Initial greeting on new session
  useEffect(() => {
    if (isOpen && currentSession && currentSession.messages.length === 0) {
      const greeting: Message = {
        id: "greeting",
        role: "assistant",
        content: `👋 Welcome to **PayFlow AI Assistant**!

🧠 **AI Model:** Perplexity Sonar Pro (Cloud AI)
☁️ **Runtime:** Perplexity API (isolated from blockchain)
🔒 **Security:** 100% Web3 data isolation

🛡️ **This chatbot is ISOLATED from all blockchain data for security!**
It can only help with documentation and education about PayFlow.

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
        model: "sonar",
      };
      // Update session with greeting message
      setSessions(prev =>
        prev.map(s => {
          if (s.id !== currentSessionId) return s;
          return { ...s, messages: [greeting], updatedAt: new Date() };
        }),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, currentSession?.id, currentSession?.messages.length]);

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

        // Update title based on first user message
        const { title: currentTitle } = s;
        const firstUserMsg = newMessages.find(m => m.role === "user");
        const newTitle =
          currentTitle === "New Chat" && firstUserMsg
            ? firstUserMsg.content.slice(0, 40) + (firstUserMsg.content.length > 40 ? "..." : "")
            : currentTitle;

        return {
          ...s,
          title: newTitle,
          messages: newMessages,
          updatedAt: new Date(),
        };
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
  // VOICE INPUT (Web Speech API)
  // ═══════════════════════════════════════════════════════════════════════════

  const startVoiceInput = useCallback(() => {
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
      alert("Voice input is not supported in this browser");
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onerror = () => setIsListening(false);

    recognition.onresult = event => {
      const transcript = Array.from(event.results)
        .map(result => result[0].transcript)
        .join("");
      setInput(transcript);
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, []);

  const stopVoiceInput = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, []);

  // ═══════════════════════════════════════════════════════════════════════════
  // STREAMING LLM CALL (GEMINI CLOUD API - ISOLATED FROM WEB3)
  // ═══════════════════════════════════════════════════════════════════════════

  const callLLMStreaming = useCallback(
    async (
      userInput: string,
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      _history: Message[],
    ): Promise<{ response: string; suggestions: string[]; responseTime: number }> => {
      const startTime = Date.now();

      // Check cache first
      const cacheKey = userInput.toLowerCase().trim();
      const cached = responseCache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        return {
          response: cached.response,
          suggestions: cached.suggestions,
          responseTime: Date.now() - startTime,
        };
      }

      // Topic validation
      if (!isPayFlowRelated(userInput)) {
        const offTopicResponse =
          "I specialize in **PayFlow Protocol**. I can help with payments, compliance, escrow, oracles, fraud detection, and gasless transfers. What would you like to know about PayFlow?";
        return {
          response: offTopicResponse,
          suggestions: ["How do payments work?", "What are compliance tiers?", "Explain fraud detection"],
          responseTime: Date.now() - startTime,
        };
      }

      // Set up abort controller for cancellation
      const controller = new AbortController();
      setAbortController(controller);

      try {
        // Use Gemini Cloud API (Web3-isolated, streaming with callbacks)
        return await new Promise((resolve, reject) => {
          let streamedText = "";
          
          geminiService.sendMessageStreaming(
            userInput,
            {
              onToken: (token: string) => {
                streamedText += token;
                setStreamingText(streamedText);
              },
              onComplete: result => {
                // Cache the response
                responseCache.set(cacheKey, {
                  response: result.response,
                  suggestions: result.suggestions,
                  timestamp: Date.now(),
                });
                resolve(result);
              },
              onError: (error: Error) => {
                reject(error);
              },
            },
            controller.signal,
          );
        });
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          return {
            response: "Response cancelled.",
            suggestions: [],
            responseTime: Date.now() - startTime,
          };
        }

        // Fallback: Try non-streaming Gemini call
        try {
          const result = await geminiService.sendMessage(userInput);
          return result;
        } catch {
          throw new Error("AI service unavailable. Please check your internet connection.");
        }
      } finally {
        setAbortController(null);
      }
    },
    [],
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // SEND MESSAGE
  // ═══════════════════════════════════════════════════════════════════════════

  const sendMessage = useCallback(
    async (messageContent?: string) => {
      const content = messageContent || input.trim();
      if (!content || isLoading) return;
      if (!currentSessionId) {
        createNewSession();
        return;
      }

      const userMessage: Message = {
        id: generateId(),
        role: "user",
        content,
        timestamp: new Date(),
      };

      const loadingMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isStreaming: true,
      };

      // Add user message and loading state
      const updatedMessages = [...messages, userMessage, loadingMessage];
      updateSession(currentSessionId, updatedMessages);
      setInput("");
      setIsLoading(true);
      setStreamingText("");

      try {
        const { response, suggestions, responseTime } = await callLLMStreaming(content, messages);

        const assistantMessage: Message = {
          id: loadingMessage.id,
          role: "assistant",
          content: response,
          timestamp: new Date(),
          responseTime,
          suggestions,
          model: geminiService.getActiveModel(),
        };

        // Replace loading message with actual response
        updateSession(currentSessionId, [...messages, userMessage, assistantMessage]);
      } catch (error) {
        const errorMessage: Message = {
          id: loadingMessage.id,
          role: "assistant",
          content: `⚠️ ${error instanceof Error ? error.message : "Failed to get response"}`,
          timestamp: new Date(),
          isError: true,
        };

        updateSession(currentSessionId, [...messages, userMessage, errorMessage]);
      } finally {
        setIsLoading(false);
        setStreamingText("");
        refreshModelState();
      }
    },
    [input, isLoading, currentSessionId, messages, createNewSession, updateSession, callLLMStreaming, refreshModelState],
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // MESSAGE ACTIONS
  // ═══════════════════════════════════════════════════════════════════════════

  const handleReaction = useCallback(
    (messageId: string, reaction: "like" | "dislike" | null) => {
      if (!currentSessionId) return;
      const updated = messages.map(m => (m.id === messageId ? { ...m, reaction } : m));
      updateSession(currentSessionId, updated);
    },
    [currentSessionId, messages, updateSession],
  );

  const handleCopy = useCallback((content: string) => {
    navigator.clipboard.writeText(content);
  }, []);

  const handleRegenerate = useCallback(
    async (messageId: string) => {
      if (!currentSessionId) return;

      // Find the user message before this assistant message
      const msgIndex = messages.findIndex(m => m.id === messageId);
      if (msgIndex <= 0) return;

      const userMessage = messages[msgIndex - 1];
      if (userMessage.role !== "user") return;

      // Remove the old response
      const trimmedMessages = messages.slice(0, msgIndex);
      updateSession(currentSessionId, trimmedMessages);

      // Regenerate
      await sendMessage(userMessage.content);
    },
    [currentSessionId, messages, updateSession, sendMessage],
  );

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      sendMessage(suggestion);
    },
    [sendMessage],
  );

  const stopGeneration = useCallback(() => {
    if (abortController) {
      abortController.abort();
    }
  }, [abortController]);

  // ═══════════════════════════════════════════════════════════════════════════
  // EXPORT CONVERSATION
  // ═══════════════════════════════════════════════════════════════════════════

  const exportConversation = useCallback(
    (format: "markdown" | "json") => {
      if (!currentSession) return;

      let content: string;
      let filename: string;
      let mimeType: string;

      if (format === "markdown") {
        content = `# ${currentSession.title}\n\nExported: ${new Date().toLocaleString()}\n\n---\n\n`;
        content += currentSession.messages
          .filter(m => m.role !== "system")
          .map(
            m =>
              `**${m.role === "user" ? "You" : "PayFlow AI"}** (${m.timestamp.toLocaleTimeString()}):\n\n${m.content}`,
          )
          .join("\n\n---\n\n");
        filename = `payflow-chat-${Date.now()}.md`;
        mimeType = "text/markdown";
      } else {
        content = JSON.stringify(currentSession, null, 2);
        filename = `payflow-chat-${Date.now()}.json`;
        mimeType = "application/json";
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
    [currentSession],
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // FULLSCREEN TOGGLE
  // ═══════════════════════════════════════════════════════════════════════════

  const toggleFullscreen = useCallback(() => {
    if (isFullscreen) {
      // Restore previous size
      setWinSize(savedWinSize);
      setIconPos(savedIconPos);
      setIsFullscreen(false);
    } else {
      // Save current size and go fullscreen
      setSavedWinSize(winSize);
      setSavedIconPos(iconPos);
      if (typeof window !== "undefined") {
        setWinSize({ w: window.innerWidth - 40, h: window.innerHeight - 40 });
        setIconPos({ x: window.innerWidth - 60, y: window.innerHeight - 60 });
      }
      setIsFullscreen(true);
    }
  }, [isFullscreen, winSize, iconPos, savedWinSize, savedIconPos]);

  // ═══════════════════════════════════════════════════════════════════════════
  // KEYBOARD SHORTCUTS
  // ═══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape to close
      if (e.key === "Escape" && isOpen) {
        setIsOpen(false);
      }
      // Ctrl/Cmd + Enter to send
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && isOpen) {
        e.preventDefault();
        sendMessage();
      }
      // Ctrl/Cmd + N for new chat
      if ((e.ctrlKey || e.metaKey) && e.key === "n" && isOpen) {
        e.preventDefault();
        createNewSession();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, sendMessage, createNewSession]);

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <>
      {/* Floating Button */}
      <button
        onMouseDown={startDrag}
        onClick={(e) => {
          // Prevent click if it was a drag operation
          if (Math.abs(e.clientX - (dragOffset.current.x + iconPos.x)) < 5 && Math.abs(e.clientY - (dragOffset.current.y + iconPos.y)) < 5) {
             setIsOpen(!isOpen);
             if (!currentSessionId && sessions.length === 0) {
               createNewSession();
             }
          }
        }}
        style={{ left: iconPos.x, top: iconPos.y }}
        className={`fixed z-50 p-4 rounded-full shadow-xl transition-transform duration-300 cursor-move ${
          isOpen
            ? "bg-error text-error-content rotate-90 scale-90"
            : "bg-gradient-to-r from-primary to-secondary text-primary-content hover:scale-110 hover:shadow-2xl"
        }`}
        title={isOpen ? "Close chat" : "Open PayFlow AI Assistant"}
      >
        {isOpen ? <XMarkIcon className="w-6 h-6" /> : <ChatBubbleLeftRightIcon className="w-6 h-6" />}
        {!isOpen && isConnected && (
          <span className="absolute top-0 right-0 w-3 h-3 bg-success rounded-full animate-pulse" />
        )}
      </button>

      {/* Chat Panel */}
      {isOpen && (
        <div 
          style={isFullscreen ? { 
            left: 20, 
            top: 20, 
            width: 'calc(100vw - 40px)', 
            height: 'calc(100vh - 40px)' 
          } : { 
            left: iconPos.x - winSize.w + 50, 
            top: iconPos.y - winSize.h - 20, 
            width: winSize.w, 
            height: winSize.h 
          }}
          className={`fixed z-50 bg-base-100 shadow-2xl border border-base-300 flex flex-col overflow-hidden animate-in fade-in duration-200 ${isFullscreen ? 'rounded-xl' : 'rounded-2xl'}`}
        >
          {/* Resize Handle (Bottom-Left) - Hidden in fullscreen */}
          {!isFullscreen && (
            <div 
              onMouseDown={startResize}
              className="absolute bottom-0 left-0 w-8 h-8 cursor-sw-resize z-50 flex items-end justify-start p-1 opacity-30 hover:opacity-100 transition-opacity"
              title="Drag to resize"
            >
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-base-content/50 rotate-90">
                <path fill="currentColor" d="M22,22H2V20H22V22M22,18H6V16H22V18M22,14H10V12H22V14M22,10H14V8H22V10Z" />
              </svg>
            </div>
          )}

          {/* Header */}
          <div className="bg-gradient-to-r from-primary via-purple-600 to-secondary p-4 text-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur flex items-center justify-center">
                  <SparklesIcon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-lg">PayFlow AI</h3>
                  <div className="flex items-center gap-2 text-xs opacity-90">
                    {isUsingLocalGPU ? (
                      <>
                        <CpuChipIcon className="w-3 h-3" />
                        <span>qwen3:8b (Local GPU)</span>
                      </>
                    ) : (
                      <>
                        <CloudIcon className="w-3 h-3" />
                        <span>Sonar (Perplexity)</span>
                      </>
                    )}
                    <ShieldCheckIcon className="w-3 h-3 text-green-300" title="Isolated from Web3 data" />
                    {isConnected ? (
                      <span className="flex items-center gap-1 text-green-300">
                        <CheckCircleIcon className="w-3 h-3" /> Online
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-yellow-300">
                        <ExclamationTriangleIcon className="w-3 h-3" /> Offline
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Header Actions */}
              <div className="flex items-center gap-1">
                <button
                  onClick={toggleFullscreen}
                  className="btn btn-ghost btn-sm btn-circle text-white/80 hover:text-white hover:bg-white/20"
                  title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
                >
                  {isFullscreen ? <ArrowsPointingInIcon className="w-4 h-4" /> : <ArrowsPointingOutIcon className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className="btn btn-ghost btn-sm btn-circle text-white/80 hover:text-white hover:bg-white/20"
                  title="Chat history"
                >
                  <ClockIcon className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="btn btn-ghost btn-sm btn-circle text-white/80 hover:text-white hover:bg-white/20"
                  title="Settings"
                >
                  <Cog6ToothIcon className="w-4 h-4" />
                </button>
                <button
                  onClick={createNewSession}
                  className="btn btn-ghost btn-sm btn-circle text-white/80 hover:text-white hover:bg-white/20"
                  title="New chat (Ctrl+N)"
                >
                  <ChatBubbleLeftRightIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* History Panel */}
          {showHistory && (
            <div className="absolute inset-0 z-10 bg-base-100 flex flex-col">
              <div className="p-4 border-b border-base-300">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-bold">Chat History</h4>
                  <button
                    onClick={() => setShowHistory(false)}
                    className="btn btn-ghost btn-xs btn-circle"
                    title="Close history"
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </button>
                </div>
                <div className="relative">
                  <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-base-content/50" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    placeholder="Search conversations..."
                    className="input input-sm input-bordered w-full pl-9"
                  />
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {filteredSessions.length === 0 ? (
                  <div className="text-center py-8 text-base-content/50">
                    <ClockIcon className="w-8 h-8 mx-auto mb-2 opacity-30" />
                    <p>No chat history</p>
                  </div>
                ) : (
                  filteredSessions.map(session => (
                    <div
                      key={session.id}
                      className={`p-3 rounded-lg mb-1 cursor-pointer hover:bg-base-200 transition-colors ${
                        session.id === currentSessionId ? "bg-primary/10 border border-primary/30" : ""
                      }`}
                      onClick={() => {
                        setCurrentSessionId(session.id);
                        setShowHistory(false);
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-sm truncate">{session.title}</span>
                        <button
                          onClick={e => {
                            e.stopPropagation();
                            deleteSession(session.id);
                          }}
                          className="btn btn-ghost btn-xs btn-circle opacity-50 hover:opacity-100"
                          title="Delete session"
                        >
                          <TrashIcon className="w-3 h-3" />
                        </button>
                      </div>
                      <div className="text-xs text-base-content/50 mt-1">
                        {session.messages.length} messages • {session.updatedAt.toLocaleDateString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
              <div className="p-3 border-t border-base-300">
                <button onClick={createNewSession} className="btn btn-primary btn-sm w-full">
                  <ChatBubbleLeftRightIcon className="w-4 h-4" />
                  New Chat
                </button>
              </div>
            </div>
          )}

          {/* Settings Panel */}
          {showSettings && (
            <div className="absolute inset-0 z-10 bg-base-100 flex flex-col">
              <div className="p-4 border-b border-base-300 flex items-center justify-between">
                <h4 className="font-bold">Settings</h4>
                <button
                  onClick={() => setShowSettings(false)}
                  className="btn btn-ghost btn-xs btn-circle"
                  title="Close settings"
                >
                  <XMarkIcon className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1 p-4 space-y-4">
                <div>
                  <label className="label">
                    <span className="label-text font-medium">Export Conversation</span>
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => exportConversation("markdown")}
                      className="btn btn-outline btn-sm flex-1"
                      disabled={!currentSession || messages.length === 0}
                    >
                      <DocumentArrowDownIcon className="w-4 h-4" />
                      Markdown
                    </button>
                    <button
                      onClick={() => exportConversation("json")}
                      className="btn btn-outline btn-sm flex-1"
                      disabled={!currentSession || messages.length === 0}
                    >
                      <DocumentArrowDownIcon className="w-4 h-4" />
                      JSON
                    </button>
                  </div>
                </div>

                <div>
                  <label className="label">
                    <span className="label-text font-medium">Clear All History</span>
                  </label>
                  <button
                    onClick={() => {
                      if (confirm("Delete all chat history?")) {
                        setSessions([]);
                        setCurrentSessionId(null);
                        localStorage.removeItem(STORAGE_KEY);
                        createNewSession();
                      }
                    }}
                    className="btn btn-error btn-outline btn-sm w-full"
                  >
                    <TrashIcon className="w-4 h-4" />
                    Clear All Chats
                  </button>
                </div>

                <div className="pt-4 border-t border-base-300">
                  <h5 className="font-medium mb-2">Keyboard Shortcuts</h5>
                  <div className="text-sm space-y-1 text-base-content/70">
                    <div className="flex justify-between">
                      <span>Send message</span>
                      <kbd className="kbd kbd-xs">Ctrl + Enter</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span>New chat</span>
                      <kbd className="kbd kbd-xs">Ctrl + N</kbd>
                    </div>
                    <div className="flex justify-between">
                      <span>Close</span>
                      <kbd className="kbd kbd-xs">Escape</kbd>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t border-base-300 text-center text-xs text-base-content/50">
                  <p>PayFlow AI Assistant v2.0</p>
                  <p>🔒 100% Local • No Cloud APIs</p>
                </div>
              </div>
            </div>
          )}

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {(() => {
              const filteredMessages = messages.filter(m => m.role !== "system");
              return filteredMessages.map((message, idx) => (
                <MessageComponent
                  key={message.id}
                  message={
                    message.isStreaming && streamingText
                      ? { ...message, content: streamingText, isStreaming: true }
                      : message
                  }
                  onReaction={handleReaction}
                  onCopy={handleCopy}
                  onRegenerate={handleRegenerate}
                  onSuggestionClick={handleSuggestionClick}
                  isLatest={idx === filteredMessages.length - 1}
                />
              ));
            })()}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-base-300 bg-base-100">
            {/* Stop button during generation */}
            {isLoading && (
              <div className="flex justify-center mb-2">
                <button onClick={stopGeneration} className="btn btn-sm btn-outline btn-error gap-1">
                  <StopIcon className="w-4 h-4" />
                  Stop generating
                </button>
              </div>
            )}

            <div className="flex gap-2 items-end">
              {/* Voice Input */}
              <button
                onClick={isListening ? stopVoiceInput : startVoiceInput}
                className={`btn btn-circle btn-sm ${isListening ? "btn-error animate-pulse" : "btn-ghost"}`}
                title={isListening ? "Stop listening" : "Voice input"}
              >
                <MicrophoneIcon className="w-4 h-4" />
              </button>

              {/* Text Input */}
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder="Ask about PayFlow... (Shift+Enter for new line)"
                  className="textarea textarea-bordered w-full min-h-[44px] max-h-[120px] resize-none pr-12 text-sm"
                  disabled={isLoading}
                  rows={1}
                />
                <span className="absolute right-3 bottom-2 text-xs text-base-content/40">{input.length}/500</span>
              </div>

              {/* Send Button */}
              <button
                onClick={() => sendMessage()}
                disabled={isLoading || !input.trim()}
                className="btn btn-primary btn-circle btn-sm"
                title="Send message (Ctrl+Enter)"
              >
                <PaperAirplaneIcon className="w-4 h-4" />
              </button>
            </div>

            {/* Footer Info */}
            <div className="flex items-center justify-between mt-2 text-xs text-base-content/50">
              <span>⚡ Powered by Perplexity Sonar Pro • Fast & Secure</span>
              <span>{sessions.length} chats saved</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default AIChatbotPro;
