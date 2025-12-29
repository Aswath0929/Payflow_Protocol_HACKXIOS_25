"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  PAYFLOW AI CHATBOT - AUTONOMOUS LOCAL AI AGENT                          ║
 * ║  Context-Aware Protocol Support Interface                                ║
 * ╠══════════════════════════════════════════════════════════════════════════╣
 * ║  Built for Hackxios 2K25 - PayPal & Visa Track                           ║
 * ║                                                                           ║
 * ║  Features:                                                                ║
 * ║  • 100% Local LLM (Qwen3:8B via Ollama) - No cloud APIs                  ║
 * ║  • Protocol-aware responses (knows PayFlow contracts)                    ║
 * ║  • Real-time transaction analysis suggestions                            ║
 * ║  • Compliance guidance (KYC tiers, Travel Rule)                          ║
 * ║  • Fraud detection explanations                                          ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowPathIcon,
  ChatBubbleLeftRightIcon,
  CheckCircleIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  PaperAirplaneIcon,
  SparklesIcon,
  UserCircleIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  isLoading?: boolean;
  metadata?: {
    riskScore?: number;
    typologies?: string[];
    recommendation?: string;
  };
}

interface ChatbotProps {
  /** Current wallet address for context */
  walletAddress?: string;
  /** Current transaction context */
  transactionContext?: {
    sender?: string;
    recipient?: string;
    amount?: number;
    token?: string;
  };
  /** Callback when user requests action */
  onActionRequest?: (action: string, params: Record<string, unknown>) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// PROTOCOL KNOWLEDGE BASE
// ═══════════════════════════════════════════════════════════════════════════

const PROTOCOL_CONTEXT = `You are PayFlow AI Assistant, a SPECIALIZED expert ONLY on the PayFlow Protocol.

⚠️ CRITICAL RULES - YOU MUST FOLLOW THESE:
1. You ONLY answer questions about PayFlow Protocol and its features
2. If a question is NOT related to PayFlow (e.g., general coding, weather, news, other topics), respond with:
   "I'm PayFlow AI Assistant, specialized in helping you understand and use PayFlow Protocol. I can help you with payments, compliance, escrow, oracles, and fraud detection. What would you like to know about PayFlow?"
3. Do NOT answer general knowledge questions, coding help unrelated to PayFlow, or any off-topic queries
4. Always guide users back to PayFlow features

PAYFLOW PROTOCOL KNOWLEDGE BASE:

1. **Core Smart Contracts**:
   - PayFlowCore.sol: createPayment(), executePayment(), settleWithFX(), cancelPayment()
   - ComplianceEngine.sol: 5-tier KYC system with setUserTier(), checkCompliance()
   - SmartEscrow.sol: 4 release types (TIME_BASED, APPROVAL, ORACLE, MULTI_SIG)
   - OracleAggregator.sol: Chainlink (60%) + Pyth (40%) weighted price consensus
   - AuditRegistry.sol: Immutable compliance logging for regulators
   - FraudOracle.sol: AI-powered risk scoring with on-chain verification
   - PayFlowPaymaster.sol: ERC-4337 gasless transactions

2. **Compliance Tiers & Limits**:
   - NONE (Tier 0): $1K/day, $5K/month - No verification needed
   - BASIC (Tier 1): $10K/day, $50K/month - Email + phone verification
   - STANDARD (Tier 2): $100K/day, $500K/month - Government ID required
   - ENHANCED (Tier 3): $1M/day, $5M/month - Full KYC + AML screening
   - INSTITUTIONAL (Tier 4): Unlimited - Corporate KYC + UBO verification

3. **AI Fraud Detection System**:
   - 4-Model Ensemble: Neural Network (25%) + Typology Detector (25%) + Qwen3 LLM (30%) + Compliance Engine (20%)
   - 15 Fraud Typologies: Mixing, Layering, Wash Trading, Structuring, Flash Loans, Sybil Attacks, etc.
   - Risk Levels: SAFE (0-20), LOW (21-40), MEDIUM (41-60), HIGH (61-80), CRITICAL (81-100)
   - All verdicts are ECDSA-signed for on-chain verification

4. **Gasless Transactions (ERC-4337)**:
   - Users send stablecoins without holding ETH
   - 0.1% fee deduction from transfer amount OR sponsor-subsidized
   - Rate limit: 100 transactions per hour per user
   - Corporate sponsors can whitelist employees for free transactions

5. **Supported Stablecoins**:
   - PYUSD (PayPal USD) - Primary integration
   - USDC (Circle)
   - DAI (MakerDAO)
   - USDT (Tether)

6. **Smart Escrow Release Types**:
   - TIME_BASED: Automatic release after deadline
   - APPROVAL: Manual approval by designated party
   - ORACLE: Release triggered by price oracle condition
   - MULTI_SIG: Requires multiple signatures to release

Remember: ONLY answer PayFlow-related questions. Politely redirect off-topic queries.`;

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export const AIChatbot: React.FC<ChatbotProps> = ({
  walletAddress,
  transactionContext,
  onActionRequest: _onActionRequest,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reserved for future action handling
  void _onActionRequest;

  // ═══════════════════════════════════════════════════════════════════════
  // CONNECTION CHECK
  // ═══════════════════════════════════════════════════════════════════════

  const checkConnection = useCallback(async () => {
    try {
      const response = await fetch("http://localhost:8000/health", {
        method: "GET",
        signal: AbortSignal.timeout(3000),
      });

      if (response.ok) {
        setIsConnected(true);
        setConnectionError(null);
      } else {
        throw new Error("API not healthy");
      }
    } catch {
      setIsConnected(false);
      setConnectionError("AI Oracle offline. Start with: python -m uvicorn expertAPI:app --port 8000");
    }
  }, []);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, [checkConnection]);

  // ═══════════════════════════════════════════════════════════════════════
  // SCROLL TO BOTTOM
  // ═══════════════════════════════════════════════════════════════════════

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ═══════════════════════════════════════════════════════════════════════
  // INITIAL GREETING
  // ═══════════════════════════════════════════════════════════════════════

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const greeting: Message = {
        id: "greeting",
        role: "assistant",
        content: `👋 Hello! I'm PayFlow AI Assistant, powered by **Qwen3:8B** running 100% locally.

I can help you with:
• 💸 **Payment flows** - Create, approve, execute transfers
• 🛡️ **Compliance** - KYC tiers, Travel Rule requirements
• 🔮 **Oracle prices** - Real-time FX rates from Chainlink + Pyth
• 🔒 **Smart Escrow** - Conditional payment setups
• 🧠 **Fraud detection** - Transaction risk analysis
• ⛽ **Gasless transfers** - Visa-style sponsored transactions

What would you like to know?`,
        timestamp: new Date(),
      };
      setMessages([greeting]);
    }
  }, [isOpen, messages.length]);

  // ═══════════════════════════════════════════════════════════════════════
  // SEND MESSAGE
  // ═══════════════════════════════════════════════════════════════════════

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Add loading message
    const loadingId = `loading-${Date.now()}`;
    setMessages(prev => [
      ...prev,
      {
        id: loadingId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isLoading: true,
      },
    ]);

    try {
      const response = await callLocalLLM(userMessage.content);

      // Replace loading message with response
      setMessages(prev =>
        prev.map(msg => (msg.id === loadingId ? { ...msg, content: response, isLoading: false } : msg)),
      );
    } catch (error) {
      // Replace loading with error
      setMessages(prev =>
        prev.map(msg =>
          msg.id === loadingId
            ? {
                ...msg,
                content: `⚠️ Error: ${error instanceof Error ? error.message : "Failed to get response"}. Please ensure the AI Oracle is running.`,
                isLoading: false,
              }
            : msg,
        ),
      );
    } finally {
      setIsLoading(false);
    }
  };

  // ═══════════════════════════════════════════════════════════════════════
  // TOPIC VALIDATION - Ensures queries are PayFlow-related
  // ═══════════════════════════════════════════════════════════════════════

  const PAYFLOW_KEYWORDS = [
    // Core concepts
    "payflow",
    "payment",
    "transfer",
    "send",
    "receive",
    "transaction",
    "tx",
    // Contracts
    "contract",
    "escrow",
    "oracle",
    "compliance",
    "kyc",
    "audit",
    "paymaster",
    // Tokens
    "pyusd",
    "usdc",
    "dai",
    "usdt",
    "stablecoin",
    "token",
    "erc20",
    // Features
    "gasless",
    "gas",
    "fee",
    "sponsor",
    "whitelist",
    "tier",
    "limit",
    // Fraud/Risk
    "fraud",
    "risk",
    "score",
    "block",
    "approve",
    "review",
    "typology",
    // Technical
    "wallet",
    "address",
    "deploy",
    "sepolia",
    "mainnet",
    "chain",
    // Actions
    "create",
    "execute",
    "cancel",
    "settle",
    "release",
    "lock",
    // Questions about platform
    "how",
    "what",
    "why",
    "when",
    "where",
    "can i",
    "help",
    "explain",
    "tell me",
  ];

  const isPayFlowRelated = (query: string): boolean => {
    const lowerQuery = query.toLowerCase();
    // Check if any PayFlow keyword is present
    return (
      PAYFLOW_KEYWORDS.some(keyword => lowerQuery.includes(keyword)) ||
      // Also allow greetings and basic questions
      lowerQuery.length < 20 ||
      /^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure)/.test(lowerQuery)
    );
  };

  const OFF_TOPIC_RESPONSE = `I'm **PayFlow AI Assistant**, specialized exclusively in helping you understand and use the PayFlow Protocol.

I can help you with:
• 💸 **Payments** - Creating, executing, and tracking transfers
• 🛡️ **Compliance** - KYC tiers, limits, and verification
• 🔒 **Smart Escrow** - Conditional payment setups
• 🔮 **Oracles** - Price feeds and FX rates
• 🧠 **Fraud Detection** - Risk scores and typologies
• ⛽ **Gasless Transfers** - Send without ETH

What would you like to know about PayFlow?`;

  // ═══════════════════════════════════════════════════════════════════════
  // LOCAL LLM CALL
  // ═══════════════════════════════════════════════════════════════════════

  const callLocalLLM = async (userInput: string): Promise<string> => {
    // Client-side topic filter - catch obvious off-topic queries
    if (!isPayFlowRelated(userInput)) {
      return OFF_TOPIC_RESPONSE;
    }

    // Build context with current state
    const contextInfo = [];
    if (walletAddress) {
      contextInfo.push(`User wallet: ${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`);
    }
    if (transactionContext) {
      contextInfo.push(`Current transaction: ${JSON.stringify(transactionContext)}`);
    }

    const fullPrompt = `${PROTOCOL_CONTEXT}

${contextInfo.length > 0 ? `CURRENT USER CONTEXT:\n${contextInfo.join("\n")}\n\n` : ""}USER QUESTION: ${userInput}

RESPONSE GUIDELINES:
- If this question is about PayFlow Protocol, answer helpfully with specific details
- If this question is NOT about PayFlow (e.g., general knowledge, other topics), politely redirect to PayFlow features
- Use markdown formatting for clarity
- Reference specific contract functions when applicable
- Keep responses concise but informative`;

    // Try Expert AI Oracle first
    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userInput,
          context: fullPrompt,
          use_thinking: false,
        }),
        signal: AbortSignal.timeout(30000),
      });

      if (response.ok) {
        const data = await response.json();
        return data.response || data.message || "I understood your question but couldn't generate a response.";
      }
    } catch {
      // Fall through to direct Ollama
    }

    // Fallback: Direct Ollama call
    try {
      const ollamaResponse = await fetch("http://localhost:11434/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "qwen3:8b",
          prompt: fullPrompt,
          stream: false,
          options: {
            temperature: 0.7,
            num_predict: 500,
          },
        }),
        signal: AbortSignal.timeout(60000),
      });

      if (ollamaResponse.ok) {
        const data = await ollamaResponse.json();
        // Remove thinking tags if present
        let response = data.response || "";
        response = response.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
        return response || "I understood your question but couldn't generate a response.";
      }
    } catch {
      throw new Error("Both AI Oracle and Ollama are unavailable");
    }

    throw new Error("Failed to get response from AI");
  };

  // ═══════════════════════════════════════════════════════════════════════
  // QUICK ACTIONS
  // ═══════════════════════════════════════════════════════════════════════

  const quickActions = [
    { label: "Check my KYC tier", query: "What KYC tier am I and what are the transaction limits?" },
    { label: "Analyze transaction", query: "Analyze the current transaction for fraud risk" },
    { label: "Oracle prices", query: "What are the current ETH/USD and BTC/USD prices from oracles?" },
    { label: "How escrow works", query: "Explain how SmartEscrow works with the 4 release types" },
    { label: "Gasless transfers", query: "How do gasless PYUSD transfers work with the Paymaster?" },
  ];

  // ═══════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed bottom-6 right-6 z-50 p-4 rounded-full shadow-lg transition-all duration-300 ${
          isOpen ? "bg-error text-error-content rotate-90" : "bg-primary text-primary-content hover:scale-110"
        }`}
      >
        {isOpen ? <XMarkIcon className="w-6 h-6" /> : <ChatBubbleLeftRightIcon className="w-6 h-6" />}
      </button>

      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 w-96 h-[600px] bg-base-100 rounded-2xl shadow-2xl border border-base-300 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-primary to-secondary p-4 text-primary-content">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <SparklesIcon className="w-6 h-6" />
                <div>
                  <h3 className="font-bold">PayFlow AI Assistant</h3>
                  <p className="text-xs opacity-80">Powered by Qwen3:8B (Local)</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {isConnected ? (
                  <div className="flex items-center gap-1 text-xs">
                    <CheckCircleIcon className="w-4 h-4 text-success" />
                    <span>Online</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1 text-xs text-warning">
                    <ExclamationTriangleIcon className="w-4 h-4" />
                    <span>Offline</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Connection Error Banner */}
          {connectionError && (
            <div className="bg-warning/10 border-b border-warning/30 px-4 py-2 text-xs text-warning flex items-center gap-2">
              <ExclamationTriangleIcon className="w-4 h-4 flex-shrink-0" />
              <span className="truncate">{connectionError}</span>
              <button onClick={checkConnection} className="btn btn-xs btn-ghost" title="Retry connection">
                <ArrowPathIcon className="w-3 h-3" />
              </button>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map(message => (
              <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                {/* Avatar */}
                <div
                  className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.role === "user" ? "bg-primary" : "bg-secondary"
                  }`}
                >
                  {message.role === "user" ? (
                    <UserCircleIcon className="w-5 h-5 text-primary-content" />
                  ) : (
                    <CpuChipIcon className="w-5 h-5 text-secondary-content" />
                  )}
                </div>

                {/* Message Bubble */}
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                    message.role === "user"
                      ? "bg-primary text-primary-content rounded-br-none"
                      : "bg-base-200 text-base-content rounded-bl-none"
                  }`}
                >
                  {message.isLoading ? (
                    <div className="flex items-center gap-2">
                      <span className="loading loading-dots loading-sm"></span>
                      <span className="text-sm opacity-70">Thinking...</span>
                    </div>
                  ) : (
                    <div
                      className="prose prose-sm max-w-none"
                      dangerouslySetInnerHTML={{
                        __html: formatMarkdown(message.content),
                      }}
                    />
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Actions */}
          {messages.length <= 1 && (
            <div className="px-4 pb-2">
              <p className="text-xs text-base-content/60 mb-2">Quick questions:</p>
              <div className="flex flex-wrap gap-1">
                {quickActions.map((action, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInput(action.query);
                      setTimeout(() => sendMessage(), 100);
                    }}
                    className="btn btn-xs btn-outline"
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="p-4 border-t border-base-300">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendMessage()}
                placeholder="Ask about PayFlow..."
                className="input input-bordered flex-1 input-sm"
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                className="btn btn-primary btn-sm"
                title="Send message"
              >
                <PaperAirplaneIcon className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-base-content/50 mt-2 text-center">
              🔒 100% local AI • Your data never leaves your machine
            </p>
          </div>
        </div>
      )}
    </>
  );
};

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: Simple Markdown Formatter
// ═══════════════════════════════════════════════════════════════════════════

function formatMarkdown(text: string): string {
  return (
    text
      // Bold
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      // Italic
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      // Code
      .replace(/`(.*?)`/g, "<code class='bg-base-300 px-1 rounded'>$1</code>")
      // Lists
      .replace(/^• /gm, "• ")
      // Line breaks
      .replace(/\n/g, "<br />")
  );
}

export default AIChatbot;
