/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║  PAYFLOW CLOUD AI PROVIDER - ISOLATED WEB2 CHATBOT                                   ║
 * ╠══════════════════════════════════════════════════════════════════════════════════════╣
 * ║                                                                                       ║
 * ║  🌐 WEB2 LAYER: Cloud AI for Chatbot (Internet-facing, isolated)                     ║
 * ║  🔒 WEB3 LAYER: Local Qwen3:8B for Fraud Detection (Air-gapped, secure)              ║
 * ║                                                                                       ║
 * ║  SECURITY ARCHITECTURE:                                                               ║
 * ║  ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
 * ║  │                                                                                 │ ║
 * ║  │   USER INTERFACE                                                                │ ║
 * ║  │         │                                                                       │ ║
 * ║  │         ├────────────────────────┬──────────────────────────────────────┐       │ ║
 * ║  │         │                        │                                      │       │ ║
 * ║  │         ▼                        ▼                                      │       │ ║
 * ║  │   ┌─────────────┐          ┌─────────────┐                              │       │ ║
 * ║  │   │  CHATBOT    │          │   FRAUD     │                              │       │ ║
 * ║  │   │  (Web2)     │          │ DETECTION   │                              │       │ ║
 * ║  │   │             │          │  (Web3)     │                              │       │ ║
 * ║  │   └──────┬──────┘          └──────┬──────┘                              │       │ ║
 * ║  │          │                        │                                      │       │ ║
 * ║  │          ▼                        ▼                                      │       │ ║
 * ║  │   ┌─────────────┐          ┌─────────────┐                              │       │ ║
 * ║  │   │ Cloud API   │          │ LOCAL GPU   │                              │       │ ║
 * ║  │   │ (OpenAI/    │          │ Qwen3:8B    │                              │       │ ║
 * ║  │   │  Gemini/    │          │ (8GB VRAM)  │                              │       │ ║
 * ║  │   │  Claude)    │          │             │                              │       │ ║
 * ║  │   └──────┬──────┘          └─────────────┘                              │       │ ║
 * ║  │          │                        │                                      │       │ ║
 * ║  │          ▼                        ▼                                      │       │ ║
 * ║  │   ┌─────────────┐          ┌─────────────┐                              │       │ ║
 * ║  │   │  INTERNET   │          │  AIR-GAPPED │                              │       │ ║
 * ║  │   │  (Public)   │          │  (Secure)   │                              │       │ ║
 * ║  │   └─────────────┘          └─────────────┘                              │       │ ║
 * ║  │                                                                         │       │ ║
 * ║  │   ✅ No blockchain data sent to cloud                                   │       │ ║
 * ║  │   ✅ Transaction analysis stays local                                   │       │ ║
 * ║  │   ✅ Full GPU for fraud detection                                       │       │ ║
 * ║  │                                                                         │       │ ║
 * ║  └─────────────────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                                       ║
 * ║  Hackxios 2K25 - PayFlow Protocol                                                    ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 */

// ═══════════════════════════════════════════════════════════════════════════════
//                              TYPES & INTERFACES
// ═══════════════════════════════════════════════════════════════════════════════

export type AIProvider = "openai" | "google" | "anthropic" | "groq" | "openrouter";

export interface CloudAIConfig {
  provider: AIProvider;
  apiKey: string;
  model: string;
  baseUrl?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface StreamCallback {
  onToken: (token: string) => void;
  onComplete: (fullResponse: string) => void;
  onError: (error: Error) => void;
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              PROVIDER CONFIGURATIONS
// ═══════════════════════════════════════════════════════════════════════════════

export const PROVIDER_CONFIGS: Record<AIProvider, {
  name: string;
  baseUrl: string;
  defaultModel: string;
  models: string[];
  streamSupport: boolean;
}> = {
  openai: {
    name: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-4o-mini",
    models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    streamSupport: true,
  },
  google: {
    name: "Google AI (Gemini)",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    defaultModel: "gemini-1.5-flash",
    models: ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
    streamSupport: true,
  },
  anthropic: {
    name: "Anthropic (Claude)",
    baseUrl: "https://api.anthropic.com/v1",
    defaultModel: "claude-3-haiku-20240307",
    models: ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
    streamSupport: true,
  },
  groq: {
    name: "Groq (Ultra Fast)",
    baseUrl: "https://api.groq.com/openai/v1",
    defaultModel: "llama-3.3-70b-versatile",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    streamSupport: true,
  },
  openrouter: {
    name: "OpenRouter (Multi-Model)",
    baseUrl: "https://openrouter.ai/api/v1",
    defaultModel: "google/gemini-2.0-flash-exp:free",
    models: [
      "google/gemini-2.0-flash-exp:free",
      "meta-llama/llama-3.2-3b-instruct:free",
      "qwen/qwen-2.5-72b-instruct:free",
    ],
    streamSupport: true,
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//                              PAYFLOW SYSTEM PROMPT
// ═══════════════════════════════════════════════════════════════════════════════

export const PAYFLOW_SYSTEM_PROMPT = `You are PayFlow AI Assistant, a SPECIALIZED expert ONLY on the PayFlow Protocol.

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

**Compliance Tiers & Limits:**
- NONE: $1K/day, $5K/month | BASIC: $10K/day, $50K/month
- STANDARD: $100K/day, $500K/month | ENHANCED: $1M/day, $5M/month
- INSTITUTIONAL: Unlimited

**AI Fraud Detection System:**
- Expert AI Oracle v3.0 with 5-Model Neural Ensemble
- Models: Deep MLP, Gradient Boost, Graph Attention, Temporal LSTM, Isolation Forest
- Risk Levels: SAFE (0-20), LOW (21-40), MEDIUM (41-60), HIGH (61-80), CRITICAL (81-100)
- Local GPU inference for security (no data leaves device)

**Supported Tokens:** PYUSD, USDC, DAI, USDT

**Unique Features:**
- Gasless transactions via ERC-4337
- Multi-oracle price feeds
- AI-powered real-time fraud detection
- Cryptographically signed ML verdicts`;

// ═══════════════════════════════════════════════════════════════════════════════
//                              CLOUD AI CLIENT
// ═══════════════════════════════════════════════════════════════════════════════

export class CloudAIClient {
  private config: CloudAIConfig;
  private providerConfig: typeof PROVIDER_CONFIGS[AIProvider];

  constructor(config: CloudAIConfig) {
    this.config = {
      ...config,
      maxTokens: config.maxTokens || 1024,
      temperature: config.temperature || 0.7,
    };
    this.providerConfig = PROVIDER_CONFIGS[config.provider];
  }

  /**
   * Generate a response (non-streaming)
   */
  async generate(messages: ChatMessage[]): Promise<string> {
    switch (this.config.provider) {
      case "openai":
      case "groq":
      case "openrouter":
        return this.generateOpenAICompatible(messages);
      case "google":
        return this.generateGemini(messages);
      case "anthropic":
        return this.generateAnthropic(messages);
      default:
        throw new Error(`Unsupported provider: ${this.config.provider}`);
    }
  }

  /**
   * Generate with streaming (for real-time token display)
   */
  async generateStream(messages: ChatMessage[], callbacks: StreamCallback): Promise<void> {
    switch (this.config.provider) {
      case "openai":
      case "groq":
      case "openrouter":
        return this.streamOpenAICompatible(messages, callbacks);
      case "google":
        return this.streamGemini(messages, callbacks);
      case "anthropic":
        return this.streamAnthropic(messages, callbacks);
      default:
        throw new Error(`Unsupported provider: ${this.config.provider}`);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //                         OPENAI-COMPATIBLE API
  // ═══════════════════════════════════════════════════════════════════════════

  private async generateOpenAICompatible(messages: ChatMessage[]): Promise<string> {
    const baseUrl = this.config.baseUrl || this.providerConfig.baseUrl;
    
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.config.apiKey}`,
        ...(this.config.provider === "openrouter" && {
          "HTTP-Referer": "https://payflow.protocol",
          "X-Title": "PayFlow Protocol",
        }),
      },
      body: JSON.stringify({
        model: this.config.model || this.providerConfig.defaultModel,
        messages: [
          { role: "system", content: PAYFLOW_SYSTEM_PROMPT },
          ...messages,
        ],
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
        stream: false,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || "";
  }

  private async streamOpenAICompatible(messages: ChatMessage[], callbacks: StreamCallback): Promise<void> {
    const baseUrl = this.config.baseUrl || this.providerConfig.baseUrl;
    
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.config.apiKey}`,
        ...(this.config.provider === "openrouter" && {
          "HTTP-Referer": "https://payflow.protocol",
          "X-Title": "PayFlow Protocol",
        }),
      },
      body: JSON.stringify({
        model: this.config.model || this.providerConfig.defaultModel,
        messages: [
          { role: "system", content: PAYFLOW_SYSTEM_PROMPT },
          ...messages,
        ],
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
        stream: true,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      callbacks.onError(new Error(`API Error: ${response.status} - ${error}`));
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError(new Error("No response body"));
      return;
    }

    const decoder = new TextDecoder();
    let fullResponse = "";

    try {
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
              callbacks.onToken(token);
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }

      callbacks.onComplete(fullResponse);
    } catch (error) {
      callbacks.onError(error as Error);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //                              GOOGLE GEMINI API
  // ═══════════════════════════════════════════════════════════════════════════

  private async generateGemini(messages: ChatMessage[]): Promise<string> {
    const model = this.config.model || this.providerConfig.defaultModel;
    const url = `${this.providerConfig.baseUrl}/models/${model}:generateContent?key=${this.config.apiKey}`;

    const contents = messages.map(m => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.role === "system" ? PAYFLOW_SYSTEM_PROMPT + "\n\n" + m.content : m.content }],
    }));

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents,
        generationConfig: {
          maxOutputTokens: this.config.maxTokens,
          temperature: this.config.temperature,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gemini API Error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  }

  private async streamGemini(messages: ChatMessage[], callbacks: StreamCallback): Promise<void> {
    const model = this.config.model || this.providerConfig.defaultModel;
    const url = `${this.providerConfig.baseUrl}/models/${model}:streamGenerateContent?alt=sse&key=${this.config.apiKey}`;

    const contents = [
      { role: "user", parts: [{ text: PAYFLOW_SYSTEM_PROMPT }] },
      { role: "model", parts: [{ text: "I understand. I am PayFlow AI Assistant." }] },
      ...messages.map(m => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      })),
    ];

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents,
        generationConfig: {
          maxOutputTokens: this.config.maxTokens,
          temperature: this.config.temperature,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      callbacks.onError(new Error(`Gemini API Error: ${response.status} - ${error}`));
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError(new Error("No response body"));
      return;
    }

    const decoder = new TextDecoder();
    let fullResponse = "";

    try {
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
              callbacks.onToken(token);
            }
          } catch {
            // Skip malformed lines
          }
        }
      }

      callbacks.onComplete(fullResponse);
    } catch (error) {
      callbacks.onError(error as Error);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //                              ANTHROPIC CLAUDE API
  // ═══════════════════════════════════════════════════════════════════════════

  private async generateAnthropic(messages: ChatMessage[]): Promise<string> {
    const response = await fetch(`${this.providerConfig.baseUrl}/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.config.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: this.config.model || this.providerConfig.defaultModel,
        system: PAYFLOW_SYSTEM_PROMPT,
        messages: messages.filter(m => m.role !== "system"),
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Claude API Error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.content?.[0]?.text || "";
  }

  private async streamAnthropic(messages: ChatMessage[], callbacks: StreamCallback): Promise<void> {
    const response = await fetch(`${this.providerConfig.baseUrl}/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.config.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: this.config.model || this.providerConfig.defaultModel,
        system: PAYFLOW_SYSTEM_PROMPT,
        messages: messages.filter(m => m.role !== "system"),
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
        stream: true,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      callbacks.onError(new Error(`Claude API Error: ${response.status} - ${error}`));
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError(new Error("No response body"));
      return;
    }

    const decoder = new TextDecoder();
    let fullResponse = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n").filter(line => line.startsWith("data: "));

        for (const line of lines) {
          try {
            const data = JSON.parse(line.replace("data: ", ""));
            if (data.type === "content_block_delta") {
              const token = data.delta?.text || "";
              if (token) {
                fullResponse += token;
                callbacks.onToken(token);
              }
            }
          } catch {
            // Skip malformed lines
          }
        }
      }

      callbacks.onComplete(fullResponse);
    } catch (error) {
      callbacks.onError(error as Error);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              FACTORY FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a Cloud AI client with automatic provider detection
 */
export function createCloudAIClient(apiKey: string, provider?: AIProvider, model?: string): CloudAIClient {
  // Auto-detect provider from API key format
  let detectedProvider: AIProvider = provider || "openrouter";
  
  if (!provider) {
    if (apiKey.startsWith("sk-ant-")) {
      detectedProvider = "anthropic";
    } else if (apiKey.startsWith("sk-") || apiKey.startsWith("sk-proj-")) {
      detectedProvider = "openai";
    } else if (apiKey.startsWith("gsk_")) {
      detectedProvider = "groq";
    } else if (apiKey.length === 39) {
      detectedProvider = "google";
    }
  }

  return new CloudAIClient({
    provider: detectedProvider,
    apiKey,
    model: model || PROVIDER_CONFIGS[detectedProvider].defaultModel,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              ENVIRONMENT CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Get configured Cloud AI client from environment
 */
export function getCloudAIFromEnv(): CloudAIClient | null {
  // Check for API keys in order of preference
  const openaiKey = process.env.NEXT_PUBLIC_OPENAI_API_KEY;
  const geminiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY;
  const anthropicKey = process.env.NEXT_PUBLIC_ANTHROPIC_API_KEY;
  const groqKey = process.env.NEXT_PUBLIC_GROQ_API_KEY;
  const openrouterKey = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY;

  if (openrouterKey) {
    return createCloudAIClient(openrouterKey, "openrouter");
  }
  if (groqKey) {
    return createCloudAIClient(groqKey, "groq");
  }
  if (geminiKey) {
    return createCloudAIClient(geminiKey, "google");
  }
  if (openaiKey) {
    return createCloudAIClient(openaiKey, "openai");
  }
  if (anthropicKey) {
    return createCloudAIClient(anthropicKey, "anthropic");
  }

  return null;
}
