/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║  PAYFLOW GEMINI AI SERVICE - CLOUD-ISOLATED KNOWLEDGE ENGINE                          ║
 * ╠══════════════════════════════════════════════════════════════════════════════════════╣
 * ║  Built for Hackxios 2K25 - PayPal & Visa Track                                        ║
 * ║                                                                                       ║
 * ║  🔒 SECURITY ARCHITECTURE:                                                            ║
 * ║  ┌─────────────────────────────────────────────────────────────────────────────────┐  ║
 * ║  │  WEB2 LAYER (THIS SERVICE)              WEB3 LAYER (ISOLATED)                  │  ║
 * ║  │  ┌────────────────────────┐            ┌────────────────────────┐               │  ║
 * ║  │  │  Gemini API (Cloud)    │     🔥     │  Fraud Detection (GPU) │               │  ║
 * ║  │  │  ✅ Documentation Q&A  │  FIREWALL  │  ✅ Transaction Scoring │              │  ║
 * ║  │  │  ✅ Protocol Education │     ║      │  ✅ On-chain Data       │              │  ║
 * ║  │  │  ❌ NO TX DATA ACCESS  │     ║      │  ✅ Wallet Interactions │              │  ║
 * ║  │  └────────────────────────┘            └────────────────────────┘               │  ║
 * ║  └─────────────────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                                       ║
 * ║  📚 KNOWLEDGE BASE: Complete PayFlow Protocol documentation                           ║
 * ║  🚀 STREAMING: Real-time token display                                               ║
 * ║  🧠 MODEL: Gemini 2.0 Flash                                                          ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION - HYBRID MODE: Perplexity Cloud + Local Ollama Fallback
// ═══════════════════════════════════════════════════════════════════════════════

const GEMINI_API_KEY = ""; // Key is handled in /api/perplexity route
const GEMINI_ENDPOINT = "/api/perplexity";
const GEMINI_STREAM_ENDPOINT = "/api/perplexity";

// Local Ollama fallback (Qwen3:8B on RTX 4070)
const OLLAMA_ENDPOINT = "http://localhost:11434/api/generate";
const OLLAMA_MODEL = "qwen3:8b";
let USE_LOCAL_FALLBACK = false; // Auto-enabled if Cloud quota exceeded

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE PAYFLOW KNOWLEDGE BASE
// (Complete literature for Gemini context - NO blockchain/transaction data)
// ═══════════════════════════════════════════════════════════════════════════════

export const PAYFLOW_KNOWLEDGE_BASE = `
# PayFlow Protocol - Complete Knowledge Base

## Executive Summary
PayFlow Protocol is the **Missing Intelligence Layer for Institutional Stablecoin Payments**. 
Built for Hackxios 2K25 (PayPal & Visa Track), it bridges the gap between simple token transfers 
and enterprise-grade programmable money with embedded compliance.

**Market Context (2025):**
- Cross-border payments: $194.6T → $320T by 2032 (JPMorgan)
- Stablecoin volume: $15.6T in 2024 (matching Visa)
- B2B payment failure rate: 14% (legacy) vs 0% (PayFlow)
- Settlement time: 3-5 days (legacy) vs 12 seconds (PayFlow)

---

## 🏗️ Smart Contracts Architecture

### 1. PayFlowCore.sol (~980 lines)
The central routing engine for programmable payments.

**Key Functions:**
- \`createPayment(recipient, amount, token, conditions)\` - Creates a new payment with conditions
- \`executePayment(paymentId)\` - Executes after all conditions are met
- \`settleWithFX(paymentId, targetCurrency)\` - Cross-currency settlement with oracle rates
- \`cancelPayment(paymentId)\` - Cancels before execution
- \`approvePayment(paymentId)\` - Multi-sig approval

**Payment States:** PENDING → APPROVED → EXECUTED / CANCELLED / EXPIRED

### 2. ComplianceEngine.sol (~500 lines)
5-tier KYC/AML verification system.

**Compliance Tiers:**
| Tier | Daily Limit | Monthly Limit | Requirements |
|------|-------------|---------------|--------------|
| NONE | $1,000 | $5,000 | No verification |
| BASIC | $10,000 | $50,000 | Email + Phone |
| STANDARD | $100,000 | $500,000 | ID + Address |
| ENHANCED | $1,000,000 | $5,000,000 | Enhanced DD |
| INSTITUTIONAL | Unlimited | Unlimited | Full enterprise onboarding |

**Key Functions:**
- \`setComplianceTier(user, tier)\` - Sets user's KYC tier
- \`checkCompliance(sender, recipient, amount)\` - Validates transaction compliance
- \`addSanctionedAddress(address)\` - Blocks sanctioned entities
- \`recordTravelRule(paymentId, originatorHash, beneficiaryHash)\` - FATF compliance

### 3. SmartEscrow.sol (~500 lines)
Programmable conditional escrow with 4 release types.

**Escrow Types:**
- \`TIME_BASED\` - Auto-releases after timestamp (supply chain, invoices)
- \`APPROVAL\` - Requires beneficiary confirmation (service delivery)
- \`ORACLE\` - External verification (GPS, IoT sensors, API callbacks)
- \`MULTI_SIG\` - M-of-N corporate approval (enterprise treasury)

**Key Functions:**
- \`createEscrow(amount, releaseType, conditions)\` - Creates escrow
- \`releaseEscrow(escrowId)\` - Releases funds when conditions met
- \`disputeEscrow(escrowId, reason)\` - Initiates dispute resolution
- \`cancelEscrow(escrowId)\` - Cancels and refunds (if allowed)

### 4. OracleAggregator.sol (~600 lines)
Multi-source FX rate aggregation with manipulation protection.

**Oracle Sources (Weighted):**
- Chainlink: 60% weight (primary)
- Pyth Network: 40% weight (secondary)

**Protection Mechanisms:**
- 12-period TWAP (Time-Weighted Average Price)
- 5% deviation circuit breaker
- 1-hour staleness threshold
- Fallback oracle chain

**Key Functions:**
- \`getRate(baseToken, quoteToken)\` - Gets current exchange rate
- \`getTWAP(pair, periods)\` - Time-weighted average
- \`checkDeviation(newPrice, oldPrice)\` - Circuit breaker check

### 5. AuditRegistry.sol (~500 lines)
Immutable regulatory audit trail.

**Event Types:**
- PAYMENT_CREATED, PAYMENT_EXECUTED, PAYMENT_CANCELLED
- COMPLIANCE_CHECK, KYC_UPDATED, SANCTIONS_HIT
- ESCROW_CREATED, ESCROW_RELEASED, ESCROW_DISPUTED
- TRAVEL_RULE_RECORDED, RATE_UPDATE, CIRCUIT_BREAKER

**Severity Levels:** INFO, WARNING, CRITICAL, ALERT

**Query Functions:**
- \`getEventsByEntity(address)\` - All events for an entity
- \`getEventsByType(eventType)\` - Filter by event type
- \`getEventsByTimeRange(start, end)\` - Time-based queries
- \`exportAuditReport(jurisdiction)\` - Regulatory exports

---

## 💰 Supported Tokens

| Token | Address (Sepolia) | Decimals | Issuer |
|-------|-------------------|----------|--------|
| PYUSD | 0x... | 6 | PayPal |
| USDC | 0x... | 6 | Circle |
| DAI | 0x... | 18 | MakerDAO |
| USDT | 0x... | 6 | Tether |

---

## 🔐 Gasless Transfers (ERC-4337)

PayFlow implements **Account Abstraction** via PayFlowPaymaster.sol:

**How It Works:**
1. User signs transaction (no ETH needed)
2. PayFlowPaymaster sponsors gas fees
3. Protocol deducts fee from transferred stablecoin (0.1%)
4. User receives full amount minus micro-fee

**Eligibility:**
- Whitelisted addresses (verified users)
- Transactions > $10 value
- Non-sanctioned parties

---

## 🧠 AI Fraud Detection System

**4-Model Ensemble Scoring:**
| Model | Weight | Focus |
|-------|--------|-------|
| Neural Network | 25% | Pattern recognition |
| Typology Rules | 25% | Known fraud patterns |
| Qwen3 LLM | 30% | Context understanding |
| Compliance Score | 20% | Regulatory signals |

**Risk Levels:**
- SAFE (0-20): Auto-approved
- LOW (21-40): Standard processing
- MEDIUM (41-60): Enhanced review
- HIGH (61-80): Manual review required
- CRITICAL (81-100): Blocked + Alert

**Fraud Indicators:**
- Rapid repeated transfers
- New address + large amount
- Sanctioned geography
- Structuring patterns
- Time-of-day anomalies

---

## 🌍 Compliance & Regulations

**FATF Travel Rule (Regulation 16):**
- Applies to transfers > $3,000 (US) / proposed $250
- Requires originator + beneficiary data sharing
- PayFlow hashes data on-chain for privacy + auditability

**Supported Jurisdictions:**
- USA (FinCEN), EU (6AMLD), UK (FCA)
- Singapore (MAS), Japan (FSA), UAE (CBUAE)
- 85+ countries enforcing in 2025

---

## 🔮 How PayFlow Compares

**vs Visa:** Visa settles USDC but lacks programmability. PayFlow adds conditional logic.
**vs PayPal:** PYUSD is a stablecoin. PayFlow makes it *smart money* with escrow + compliance.
**vs Stripe/Bridge:** Developer APIs, but compliance is your problem. PayFlow embeds it.
**vs SWIFT:** ISO 20022 modernization is slow. PayFlow is live today.
**vs JPM/Mastercard:** Private rails for banks only. PayFlow is open to everyone.

---

## 📊 Key Metrics

- **Settlement Time:** 12 seconds (1 Ethereum block)
- **Transaction Cost:** ~$0.50 in gas (Sepolia)
- **Failure Rate:** 0% (vs 14% traditional B2B)
- **Audit Query Time:** < 1 second
- **Oracle Latency:** < 3 seconds for FX rates

---

## 🚀 Quick Start Examples

**Creating a Compliant Payment:**
\`\`\`
1. Sender has STANDARD tier
2. Recipient has BASIC tier  
3. Amount: $50,000 USDC
4. Travel Rule data attached
5. Execute → 12 second settlement
\`\`\`

**Setting Up Time-Based Escrow:**
\`\`\`
1. Lock $100,000 PYUSD
2. Release Type: TIME_BASED
3. Release After: 30 days
4. Beneficiary: Supplier address
5. Auto-releases on timestamp
\`\`\`

---

## ❓ Common Questions

**Q: How do I get a higher KYC tier?**
A: Complete verification through the ComplianceEngine. BASIC needs email/phone, STANDARD needs ID, ENHANCED needs full due diligence.

**Q: What happens if oracles disagree?**
A: The 5% circuit breaker triggers, pausing settlement until rates stabilize or manual intervention.

**Q: Can I dispute an escrow payment?**
A: Yes, call disputeEscrow() before the release condition is met. This pauses release and triggers arbitration.

**Q: How does gasless work?**
A: The PayFlowPaymaster pays your gas. A tiny fee (0.1%) is taken from the transferred tokens.

**Q: Is my data private?**
A: Travel Rule data is hashed (SHA-256) before storing on-chain. Raw data stays off-chain with the counterparties.

---

## 🎯 Hackxios 2K25 Context

PayFlow was built specifically for the **PayPal & Visa Track**:
- Integrates with PYUSD (PayPal's stablecoin)
- Matches Visa's settlement standards
- Solves the 14% B2B failure problem
- Enables 12-second cross-border settlement

**Team:** Built with ❤️ for Hackxios 2K25
**Network:** Ethereum Sepolia (testnet) + Mainnet ready
`;

// ═══════════════════════════════════════════════════════════════════════════════
// SECURITY ISOLATION - NO BLOCKCHAIN DATA ALLOWED
// ═══════════════════════════════════════════════════════════════════════════════

const ISOLATION_RULES = `
⚠️ CRITICAL SECURITY RULES - YOU MUST FOLLOW THESE:

1. **NEVER** access, request, or discuss specific wallet addresses
2. **NEVER** reference actual transaction hashes or amounts
3. **NEVER** provide real-time blockchain data
4. **NEVER** execute or simulate blockchain transactions
5. **ONLY** answer questions about:
   - PayFlow Protocol documentation
   - How features work conceptually
   - Compliance requirements
   - General best practices
   - Educational explanations

If asked about specific transactions, wallets, or real-time data, respond:
"I'm the PayFlow educational assistant. For transaction-specific queries, please use the on-chain interface."

This chatbot is completely ISOLATED from the blockchain for security.
`;

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM PROMPT
// ═══════════════════════════════════════════════════════════════════════════════

const SYSTEM_PROMPT = `You are **PayFlow AI Assistant**, an expert guide for the PayFlow Protocol.

${ISOLATION_RULES}

${PAYFLOW_KNOWLEDGE_BASE}

**Response Style:**
- Be concise but thorough (3-5 paragraphs max)
- Use markdown formatting (headers, bold, lists)
- Reference specific functions when applicable
- End with 2-3 suggested follow-up questions
- Be friendly and helpful

**Format for suggestions:**
At the end, add: "---" followed by suggestions in format:
[Suggestion 1] | [Suggestion 2] | [Suggestion 3]
`;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface GeminiMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface GeminiResponse {
  response: string;
  suggestions: string[];
  responseTime: number;
  tokensUsed?: number;
}

export interface GeminiStreamCallbacks {
  onToken: (token: string) => void;
  onComplete: (response: GeminiResponse) => void;
  onError: (error: Error) => void;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN SERVICE CLASS (ADAPTED FOR PERPLEXITY)
// ═══════════════════════════════════════════════════════════════════════════════

export class GeminiService {
  private apiKey: string;
  private conversationHistory: GeminiMessage[] = [];
  private maxHistoryLength: number = 10;
  private useLocalFallback: boolean = false;

  constructor(apiKey: string = GEMINI_API_KEY) {
    this.apiKey = apiKey;
  }

  /**
   * Check if Ollama is available locally
   */
  private async checkOllamaHealth(): Promise<boolean> {
    try {
      const response = await fetch("http://localhost:11434/api/tags", {
        method: "GET",
        signal: AbortSignal.timeout(2000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Send message to local Ollama
   */
  private async sendToOllama(message: string): Promise<string> {
    const systemPrompt = `You are PayFlow AI Assistant - a helpful assistant for the PayFlow Protocol.
PayFlow is a Web3 payment protocol with:
- Smart Escrow (time-based, approval, oracle, multi-sig)
- Compliance Engine (5 KYC tiers: NONE, BASIC, STANDARD, ENHANCED, INSTITUTIONAL)
- Oracle Aggregator (Chainlink, RedStone, Tellor, Band, DIA)
- AI Fraud Detection (neural heuristics + GPU verification)
- Gasless transfers via Paymaster

Answer questions about PayFlow concisely. Keep responses under 300 words.`;

    const response = await fetch(OLLAMA_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: `${systemPrompt}\n\nUser: ${message}\n\nAssistant:`,
        stream: false,
        options: {
          temperature: 0.7,
          num_predict: 512,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.status}`);
    }

    const data = await response.json();
    // Remove /think tags from Qwen3 response
    let text = data.response || "No response generated.";
    text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    return text;
  }

  /**
   * Check if the AI service is available (Perplexity or Ollama fallback)
   */
  async checkHealth(): Promise<boolean> {
    try {
      // First try Perplexity
      const response = await fetch(GEMINI_ENDPOINT, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "sonar",
          messages: [{ role: "user", content: "ping" }],
          max_tokens: 5
        }),
        signal: AbortSignal.timeout(5000),
      });
      
      if (response.ok) {
        this.useLocalFallback = false;
        USE_LOCAL_FALLBACK = false;
        return true;
      }
      
      // If Perplexity fails, try Ollama fallback
      console.log("Perplexity check failed, switching to local Ollama fallback...");
      const ollamaOk = await this.checkOllamaHealth();
      if (ollamaOk) {
        this.useLocalFallback = true;
        USE_LOCAL_FALLBACK = true;
        return true;
      }
      
      return false;
    } catch {
      // Try Ollama as fallback
      const ollamaOk = await this.checkOllamaHealth();
      if (ollamaOk) {
        this.useLocalFallback = true;
        USE_LOCAL_FALLBACK = true;
        return true;
      }
      return false;
    }
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }

  /**
   * Validate that query is PayFlow-related
   */
  private isPayFlowRelated(query: string): boolean {
    const keywords = [
      "payflow", "payment", "transfer", "escrow", "oracle", "compliance",
      "kyc", "aml", "sanctions", "travel rule", "stablecoin", "pyusd",
      "usdc", "dai", "usdt", "gasless", "paymaster", "audit", "fraud",
      "risk", "tier", "limit", "settlement", "cross-border", "fx", "rate",
      "contract", "execute", "create", "approve", "cancel", "how", "what",
      "why", "when", "explain", "help", "work", "use", "hackxios", "visa"
    ];
    const lowerQuery = query.toLowerCase();
    return keywords.some(kw => lowerQuery.includes(kw)) || query.length < 25;
  }

  /**
   * Parse suggestions from response
   */
  private parseSuggestions(response: string): { cleanResponse: string; suggestions: string[] } {
    let cleanResponse = response;
    let suggestions: string[] = [];

    const suggestionMatch = response.match(/---\s*\n?(.*?)$/s);
    if (suggestionMatch) {
      cleanResponse = response.replace(/---\s*\n?.*$/s, "").trim();
      const suggestionText = suggestionMatch[1];
      suggestions = suggestionText
        .split("|")
        .map(s => s.replace(/^\[|\]$/g, "").trim())
        .filter(s => s.length > 0 && s.length < 80)
        .slice(0, 3);
    }

    if (suggestions.length === 0) {
      suggestions = [
        "How does compliance work?",
        "Tell me about escrow types",
        "Explain gasless transfers"
      ];
    }

    return { cleanResponse, suggestions };
  }

  /**
   * Send a message and get a response (non-streaming)
   */
  async sendMessage(userMessage: string): Promise<GeminiResponse> {
    const startTime = Date.now();

    // Topic validation
    if (!this.isPayFlowRelated(userMessage)) {
      return {
        response: "I specialize in **PayFlow Protocol**. I can help with payments, compliance, escrow, oracles, fraud detection, and gasless transfers. What would you like to know about PayFlow?",
        suggestions: ["How do payments work?", "What are compliance tiers?", "Explain fraud detection"],
        responseTime: Date.now() - startTime,
      };
    }

    // Add to history
    this.conversationHistory.push({
      role: "user",
      content: userMessage,
    });

    // Trim history if too long
    if (this.conversationHistory.length > this.maxHistoryLength * 2) {
      this.conversationHistory = this.conversationHistory.slice(-this.maxHistoryLength * 2);
    }

    try {
      // Use local Ollama fallback if needed
      if (this.useLocalFallback || USE_LOCAL_FALLBACK) {
        const ollamaResponse = await this.sendToOllama(userMessage);
        
        this.conversationHistory.push({
          role: "assistant",
          content: ollamaResponse,
        });
        
        const { cleanResponse, suggestions } = this.parseSuggestions(ollamaResponse);
        
        return {
          response: cleanResponse,
          suggestions,
          responseTime: Date.now() - startTime,
        };
      }

      // Use Perplexity API
      const response = await fetch(GEMINI_ENDPOINT, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: "sonar",
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            ...this.conversationHistory
          ],
          temperature: 0.7,
          max_tokens: 1024,
        }),
      });

      if (!response.ok) {
        // If quota exceeded or other error, switch to fallback
        console.log(`Perplexity error ${response.status}, switching to Ollama fallback...`);
        this.useLocalFallback = true;
        USE_LOCAL_FALLBACK = true;
        
        const ollamaResponse = await this.sendToOllama(userMessage);
        
        this.conversationHistory.push({
          role: "assistant",
          content: ollamaResponse,
        });
        
        const { cleanResponse, suggestions } = this.parseSuggestions(ollamaResponse);
        
        return {
          response: cleanResponse,
          suggestions,
          responseTime: Date.now() - startTime,
        };
      }

      const data = await response.json();
      const rawResponse = data.choices?.[0]?.message?.content || "No response generated.";

      // Add assistant response to history
      this.conversationHistory.push({
        role: "assistant",
        content: rawResponse,
      });

      const { cleanResponse, suggestions } = this.parseSuggestions(rawResponse);

      return {
        response: cleanResponse,
        suggestions,
        responseTime: Date.now() - startTime,
        tokensUsed: data.usage?.total_tokens,
      };
    } catch (error) {
      // Remove failed message from history
      this.conversationHistory.pop();
      
      // Try Ollama fallback on any error
      try {
        const ollamaOk = await this.checkOllamaHealth();
        if (ollamaOk) {
          this.useLocalFallback = true;
          USE_LOCAL_FALLBACK = true;
          
          this.conversationHistory.push({
            role: "user",
            content: userMessage,
          });
          
          const ollamaResponse = await this.sendToOllama(userMessage);
          
          this.conversationHistory.push({
            role: "assistant",
            content: ollamaResponse,
          });
          
          const { cleanResponse, suggestions } = this.parseSuggestions(ollamaResponse);
          
          return {
            response: cleanResponse,
            suggestions,
            responseTime: Date.now() - startTime,
          };
        }
      } catch {
        // Fallback also failed
      }
      
      throw error;
    }
  }

  /**
   * Send a message with streaming response
   */
  async sendMessageStreaming(
    userMessage: string,
    callbacks: GeminiStreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const startTime = Date.now();

    // Topic validation
    if (!this.isPayFlowRelated(userMessage)) {
      const offTopicResponse = "I specialize in **PayFlow Protocol**. I can help with payments, compliance, escrow, oracles, fraud detection, and gasless transfers. What would you like to know about PayFlow?";
      callbacks.onToken(offTopicResponse);
      callbacks.onComplete({
        response: offTopicResponse,
        suggestions: ["How do payments work?", "What are compliance tiers?", "Explain fraud detection"],
        responseTime: Date.now() - startTime,
      });
      return;
    }

    // Add to history
    this.conversationHistory.push({
      role: "user",
      content: userMessage,
    });

    try {
      const response = await fetch(GEMINI_STREAM_ENDPOINT, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "sonar",
          messages: [
            { role: "system", content: SYSTEM_PROMPT },
            ...this.conversationHistory
          ],
          temperature: 0.7,
          max_tokens: 1024,
          stream: true
        }),
        signal,
      });

      if (!response.ok) {
        console.log(`Perplexity error ${response.status}, using Ollama fallback...`);
        this.useLocalFallback = true;
        USE_LOCAL_FALLBACK = true;
        
        const ollamaResponse = await this.sendToOllama(userMessage);
        
        // Simulate streaming
        callbacks.onToken(ollamaResponse);
        
        this.conversationHistory.push({
          role: "assistant",
          content: ollamaResponse,
        });
        
        const { cleanResponse, suggestions } = this.parseSuggestions(ollamaResponse);
        
        callbacks.onComplete({
          response: cleanResponse,
          suggestions,
          responseTime: Date.now() - startTime,
        });
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let fullResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ") && line !== "data: [DONE]") {
            try {
              const data = JSON.parse(line.slice(6));
              const token = data.choices?.[0]?.delta?.content || "";
              if (token) {
                fullResponse += token;
                callbacks.onToken(token);
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }

      // Add to history
      this.conversationHistory.push({
        role: "assistant",
        content: fullResponse,
      });

      const { cleanResponse, suggestions } = this.parseSuggestions(fullResponse);

      callbacks.onComplete({
        response: cleanResponse,
        suggestions,
        responseTime: Date.now() - startTime,
      });
    } catch (error) {
      this.conversationHistory.pop();
      
      // Try Ollama fallback on error
      try {
        const ollamaOk = await this.checkOllamaHealth();
        if (ollamaOk) {
          this.useLocalFallback = true;
          USE_LOCAL_FALLBACK = true;
          
          this.conversationHistory.push({
            role: "user",
            content: userMessage,
          });
          
          const ollamaResponse = await this.sendToOllama(userMessage);
          
          callbacks.onToken(ollamaResponse);
          
          this.conversationHistory.push({
            role: "assistant",
            content: ollamaResponse,
          });
          
          const { cleanResponse, suggestions } = this.parseSuggestions(ollamaResponse);
          
          callbacks.onComplete({
            response: cleanResponse,
            suggestions,
            responseTime: Date.now() - startTime,
          });
          return;
        }
      } catch {
        // Fallback also failed
      }
      
      if ((error as Error).name === "AbortError") {
        callbacks.onComplete({
          response: "Response cancelled.",
          suggestions: [],
          responseTime: Date.now() - startTime,
        });
      } else {
        callbacks.onError(error as Error);
      }
    }
  }

  /**
   * Get current model being used
   */
  getActiveModel(): string {
    return this.useLocalFallback || USE_LOCAL_FALLBACK ? "qwen3:8b (Local GPU)" : "sonar (Perplexity)";
  }

  /**
   * Check if using local fallback
   */
  isUsingLocalFallback(): boolean {
    return this.useLocalFallback || USE_LOCAL_FALLBACK;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SINGLETON INSTANCE
// ═══════════════════════════════════════════════════════════════════════════════

export const geminiService = new GeminiService();

export default GeminiService;
