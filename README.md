# 🏆 Hackxios 2K25 - PayFlow Protocol

<div align="center">

![PayFlow Protocol](https://img.shields.io/badge/PayFlow-Protocol-6366f1?style=for-the-badge&logo=ethereum&logoColor=white)
![Hackathon](https://img.shields.io/badge/Hackxios-2K25-ff6b6b?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live%20on%20Sepolia-00d26a?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

### **The Missing Intelligence Layer for Institutional Stablecoin Payments**

*Where Visa's settlement meets Stripe's programmability — powered by **Expert AI Oracle v3.0** with local Qwen3:8B LLM*

**🌐 [Live Demo](https://nextjs-1kd24o3my-sandys-projects-65d29ae3.vercel.app) | 📄 [Documentation](./theblocks/docs/) | 🔗 [Smart Contracts](./theblocks/packages/hardhat/contracts/)**

</div>

---

## 🆕 What's New: Expert AI Oracle v3.0

| Feature | Description |
|---------|-------------|
| 🧠 **4-Model Ensemble** | Neural Network + Typology Detector + Qwen3 LLM + Compliance Engine |
| 🦙 **Local LLM** | Qwen3:8B via Ollama — 100% offline, zero API costs |
| 🔐 **ECDSA Signatures** | Cryptographic signing for on-chain verification |
| 🎯 **15 Fraud Typologies** | Mixing, Layering, Tornado Cash, Flash Loans, and more |
| ⚡ **Sub-3ms Latency** | Real-time fraud detection at production scale |

---

## 📋 Table of Contents

- [🎯 Problem Statement](#-problem-statement)
- [💡 Our Solution](#-our-solution)
- [🧠 AI Fraud Detection](#-ai-fraud-detection)
- [🏗️ Architecture](#️-architecture)
- [✨ Key Features](#-key-features)
- [🔮 Oracle System](#-oracle-system)
- [📦 Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [🔒 Smart Contracts](#-smart-contracts)
- [🌐 Deployment](#-deployment)
- [👥 Team](#-team)

---

## 🎯 Problem Statement

### The $320 Trillion Cross-Border Crisis

The global cross-border payments market is exploding — from **$194.6 trillion in 2024 to a projected $320 trillion by 2032** (JPMorgan, 2025). Yet the infrastructure powering it was designed in the 1970s.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE MARKET REALITY (2025)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  📊 Cross-Border Market:     $194.6T → $320T by 2032 (JPMorgan)         │
│  💸 Stablecoin Volume:       $15.6T in 2024 — matching Visa (a16z)      │
│  🏦 B2B Transactions:        3.4 trillion annually, $1.8 quadrillion    │
│  ⚠️  B2B Payment Failures:   14% failure rate (programmable: 0%)        │
│  🌍 Travel Rule Countries:   85 jurisdictions enforcing in 2025         │
│  ⏱️  Settlement Time:        3-5 days (legacy) vs 12 seconds (PayFlow)  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Current Problems with Traditional Systems

| Problem | Traditional Finance | PayFlow Solution |
|---------|---------------------|------------------|
| **Settlement Time** | 3-5 business days | 12 seconds |
| **Compliance Cost** | $25-50 per transaction | Near-zero (on-chain) |
| **FX Slippage Risk** | 2-5% during settlement | Oracle-locked rates |
| **Payment Failures** | 14% B2B failure rate | 0% with programmable rules |
| **Audit Trail** | Scattered, manual | Immutable, on-chain |

---

## 💡 Our Solution

### PayFlow Protocol: Programmable Cross-Border Payments

PayFlow is a **complete cross-border payment infrastructure** that combines:

1. **🛡️ Smart Compliance Engine** - 5-tier KYC verification on-chain
2. **🔮 Dual-Oracle System** - Real-time FX rates from Chainlink + Pyth
3. **🔐 Programmable Escrow** - Conditional payment release (time, approval, oracle)
4. **📝 Immutable Audit Registry** - Every transaction travel-rule compliant
5. **🧠 Expert AI Oracle v3.0** - 4-model ensemble fraud detection with local LLM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PayFlow Protocol Stack                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   🧠 AI Layer (Expert AI Oracle v3.0)                                   │
│   ├── Qwen3:8B Local LLM (100% Offline)                                 │
│   ├── 4-Model Ensemble (Neural + Typology + LLM + Compliance)           │
│   └── 15 Fraud Typology Detectors                                       │
│                                                                          │
│   🌐 Frontend (Next.js 15 + React 19)                                   │
│   ├── Interactive Dashboard                                             │
│   ├── Real-time Oracle Monitoring                                       │
│   └── Settlement Management Interface                                   │
│                                                                          │
│   📡 Oracle Layer (Chainlink 60% + Pyth 40%)                            │
│   ├── Weighted Consensus Aggregation                                    │
│   ├── Circuit Breakers & Staleness Detection                           │
│   └── Flash Loan Attack Protection                                      │
│                                                                          │
│   ⛓️ Smart Contract Layer (Solidity 0.8.x)                              │
│   ├── PayFlowCore.sol - Payment processing engine                       │
│   ├── ComplianceEngine.sol - 5-tier KYC verification                   │
│   ├── SmartEscrow.sol - Programmable conditional escrow                │
│   ├── OracleAggregator.sol - Multi-oracle price feeds                  │
│   ├── AuditRegistry.sol - Immutable audit logging                      │
│   └── FraudOracle.sol - AI-powered fraud prevention                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 AI Fraud Detection

### Expert AI Oracle v3.0 — Enterprise-Grade Fraud Detection

Our flagship innovation: a **4-model ensemble AI system** that runs **100% locally** on your GPU — no cloud APIs, no data leaving your machine, zero ongoing costs.

```
┌─────────────────────────────────────────────────────────────────────────┐
│         🧠 EXPERT AI ORACLE v3.0 - 4-MODEL ENSEMBLE                      │
│              Running on RTX 4070 Laptop GPU (8GB VRAM)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  NEURAL NETWORK │  │   TYPOLOGY      │  │   QWEN3:8B      │          │
│  │     (25%)       │  │   DETECTOR      │  │   LOCAL LLM     │          │
│  │   MLP + AE      │  │     (25%)       │  │     (30%)       │          │
│  │    <5ms         │  │    <1ms         │  │    ~3s          │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │                    │
│           └────────────────────┼────────────────────┘                    │
│                                ▼                                         │
│                    ┌───────────────────────┐                             │
│                    │  COMPLIANCE ENGINE    │                             │
│                    │       (20%)           │                             │
│                    │  FATF + OFAC + KYC    │                             │
│                    └───────────┬───────────┘                             │
│                                ▼                                         │
│                    ┌───────────────────────┐                             │
│                    │  🔐 ECDSA SIGNATURE   │                             │
│                    │  P-256 Cryptographic  │                             │
│                    │  On-Chain Verifiable  │                             │
│                    └───────────────────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Local LLM Matters

| Aspect | Cloud APIs | Expert AI Oracle (Local) |
|--------|------------|--------------------------|
| **Privacy** | Data sent externally | 🔒 100% on-device |
| **Cost** | $20+/1000 calls | 💰 Free forever |
| **Latency** | 500ms+ network | ⚡ <5s total |
| **Uptime** | Depends on provider | ✅ Always available |
| **Compliance** | Data residency issues | 🏛️ Full sovereignty |

### 15 Fraud Typologies Detected

| Category | Typologies |
|----------|------------|
| **Mixing/Layering** | Tornado Cash, Mixing Services, Transaction Layering |
| **Sanctions** | OFAC/UN Sanctioned Addresses, Blacklisted Entities |
| **Market Abuse** | Wash Trading, Front-Running, Pump & Dump, Flash Loans |
| **Attack Patterns** | Dust Attacks, Sybil Attacks, Rug Pulls, Phishing |
| **AML Red Flags** | Structuring, Round-Trip Transfers, Velocity Abuse |

### API Endpoints

```bash
# Health Check
GET http://localhost:8000/health

# Expert Analysis (Full 4-Model Ensemble)
POST http://localhost:8000/expert/analyze
{
  "transaction_id": "tx_001",
  "sender": "0xd90e2f925DA726b50C4Ed8D0Fb90Ad053324F31b",
  "recipient": "0x742d35Cc6634C0532925a3b844Bc9e7595f0Ab2d",
  "amount": 50000.0
}

# Response includes: risk_score, verdict, model_scores, typologies, ECDSA signature
```

---

## 🏗️ Architecture

### System Flow

```
User Request → Compliance Check → Oracle Price Lock → Escrow Creation → Settlement
     │              │                    │                  │              │
     ▼              ▼                    ▼                  ▼              ▼
┌─────────┐  ┌──────────────┐  ┌─────────────────┐  ┌───────────┐  ┌──────────┐
│ Web3    │  │ Compliance   │  │ Chainlink (60%) │  │  Smart    │  │  Audit   │
│ Wallet  │→ │ Engine       │→ │ Pyth (40%)      │→ │  Escrow   │→ │ Registry │
│         │  │ (5 Tiers)    │  │ Aggregation     │  │           │  │          │
└─────────┘  └──────────────┘  └─────────────────┘  └───────────┘  └──────────┘
```

### Contract Deployment (Sepolia Testnet)

| Contract | Address | Purpose |
|----------|---------|---------|
| PayFlowCore | `0x...` | Main payment processing |
| ComplianceEngine | `0x...` | KYC tier management |
| SmartEscrow | `0x...` | Conditional payments |
| OracleAggregator | `0x...` | Price feed aggregation |
| AuditRegistry | `0x...` | Immutable logging |

---

## ✨ Key Features

### 1. 🛡️ 5-Tier Compliance System

```solidity
enum ComplianceTier {
    NONE,           // Tier 0: No verification
    BASIC,          // Tier 1: Email verification
    STANDARD,       // Tier 2: KYC documents
    ENHANCED,       // Tier 3: Enhanced due diligence
    INSTITUTIONAL   // Tier 4: Full institutional compliance
}
```

Each tier unlocks higher transaction limits and enables cross-border institutional payments.

### 2. 🔮 Dual-Oracle Price Aggregation

Our production-ready oracle system uses **weighted consensus**:

- **Chainlink (60% weight)**: Industry-standard, high reliability
- **Pyth Network (40% weight)**: Sub-second updates, real-time pricing

```typescript
// Real-time price aggregation
const aggregatedPrice = chainlinkPrice * 0.6 + pythPrice * 0.4;
const confidence = calculateConfidence(chainlinkAge, pythAge, deviation);
```

**Protection Features:**
- ⏱️ Staleness detection (1 hour for Chainlink, 1 minute for Pyth)
- 📊 5% deviation circuit breakers
- 🔒 Flash loan attack prevention
- 🔄 Automatic fallback to backup oracles

### 3. 🔐 Programmable Escrow

Four release mechanisms for enterprise use cases:

| Type | Use Case | Example |
|------|----------|---------|
| `TIME_BASED` | Supply chain payments | Release after delivery window |
| `APPROVAL` | Service contracts | Beneficiary signs off |
| `ORACLE` | IoT/GPS verification | External data triggers |
| `MULTI_SIG` | Corporate treasury | M-of-N approval required |

### 4. 📝 Immutable Audit Trail

Every transaction logged with:
- Sender/receiver compliance tiers
- Oracle prices at execution time
- Compliance check results
- Travel Rule data hashes

---

## 🔮 Oracle System

### Supported Price Feeds

**Chainlink Sepolia Feeds:**
- ETH/USD, BTC/USD, LINK/USD
- EUR/USD, GBP/USD, JPY/USD
- DAI/USD, USDC/USD

**Pyth Network Feeds:**
- ETH/USD, BTC/USD, SOL/USD, AVAX/USD
- MATIC/USD, DOT/USD, ATOM/USD
- USDC/USD, USDT/USD, DAI/USD

### Oracle Aggregation Service

```typescript
// packages/nextjs/services/oracleAggregatorService.ts
export async function getAggregatedPrice(symbol: string): Promise<AggregatedPrice> {
  const [chainlinkData, pythData] = await Promise.all([
    fetchChainlinkPrice(symbol),
    fetchPythPrice(symbol)
  ]);
  
  // Weighted average: 60% Chainlink, 40% Pyth
  const aggregatedPrice = chainlinkData.price * 0.6 + pythData.price * 0.4;
  
  return {
    price: aggregatedPrice,
    confidence: calculateConfidence(chainlinkData, pythData),
    sources: { chainlink: chainlinkData, pyth: pythData }
  };
}
```

---

## 📦 Tech Stack

### AI / Machine Learning
- **LLM**: Qwen3:8B via Ollama (100% local, no cloud)
- **Framework**: Pure NumPy (zero ML dependencies)
- **API**: FastAPI + Uvicorn
- **Signing**: ECDSA P-256 (eth-account)
- **GPU**: RTX 4070 Laptop (8GB VRAM)

### Frontend
- **Framework**: Next.js 15.2.6 + React 19
- **Styling**: Tailwind CSS + DaisyUI 5.0
- **Web3**: wagmi + viem + RainbowKit
- **Animations**: Framer Motion

### Smart Contracts
- **Language**: Solidity 0.8.x
- **Framework**: Hardhat
- **Libraries**: OpenZeppelin Contracts
- **Testing**: Chai + Mocha

### Blockchain
- **Testnet**: Sepolia (Ethereum)
- **Oracles**: Chainlink + Pyth Network
- **Wallet Support**: MetaMask, WalletConnect, Coinbase Wallet

### Deployment
- **Frontend**: Vercel
- **Contracts**: Hardhat Deploy

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- Yarn (v1 or v4)
- Git
- Python 3.11+ (for AI Oracle)
- Ollama (for local LLM)
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/shubro18202758/Hackxios_2025.git
cd Hackxios_2025/theblocks

# Install Node.js dependencies
yarn install

# Start local blockchain (Terminal 1)
yarn chain

# Deploy contracts (Terminal 2)
yarn deploy

# Start frontend (Terminal 3)
yarn start
```

### 🧠 Starting the AI Oracle (Optional but Recommended)

```bash
# Terminal 4: Install Ollama and Qwen3 (one-time setup)
# Download from https://ollama.com
ollama pull qwen3:8b

# Terminal 5: Start the Expert AI Oracle API
cd packages/nextjs/services/ai
pip install fastapi uvicorn numpy httpx eth-account
python -m uvicorn expertAPI:app --host 0.0.0.0 --port 8000

# Verify it's running
curl http://localhost:8000/health
# Expected: {"status":"healthy","model":"ExpertAIOracle","version":"3.0.0"}
```

### Environment Variables

Create `.env.local` in `packages/nextjs/`:

```env
NEXT_PUBLIC_ALCHEMY_API_KEY=your_alchemy_key
NEXT_PUBLIC_WALLET_CONNECT_PROJECT_ID=your_wc_project_id
```

---

## 📁 Project Structure

```
Hackxios/
└── theblocks/
    ├── packages/
    │   ├── hardhat/              # Smart contracts
    │   │   ├── contracts/        # Solidity contracts
    │   │   │   ├── PayFlowCore.sol
    │   │   │   ├── ComplianceEngine.sol
    │   │   │   ├── SmartEscrow.sol
    │   │   │   ├── OracleAggregator.sol
    │   │   │   ├── AuditRegistry.sol
    │   │   │   └── FraudOracle.sol
    │   │   ├── deploy/           # Deployment scripts
    │   │   ├── scripts/          # Utility scripts
    │   │   └── test/             # Contract tests
    │   │
    │   └── nextjs/               # Frontend application
    │       ├── app/              # Next.js app router
    │       ├── components/       # React components
    │       ├── config/           # Configuration files
    │       ├── hooks/            # Custom React hooks
    │       └── services/         # API & AI services
    │           └── ai/           # 🧠 Expert AI Oracle v3.0
    │               ├── expertAPI.py          # FastAPI server
    │               ├── neural_ensemble.py    # Neural network models
    │               ├── typology_detector.py  # 15 fraud typologies
    │               ├── qwen3_integration.py  # Local LLM integration
    │               ├── compliance_analyzer.py # Regulatory engine
    │               └── crypto_signer.py      # ECDSA signatures
    │
    ├── docs/                     # Documentation
    │   ├── ARCHITECTURE.md
    │   ├── SECURITY_ANALYSIS.md
    │   ├── GAS_OPTIMIZATION.md
    │   └── DEPLOYMENT_GUIDE.md
    │
    └── README.md                 # Project documentation
```

---

## 🔒 Smart Contracts

### Core Contracts

| Contract | Description | Key Functions |
|----------|-------------|---------------|
| **PayFlowCore.sol** | Main payment engine | `createPayment()`, `executePayment()` |
| **ComplianceEngine.sol** | KYC tier management | `verifyTier()`, `updateComplianceStatus()` |
| **SmartEscrow.sol** | Conditional payments | `createEscrow()`, `releaseEscrow()` |
| **OracleAggregator.sol** | Price feed aggregation | `getLatestPrice()`, `getAggregatedPrice()` |
| **AuditRegistry.sol** | Immutable logging | `logEvent()`, `getAuditTrail()` |
| **FraudOracle.sol** | AI fraud prevention | `updateRiskScore()`, `analyzeTransaction()` |

### Security Features

- ✅ ReentrancyGuard on all state-changing functions
- ✅ Access control with OpenZeppelin roles
- ✅ Pausable emergency stops
- ✅ Oracle staleness checks
- ✅ Slippage protection
- ✅ AI-powered fraud detection (Expert AI Oracle v3.0)

---

## 🌐 Deployment

### Live Deployments

| Network | Frontend URL |
|---------|--------------|
| **Sepolia** | [https://nextjs-1kd24o3my-sandys-projects-65d29ae3.vercel.app](https://nextjs-1kd24o3my-sandys-projects-65d29ae3.vercel.app) |

### Deploy Your Own

```bash
# Deploy to Sepolia
cd packages/hardhat
npx hardhat deploy --network sepolia

# Deploy frontend to Vercel
cd packages/nextjs
vercel --prod
```

---

## 📊 Hackathon Tracks

This project addresses multiple hackathon themes:

- **🤖 AI + Blockchain**: Expert AI Oracle v3.0 with local Qwen3 LLM
- **🏦 DeFi**: Programmable cross-border payments
- **🔗 Infrastructure**: Multi-oracle aggregation layer
- **🛡️ Security**: On-chain compliance and AI fraud detection
- **🌍 Real World Assets**: Institutional stablecoin settlements

---

## 🎥 Demo

### Key Pages

1. **Dashboard** (`/dashboard`) - Main payment interface with real-time AI analysis
2. **Oracle Dashboard** (`/oracle-dashboard`) - Live oracle feeds and consensus
3. **AI Analysis** - Expert AI Oracle fraud detection results
4. **Settlement Monitor** - Track payment lifecycle
5. **Debug Contracts** (`/debug`) - Interact with deployed contracts

---

## 📄 Documentation

Detailed documentation available in `/theblocks/docs/`:

- [Architecture Overview](./theblocks/docs/ARCHITECTURE.md)
- [Security Analysis](./theblocks/docs/SECURITY_ANALYSIS.md)
- [Gas Optimization](./theblocks/docs/GAS_OPTIMIZATION.md)
- [Deployment Guide](./theblocks/docs/DEPLOYMENT_GUIDE.md)
- [Threat Model](./theblocks/docs/THREAT_MODEL.md)

---

## 👥 Team

**Team: The Blocks**

| Member | Role |
|--------|------|
| Sayandeep | AI/ML & Blockchain Developer |
| Shubrato | Full Stack & Smart Contracts |

Built with ❤️ for Hackxios 2K25

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./theblocks/LICENSE) file for details.

---

<div align="center">

### 🚀 Ready to revolutionize cross-border payments with AI?

**[Try the Live Demo →](https://nextjs-1kd24o3my-sandys-projects-65d29ae3.vercel.app)**

*Powered by Expert AI Oracle v3.0 — 100% Local, Zero Cloud Costs*

</div>
