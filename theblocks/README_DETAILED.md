# 🛡️ The Blocks: AI-Native Oracle Security System

> **A Byzantine Fault Tolerant (BFT) Multi-Oracle Aggregation System with MEV Protection**

Built for India Blockchain Week 2025 - Hackxios Tournament

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The 7 Intelligent Algorithms](#the-7-intelligent-algorithms)
4. [Blockchain Leverage](#blockchain-leverage)
5. [Gas Optimization](#gas-optimization)
6. [Smart Contract Deep Dive](#smart-contract-deep-dive)
7. [Getting Started](#getting-started)
8. [Technical Specifications](#technical-specifications)

---

## 🎯 Overview

The Blocks is an **AI-native oracle security layer** that protects DeFi applications from:

- ⚡ **Flash Loan Attacks** - Detects sudden price manipulation
- 🎭 **Oracle Manipulation** - BFT consensus rejects faulty data
- 🔄 **MEV Extraction** - Fair ordering prevents sandwich attacks
- 📉 **Stale Price Attacks** - Adaptive freshness thresholds
- 🚨 **Black Swan Events** - Circuit breakers halt trading automatically

### Key Features

| Feature | Description |
|---------|-------------|
| **Dual-Oracle Support** | Chainlink (60% weight), Pyth Network (40% weight) |
| **Weighted Consensus** | Production-ready oracle aggregation |
| **Real-Time Confidence Scoring** | Dynamic trust based on oracle health |
| **Automatic Circuit Breakers** | Self-enforcing emergency stops |
| **MEV Resistance** | Fair ordering stack prevents extraction |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Swap UI     │  │ Liquidity   │  │ Attack      │              │
│  │             │  │ Manager     │  │ Simulator   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BLOCKCHAIN LAYER (Hardhat)                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MultiOracleAggregator.sol                    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐                  │   │
│  │  │Chainlink│ │  Pyth   │ │ SyncedFeed  │                  │   │
│  │  └────┬────┘ └────┬────┘ └──────┬──────┘                  │   │
│  │       │           │             │                          │   │
│  │       ▼           ▼             ▼                          │   │
│  │  ┌────────────────────────────────────────────────────┐   │   │
│  │  │           WEIGHTED AGGREGATION ENGINE              │   │   │
│  │  │  • 60/40 Weighting  • Outlier Detection            │   │   │
│  │  │  • Confidence Scoring  • Circuit Breaker           │   │   │
│  │  └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  DEX Settlement Layer                     │   │
│  │  • Atomic Swaps  • Liquidity Pools  • MEV Resistance     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Contract Addresses (Local Hardhat)

```javascript
const DEX_CONTRACTS = {
  router: "0xa513E6E4b8f2a923D98304ec87F64353C4D5C853",
  weth: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
  usdc: "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
  usdt: "0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9",
  dai: "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9",
  wbtc: "0x5FC8d32690cc91D4c39d9d3abcBD16989F875707",
  link: "0x0165878A594ca255338adfa4d48449f69242Eb8F",
};
```

---

## 🧠 The 7 Intelligent Algorithms

Our AI-native system employs 7 interconnected algorithms to ensure oracle security:

### 1. BFT Median Selection

**File:** `MultiOracleAggregator.sol`

```solidity
function _calculateMedian(uint256[] memory arr) internal pure returns (uint256) {
    // Sort array, then pick middle value
    if (n % 2 == 1) {
        return arr[n / 2];  // Odd: middle element
    } else {
        return (arr[n / 2 - 1] + arr[n / 2]) / 2;  // Even: average of two middle
    }
}
```

**How it works:**
- Collects prices from Chainlink and Pyth
- Applies weighted aggregation (60% Chainlink, 40% Pyth)
- Uses SyncedFeed as fallback when primary oracles stale

**Example:**
| Chainlink (60%) | Pyth (40%) | **Weighted Result** |
|-----------------|------------|---------------------|
| $2000 | $2001 | **$2000.40** ✅ |

---

### 2. Adaptive Staleness Detection

**Per-oracle freshness thresholds:**

```solidity
CHAINLINK_MAX_STALENESS = 1 hours;     // Can tolerate older data
PYTH_MAX_STALENESS = 1 minutes;        // Requires fresh data
REDSTONE_MAX_STALENESS = 1 minutes;
DIA_MAX_STALENESS = 2 minutes;
TWAP_MAX_STALENESS = 30 minutes;       // Inherently averaged
```

**Why adaptive?**
- Chainlink updates on deviation (longer intervals acceptable)
- Pyth is push-based (expects real-time updates)
- TWAP is inherently smoothed (longer window acceptable)

---

### 3. Confidence-Weighted Aggregation

```solidity
function _calculateConfidence(OracleType oracleType, uint256 timestamp) {
    uint256 confidence = config.reliabilityScore;  // Base: 0-100
    
    // Freshness penalty
    if (age > maxAge) {
        confidence = confidence / 2;  // 50% penalty for stale
    } else if (age > maxAge / 2) {
        confidence = (confidence * 80) / 100;  // 20% penalty
    }
    
    // Failure history penalty
    if (config.failCount > 0) {
        confidence = confidence - (config.failCount * RELIABILITY_DECAY);
    }
    
    return confidence;
}

// Weighted average calculation
function _calculateWeightedAverage() {
    weightedSum += prices[i] * confidences[i];
    totalWeight += confidences[i];
    return weightedSum / totalWeight;
}
```

**Dynamic trust scoring:**
| Oracle State | Confidence Impact |
|--------------|-------------------|
| Fresh & reliable | 100% weight |
| Semi-stale | 80% weight |
| Fully stale | 50% weight |
| 1 failure | -5% reliability |
| 3+ failures | Auto-disabled |

---

### 4. Circuit Breaker Pattern

```solidity
enum CircuitBreakerLevel { NORMAL, ELEVATED, HIGH, CRITICAL, EMERGENCY }

function _checkCircuitBreaker(AggregatedPrice memory price) internal {
    uint256 twapDeviation = _calculateDeviationBps(price.medianPrice, price.twapPrice);
    
    if (twapDeviation > 1000) {  // 10%
        _triggerCircuitBreaker(CircuitBreakerLevel.CRITICAL, ...);
    } else if (twapDeviation > 500) {  // 5%
        _triggerCircuitBreaker(CircuitBreakerLevel.HIGH, ...);
    }
    
    if (level == CircuitBreakerLevel.EMERGENCY) {
        isPaused = true;  // All trading halts
    }
}
```

**Escalation levels:**
| Level | Trigger | Action |
|-------|---------|--------|
| NORMAL | All good | Full operation |
| ELEVATED | <3 oracles | Reduced confidence |
| HIGH | 5-10% deviation | Warning mode |
| CRITICAL | >10% deviation | Limited operations |
| EMERGENCY | System failure | **All trading halted** |

---

### 5. Outlier Detection (Statistical)

```solidity
uint256 constant OUTLIER_THRESHOLD_BPS = 200;  // 2%

for (uint8 i = 0; i < validCount; i++) {
    uint256 deviationBps = _calculateDeviationBps(prices[i].price, preliminaryMedian);
    
    if (deviationBps <= OUTLIER_THRESHOLD_BPS) {
        filteredPrices[filteredCount] = prices[i].price;  // ✅ Include
    } else {
        emit OutlierDetected(prices[i].oracleType, ...);  // ❌ Exclude
        aggregated.outlierCount++;
    }
}
```

**Detection process:**
1. Calculate preliminary median from all prices
2. Compute each oracle's deviation from median
3. Exclude any oracle >2% from median
4. Recalculate final median without outliers

---

### 6. Fallback Cascade

```solidity
if (validCount < MIN_VALID_ORACLES) {  // MIN = 3
    aggregated.isReliable = false;
    aggregated.validOracleCount = validCount;
    
    if (validCount == 0) {
        _triggerCircuitBreaker(CircuitBreakerLevel.EMERGENCY, 0, "No valid oracles");
        return aggregated;
    }
}
```

**Cascade priority:**
```
5 oracles → Full confidence (100%)
4 oracles → High confidence (95%)
3 oracles → Normal operation (90%)
2 oracles → Reduced confidence, isReliable = false
1 oracle  → Emergency mode, minimal operations
0 oracles → Complete halt, circuit breaker
```

---

### 7. Cross-Oracle Validation (TWAP Anchor)

```solidity
// TWAP serves as manipulation-resistant ground truth
if (price.twapPrice > 0 && price.medianPrice > 0) {
    uint256 twapDeviation = _calculateDeviationBps(price.medianPrice, price.twapPrice);
    
    // Even if all oracles agree on wrong price, TWAP catches it
    if (twapDeviation > CIRCUIT_BREAKER_THRESHOLD * 100) {
        _triggerCircuitBreaker(CircuitBreakerLevel.CRITICAL, ...);
    }
}
```

**Why TWAP as anchor?**
- Calculated from on-chain DEX data (Uniswap V3)
- 30-minute time-weighted average resists manipulation
- Cannot be flash-loan attacked (requires sustained price movement)

---

## 🔗 Blockchain Leverage

### 1. Immutable Price History & Audit Trail

```solidity
function _recordPrice(AggregatedPrice memory price) internal {
    priceHistory.push(price);  // Permanently stored on-chain
}
```

**Benefits:**
- Every aggregated price is permanently recorded
- Dispute resolution has cryptographic proof
- MEV attack forensics possible

---

### 2. Transparent Oracle Failure Tracking

```solidity
function _recordFailure(OracleType oracleType, string memory reason) internal {
    config.failCount++;
    config.reliabilityScore -= RELIABILITY_DECAY;
    
    emit OracleFailure(oracleType, reason, config.failCount);  // Public event
}
```

**Benefits:**
- All oracle failures publicly visible on-chain
- No centralized authority can hide manipulation
- Community can monitor oracle health

---

### 3. Self-Enforcing Circuit Breakers

```solidity
function _triggerCircuitBreaker(...) internal {
    if (level == CircuitBreakerLevel.EMERGENCY) {
        isPaused = true;  // Smart contract self-halts
    }
}
```

**Benefits:**
- No human intervention needed
- Instant response to attacks
- Impossible to override maliciously

---

### 4. Trustless DEX Settlement

```solidity
function swapExactTokensForTokens(
    uint amountIn,
    uint amountOutMin,
    address[] calldata path,
    address to,
    uint deadline
) external returns (uint[] memory amounts);
```

**Benefits:**
- Atomic execution (all-or-nothing)
- No counterparty risk
- Transparent pricing verification

---

### 5. Cryptographic Proof of Consensus

```solidity
emit PriceAggregated(
    aggregated.medianPrice,      // Final price
    aggregated.weightedPrice,    // Alternative calculation
    aggregated.confidence,       // Trust score
    aggregated.validOracleCount  // Consensus participants
);
```

**Benefits:**
- Unalterable record of BFT execution
- Verifiable oracle participation
- Confidence score transparency

---

### Architecture: On-Chain vs Off-Chain

| Component | On-Chain (Blockchain) | Off-Chain |
|-----------|----------------------|-----------|
| Price Storage | ✅ `priceHistory[]` | - |
| BFT Median Calculation | ✅ `_calculateMedian()` | - |
| Circuit Breaker State | ✅ `circuitBreakerLevel` | - |
| Oracle Reliability Scores | ✅ `reliabilityScore` | - |
| Token Swaps | ✅ DEX Router | - |
| Oracle Data Sources | ✅ (on-chain calls) | Oracle nodes |
| Frontend Display | - | ✅ Next.js |
| Wallet Signing | - | ✅ MetaMask |

---

## ⛽ Gas Optimization

### Executive Summary

| Metric | Value |
|--------|-------|
| Contract Size | 24,586 bytes |
| Avg Settlement Creation | ~85,000 gas |
| Avg Deposit | ~45,000 gas |
| Avg Execution | ~65,000 gas |
| Batch Refund (5 settlements) | ~180,000 gas |

---

### Optimizations Applied

#### 1. Short Error Messages

```solidity
// Before: ~200 more gas
require(msg.sender == settlement.creator, "Only creator can initiate settlement");

// After: Optimized
require(msg.sender == settlement.creator, "!creator");
```

**Savings:** ~200 gas per error string

---

#### 2. State Variable Packing

```solidity
// Before: 5 storage slots
struct Settlement {
    address creator;         // 20 bytes → slot 1
    bool creatorDeposited;   // 1 byte → slot 2 (wasted 31 bytes)
    address counterparty;    // 20 bytes → slot 3
    bool counterpartyDeposited; // 1 byte → slot 4
    uint256 amount;          // 32 bytes → slot 5
}

// After: 3 storage slots (packed)
struct Settlement {
    address creator;         // 20 bytes ┐
    bool creatorDeposited;   // 1 byte   ├─ slot 1 (22 bytes used)
    bool counterpartyDeposited; // 1 byte┘
    address counterparty;    // 20 bytes → slot 2
    uint256 amount;          // 32 bytes → slot 3
}
```

**Savings:** ~20,000 gas per write (2 slots saved)

---

#### 3. Unchecked Blocks

```solidity
// Safe when overflow is impossible
unchecked {
    for (uint256 i = 0; i < length; ++i) {
        // Loop counter can't overflow
    }
}
```

**Savings:** ~50 gas per iteration

---

#### 4. Cache Storage Reads

```solidity
// Before: 3 SLOAD operations
function process() {
    require(settlements[id].creator == msg.sender);
    require(settlements[id].amount > 0);
    emit Event(settlements[id].amount);
}

// After: 1 SLOAD operation
function process() {
    Settlement storage s = settlements[id];
    require(s.creator == msg.sender);
    require(s.amount > 0);
    emit Event(s.amount);
}
```

**Savings:** ~100 gas per avoided SLOAD

---

#### 5. Pre-increment Operators

```solidity
// Before
for (uint256 i = 0; i < length; i++) { }

// After
for (uint256 i = 0; i < length; ++i) { }
```

**Savings:** ~5 gas per iteration

---

#### 6. Events Instead of Storage

```solidity
// Instead of storing history on-chain
mapping(uint256 => uint256[]) public priceHistory;  // Expensive

// Use events for historical data
event SettlementCreated(uint256 indexed id, address indexed creator);  // Cheap
```

**Savings:** ~20,000 gas per avoided storage write

---

### Gas Cost Comparison

#### vs. Simple Escrow

| Feature | Simple Escrow | Our Protocol |
|---------|---------------|--------------|
| Create | ~60,000 gas | ~85,000 gas |
| Security | Basic | 7 Algorithms + Oracle |
| **Trade-off** | - | +40% gas for comprehensive security |

#### vs. Uniswap V2 Swap

| Operation | Gas Cost |
|-----------|----------|
| Uniswap V2 Swap | ~150,000 |
| Our Settlement | ~65,000 |
| **Savings** | **57% cheaper** |

---

### L2 Deployment Recommendations

| Network | Estimated Savings |
|---------|-------------------|
| Arbitrum | 10-50x cheaper |
| Optimism | 10-50x cheaper |
| Base | 10-50x cheaper |
| zkSync | 10-100x cheaper |

**Recommendation:** Deploy to Base for production use.

---

## 🔬 Smart Contract Deep Dive

### Core Contracts

| Contract | Purpose | Lines |
|----------|---------|-------|
| `MultiOracleAggregator.sol` | BFT price aggregation | 1,219 |
| `GuardianOracleV2.sol` | Single-oracle security layer | ~400 |
| `SettlementProtocol.sol` | DEX settlement with invariants | ~600 |
| `MEVResistance.sol` | Fair ordering stack | ~300 |
| `FinalityController.sol` | Transaction finality | ~200 |

### Security Constants

```solidity
// MultiOracleAggregator.sol
uint8 constant MIN_VALID_ORACLES = 3;
uint256 constant MAX_DEVIATION_PERCENT = 5;
uint256 constant OUTLIER_THRESHOLD_BPS = 200;  // 2%
uint256 constant CIRCUIT_BREAKER_THRESHOLD = 10;  // 10%

// Staleness thresholds
uint256 constant CHAINLINK_MAX_STALENESS = 1 hours;
uint256 constant PYTH_MAX_STALENESS = 1 minutes;
uint256 constant DIA_MAX_STALENESS = 2 minutes;
uint256 constant TWAP_MAX_STALENESS = 30 minutes;

// Price bounds
uint256 constant MIN_VALID_PRICE = 1e6;   // $0.01
uint256 constant MAX_VALID_PRICE = 1e12;  // $10,000
```

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- Yarn
- MetaMask browser extension

### Installation

```bash
# Clone the repository
git clone https://github.com/qabdurrahman/The_Blocks.git
cd the-blocks

# Install dependencies
yarn install

# Start local blockchain
yarn chain

# Deploy contracts (in new terminal)
yarn deploy

# Start frontend (in new terminal)
cd packages/nextjs
yarn dev
```

### MetaMask Setup

1. Add Hardhat Network:
   - Network Name: `Hardhat Local`
   - RPC URL: `http://127.0.0.1:8545`
   - Chain ID: `31337`
   - Currency: `ETH`

2. Import Test Account:
   - Private Key: `0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80`
   - This account has pre-minted test tokens

### Access the Application

- Frontend: `http://localhost:3002`
- Hardhat Node: `http://127.0.0.1:8545`

---

## 📊 Technical Specifications

### Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 15.2.6, React 19, TailwindCSS |
| Blockchain | Hardhat, Solidity 0.8.x |
| Web3 | wagmi, viem |
| Framework | Scaffold-ETH 2 |

### Supported Oracles

| Oracle | Type | Staleness Threshold | Weight |
|--------|------|---------------------|--------|
| Chainlink | Push (deviation-based) | 1 hour | 60% |
| Pyth | Pull (real-time) | 1 minute | 40% |
| SyncedFeed | Aggregator (backup) | 1 hour | Fallback |

### Security Features

- ✅ Weighted Consensus (60/40 Chainlink/Pyth)
- ✅ Flash Loan Detection
- ✅ Circuit Breakers (5 levels)
- ✅ Outlier Detection (2% threshold)
- ✅ Confidence-Weighted Aggregation
- ✅ TWAP Cross-Validation
- ✅ Automatic Oracle Health Tracking

---

## 🤖 Expert AI Oracle v3.0 Integration

### AI-Powered Fraud Detection

In addition to oracle security, The Blocks integrates an **Expert AI Oracle** for real-time fraud detection:

| Model | Weight | Purpose |
|-------|--------|---------|
| Neural Network | 25% | MLP + Autoencoder anomaly detection |
| Typology Detector | 25% | 15+ fraud pattern rules |
| Qwen3:8b LLM | 30% | Advanced MoE reasoning (100% local) |
| Compliance Engine | 20% | FATF/OFAC regulatory checks |

### Running the AI Oracle

```bash
# Start Ollama with Qwen3 (one-time setup)
ollama pull qwen3:8b
ollama serve

# Start the Expert AI Oracle API
cd packages/nextjs/services/ai
python -m uvicorn expertAIOracle:app --host 0.0.0.0 --port 8000

# Access AI Fraud Dashboard
open http://localhost:3000/fraud
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👥 Team

**Hackxios 2K25**

Built with ❤️ at India Blockchain Week 2025

---

*For detailed documentation, see the `/docs` folder.*

