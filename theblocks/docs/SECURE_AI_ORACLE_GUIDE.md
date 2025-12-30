# 🔐 Secure AI Oracle Integration Guide

## The Problem You Identified

You're absolutely right to be concerned! Here's what can go wrong with naive AI + blockchain integration:

```
❌ INSECURE APPROACHES:

1. API Key in Frontend Code
   - Anyone can View Source → steal your API key
   - Attacker uses your key → you pay their bills

2. Direct API Calls from Frontend
   - Network tab reveals all API calls
   - Attacker can intercept and modify responses
   - No proof the AI actually said what you claim

3. Storing API Key in Smart Contract
   - Blockchain is PUBLIC - everything is visible
   - Contract bytecode can be decompiled
   - Game over immediately
```

## Our Secure Architecture

```
✅ SECURE ORACLE PATTERN:

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           YOUR INFRASTRUCTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────┐                    ┌─────────────────────────────────────┐  │
│  │               │                    │     PRIVATE BACKEND SERVER          │  │
│  │   Frontend    │   Only sends tx    │                                     │  │
│  │   (React)     │───data, no keys───▶│  ┌─────────────────────────────┐   │  │
│  │               │                    │  │  Environment Variables:     │   │  │
│  │               │                    │  │  - OPENAI_API_KEY          │   │  │
│  │               │                    │  │  - ORACLE_PRIVATE_KEY      │   │  │
│  │               │                    │  └─────────────────────────────┘   │  │
│  │               │                    │              │                      │  │
│  │               │                    │              ▼                      │  │
│  │               │                    │  ┌─────────────────────────────┐   │  │
│  │               │                    │  │   Call Qwen3 MoE Local LLM  │   │  │
│  │               │◀──signed result────│  │   (RTX 4070 GPU - Offline)  │   │  │
│  │               │                    │  └─────────────────────────────┘   │  │
│  │               │                    │              │                      │  │
│  │               │                    │              ▼                      │  │
│  │               │                    │  ┌─────────────────────────────┐   │  │
│  │               │                    │  │  Sign result with          │   │  │
│  │               │                    │  │  ORACLE_PRIVATE_KEY        │   │  │
│  │               │                    │  │  (Cryptographic proof)     │   │  │
│  │               │                    │  └─────────────────────────────┘   │  │
│  │               │                    │                                     │  │
│  └───────┬───────┘                    └─────────────────────────────────────┘  │
│          │                                                                      │
│          │ Submit signed result                                                │
│          ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                         BLOCKCHAIN (Ethereum)                             │ │
│  │                                                                           │ │
│  │   SecureAIOracle.sol                                                      │ │
│  │   ├── Verify signature matches registered oracle address                 │ │
│  │   ├── Check signature is recent (prevent replay attacks)                 │ │
│  │   ├── Store verified assessment on-chain                                 │ │
│  │   └── Block/flag transactions based on risk score                        │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Security Guarantees

| Attack Vector | Protection |
|--------------|------------|
| API Key Theft | Keys only exist in server environment variables |
| Response Manipulation | Cryptographic signature - can't forge without private key |
| Replay Attack | Timestamps + signature tracking on-chain |
| Oracle Impersonation | Smart contract only accepts registered oracle addresses |
| Front-running | Signature includes transaction ID - can't reuse for other tx |

## Quick Setup (15 minutes)

### Step 1: Environment Variables

Create a `.env` file for your backend (NEVER commit this!):

```bash
# packages/nextjs/.env.local (for local development)

# Qwen3 MoE Local LLM (100% Offline - No API Key Needed!)
# Start Ollama: ollama serve && ollama pull qwen3:8b
QWEN3_MODEL=qwen3:8b
OLLAMA_URL=http://localhost:11434

# (Optional) Cloud AI Fallback
# OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Oracle Private Key (generate a NEW one, don't use your wallet!)
ORACLE_PRIVATE_KEY=0x1234567890abcdef...

# API Security
API_SECRET_KEY=your-random-secret-key-for-rate-limiting
```

### Step 2: Generate Oracle Key

```bash
# In packages/nextjs/services, run:
python -c "from eth_account import Account; a = Account.create(); print(f'Private Key: {a.key.hex()}\nAddress: {a.address}')"
```

Save the private key in your `.env.local` and the address for contract setup.

### Step 3: Deploy Smart Contract

```bash
cd packages/hardhat

# Deploy SecureAIOracle
npx hardhat deploy --tags SecureAIOracle --network sepolia
```

### Step 4: Register Oracle Address

```typescript
// In hardhat console or deploy script
const secureOracle = await ethers.getContract("SecureAIOracle");

// Register your oracle (use address from Step 2)
await secureOracle.registerOracle(
  "0xYourOracleAddress", 
  8000  // 80% trust score
);
```

### Step 5: Start Backend

```bash
cd packages/nextjs/services

# Install dependencies
pip install fastapi uvicorn httpx eth-account web3 pydantic

# Start secure oracle service
uvicorn secureAiOracle:app --port 8001 --reload
```

### Step 6: Use in React

```tsx
import { useSecureAIOracle } from "~~/hooks/useSecureAIOracle";

function PaymentForm() {
  const { analyzeAndSubmit, result, isLoading, error } = useSecureAIOracle(
    "0xYourContractAddress",  // SecureAIOracle contract
    "http://localhost:8001"   // Your backend URL
  );

  const handlePayment = async () => {
    // Analyze with AI and submit proof to blockchain
    const { result, txHash } = await analyzeAndSubmit({
      sender: "0x...",
      recipient: "0x...",
      amount: 5000,
    }, "openai");  // or "anthropic" or "local"

    if (!result.approved) {
      alert(`Transaction blocked! Risk: ${result.risk_score}/100`);
      return;
    }

    // Proceed with payment...
  };
}
```

## Production Deployment

### Option A: Deploy Backend to Render/Railway/Fly.io

```yaml
# render.yaml
services:
  - type: web
    name: ai-oracle
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn secureAiOracle:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # Set manually in dashboard
      - key: ORACLE_PRIVATE_KEY
        sync: false
```

### Option B: Deploy to Vercel (Serverless)

Convert the FastAPI to Next.js API routes for serverless deployment.

### Option C: AWS Lambda + API Gateway

For maximum security, use AWS Secrets Manager for keys.

## Why This is Secure for Web3

### 1. Decentralization
- Multiple oracles can be registered
- Contract can require consensus (2 of 3 oracles agree)
- Bad oracles can be slashed/removed

### 2. Transparency
- All assessments are on-chain and auditable
- Anyone can verify the oracle's track record
- Dispute mechanism can be added

### 3. Cryptographic Proof
- Signature proves the specific oracle made the assessment
- Cannot be forged without the private key
- Timestamps prevent replay attacks

### 4. Cost Efficiency
- AI runs off-chain (no gas costs for computation)
- Only final result stored on-chain
- Can batch multiple assessments

## Common Questions

### Q: Can attackers see my AI prompts?
**A:** They can see the transaction data you send to your backend, but NOT:
- Your API keys
- Your system prompts
- The full AI conversation

### Q: What if my backend goes down?
**A:** Add fallback:
```typescript
// In useSecureAIOracle hook
if (!backendAvailable) {
  // Use local heuristics as fallback
  return analyzeLocal(data);
}
```

### Q: Can someone replay old signatures?
**A:** No:
1. Signatures expire after 5 minutes
2. Each signature hash is tracked on-chain
3. Same signature cannot be submitted twice

### Q: What if Qwen3 is not available?
**A:** The system gracefully falls back:
```python
# Qwen3 runs locally on your GPU (RTX 4070) - always available!
# But if Ollama isn't running:
try:
    result = await ai_oracle.analyze_transaction(data, use_qwen3=True)
except Exception:
    result = await ai_oracle.analyze_transaction(data, use_qwen3=False)  # Neural Net only
```

## Files Created

| File | Purpose |
|------|---------|
| `services/secureAiOracle.py` | Backend service with secure AI integration |
| `contracts/SecureAIOracle.sol` | On-chain signature verification |
| `hooks/useSecureAIOracle.ts` | React hook for frontend integration |

## Next Steps

1. ✅ Start Ollama: `ollama serve && ollama pull qwen3:8b`
2. ✅ Generate oracle private key (separate from your wallet!)
3. ✅ Deploy SecureAIOracle contract
4. ✅ Register oracle address in contract
5. ✅ Start backend service: `cd services/ai && uvicorn api:app --port 8080`
6. ✅ Integrate hook in your payment flow

**100% Offline!** Qwen3 MoE runs on your local GPU (RTX 4070) - no API keys needed!
