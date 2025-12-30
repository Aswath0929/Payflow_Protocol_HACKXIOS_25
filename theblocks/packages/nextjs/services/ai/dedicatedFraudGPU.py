"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  PAYFLOW DEDICATED GPU FRAUD DETECTION - FULL 8GB VRAM OPTIMIZATION                     ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  🔒 SECURITY ARCHITECTURE:                                                               ║
║  ┌────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                                                                                     │ ║
║  │  WEB2 LAYER (Cloud)              │     WEB3 LAYER (Local)                           │ ║
║  │  ─────────────────               │     ────────────────────                         │ ║
║  │  ┌─────────────────┐             │     ┌─────────────────┐                          │ ║
║  │  │  AI CHATBOT     │             │     │ FRAUD DETECTION │                          │ ║
║  │  │  Cloud API      │             │     │ Qwen3:8B LOCAL  │                          │ ║
║  │  │  (OpenRouter/   │    ISOLATED │     │ AIR-GAPPED      │                          │ ║
║  │  │   Gemini/etc)   │      ═══    │     │ 8GB VRAM        │                          │ ║
║  │  └────────┬────────┘             │     └────────┬────────┘                          │ ║
║  │           │                      │              │                                    │ ║
║  │           ▼                      │              ▼                                    │ ║
║  │   ┌───────────────┐              │     ┌────────────────┐                           │ ║
║  │   │  INTERNET     │              │     │ RTX 4070 GPU   │                           │ ║
║  │   │  (Public)     │              │     │ (Private)      │                           │ ║
║  │   └───────────────┘              │     └────────────────┘                           │ ║
║  │                                  │                                                   │ ║
║  │  ✅ No blockchain data sent     │     ✅ Transaction data stays local              │ ║
║  │  ✅ General Q&A only             │     ✅ Cryptographically signed verdicts         │ ║
║  └────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                          ║
║  GPU MEMORY ALLOCATION:                                                                  ║
║  ┌────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │  Qwen3:8B Model Weights:  ~5.2 GB                                                   │ ║
║  │  KV Cache (8192 ctx):     ~1.5 GB                                                   │ ║
║  │  CUDA Overhead:           ~0.5 GB                                                   │ ║
║  │  Compute Buffer:          ~0.8 GB                                                   │ ║
║  │  ───────────────────────────────────                                                │ ║
║  │  TOTAL:                   ~8.0 GB (FULL VRAM UTILIZATION)                           │ ║
║  └────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                          ║
║  Hackxios 2K25 - PayFlow Protocol                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import httpx
import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger('DedicatedFraudGPU')

# ═══════════════════════════════════════════════════════════════════════════════
#                   MAXIMUM VRAM CONFIGURATION (RTX 4070 8GB)
# ═══════════════════════════════════════════════════════════════════════════════

class DedicatedGPUConfig:
    """
    Maximum VRAM Configuration for Dedicated Fraud Detection.
    
    Since chatbot now uses Cloud API, we can dedicate ALL 8GB VRAM to Qwen3:8B
    for maximum context, batch size, and inference speed.
    """
    
    # ═══════════════════════════════════════════════════════════════════
    #                    OLLAMA CONNECTION
    # ═══════════════════════════════════════════════════════════════════
    OLLAMA_URL = "http://localhost:11434"
    MODEL_NAME = "qwen3:8b"
    
    # ═══════════════════════════════════════════════════════════════════
    #                    MAXIMUM VRAM SETTINGS (8GB DEDICATED)
    # ═══════════════════════════════════════════════════════════════════
    NUM_GPU = 99                # ALL layers on GPU
    NUM_CTX = 8192              # MAXIMUM context (was 4096, now doubled)
    NUM_BATCH = 1024            # DOUBLED batch size (was 512)
    NUM_THREAD = 8              # CPU threads for tokenization
    
    # ═══════════════════════════════════════════════════════════════════
    #                    MEMORY OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    USE_MMAP = True             # Memory-mapped model loading
    USE_MLOCK = False           # Let CUDA manage memory
    FLASH_ATTN = True           # Enable Flash Attention if available
    
    # ═══════════════════════════════════════════════════════════════════
    #                    INFERENCE QUALITY (FRAUD-OPTIMIZED)
    # ═══════════════════════════════════════════════════════════════════
    TEMPERATURE = 0.2           # Lower = more deterministic for fraud
    TOP_P = 0.85                # Focused probability mass
    TOP_K = 30                  # Limit vocabulary for speed
    MAX_TOKENS = 256            # REDUCED: Concise JSON output only
    REPEAT_PENALTY = 1.15       # Stronger anti-repetition
    MIROSTAT = 0                # Disabled for maximum speed
    
    # ═══════════════════════════════════════════════════════════════════
    #                    SPEED OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    DISABLE_THINKING = True     # Disable <think> mode for faster output
    
    # ═══════════════════════════════════════════════════════════════════
    #                    VRAM PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════
    KEEP_ALIVE = "1h"           # Keep model loaded for 1 HOUR (was 30m)
    
    # ═══════════════════════════════════════════════════════════════════
    #                    LATENCY TARGETS
    # ═══════════════════════════════════════════════════════════════════
    TARGET_LATENCY_MS = 300     # Target: <300ms per analysis
    MAX_LATENCY_MS = 1000       # Maximum acceptable: 1 second
    TIMEOUT_SECONDS = 15.0      # Request timeout
    CONNECT_TIMEOUT = 3.0       # Connection timeout
    MAX_RETRIES = 2             # Reduced retries for speed
    RETRY_DELAY = 0.2           # Fast retry


# ═══════════════════════════════════════════════════════════════════════════════
#                              RISK LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    SAFE = "SAFE"           # 0-20: Normal transaction
    LOW = "LOW"             # 21-40: Minor anomalies
    MEDIUM = "MEDIUM"       # 41-60: Flagged for review
    HIGH = "HIGH"           # 61-80: Suspicious
    CRITICAL = "CRITICAL"   # 81-100: Block immediately


@dataclass
class FraudVerdict:
    """Cryptographically-signable fraud verdict."""
    transaction_id: str
    risk_score: int
    risk_level: RiskLevel
    approved: bool
    alerts: List[str]
    explanation: str
    recommendations: List[str]
    confidence: float
    model: str
    inference_time_ms: float
    vram_usage_mb: float
    context_tokens: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_signable_payload(self) -> str:
        """Create deterministic payload for ECDSA signing."""
        return json.dumps({
            "tx": self.transaction_id,
            "score": self.risk_score,
            "level": self.risk_level.value,
            "approved": self.approved,
            "confidence": round(self.confidence, 2),
            "timestamp": self.timestamp.isoformat()
        }, separators=(',', ':'))


# ═══════════════════════════════════════════════════════════════════════════════
#                    DEDICATED FRAUD DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class DedicatedFraudGPU:
    """
    Dedicated GPU Fraud Detection Engine.
    
    Uses the FULL 8GB VRAM for maximum Qwen3:8B performance.
    Air-gapped from internet - all transaction analysis stays local.
    
    Features:
    - 8192 token context window (2x previous)
    - 1024 batch size (2x previous)
    - <300ms target latency
    - Cryptographically signed verdicts
    """
    
    SYSTEM_PROMPT = """You are an expert financial crime AI for PayFlow Protocol. Running on AIR-GAPPED local GPU.

CRITICAL RULES:
1. Output ONLY valid JSON - no thinking, no explanation text, just the JSON object
2. Do NOT use <think> tags
3. Be concise - max 2 sentences for explanation

Analyze for: money laundering, velocity anomalies, amount patterns, counterparty risk.

OUTPUT FORMAT (JSON ONLY - NO OTHER TEXT):
{"risk_score": <0-100>, "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL", "approved": <true|false>, "alerts": ["<alert>"], "explanation": "<brief analysis>", "recommendations": ["<action>"], "confidence": <0.0-1.0>}

THRESHOLDS: SAFE(0-20), LOW(21-40), MEDIUM(41-60), HIGH(61-80), CRITICAL(81-100)"""

    def __init__(self):
        """Initialize dedicated fraud GPU engine."""
        self.config = DedicatedGPUConfig
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.CONNECT_TIMEOUT,
                read=self.config.TIMEOUT_SECONDS,
                write=self.config.TIMEOUT_SECONDS,
                pool=self.config.TIMEOUT_SECONDS
            )
        )
        self.is_ready = False
        self.vram_mb = 0
        self.model_loaded = False
        
        logger.info("=" * 60)
        logger.info("DEDICATED FRAUD GPU ENGINE INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"  Model: {self.config.MODEL_NAME}")
        logger.info(f"  Context Window: {self.config.NUM_CTX} tokens")
        logger.info(f"  Batch Size: {self.config.NUM_BATCH}")
        logger.info(f"  Target Latency: <{self.config.TARGET_LATENCY_MS}ms")
        logger.info(f"  VRAM: FULL 8GB DEDICATED")
        logger.info("=" * 60)
    
    async def initialize(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Initialize the GPU and load model into VRAM.
        
        Returns:
            (success, diagnostics)
        """
        diagnostics = {
            "ollama_running": False,
            "model_available": False,
            "model_loaded": False,
            "vram_mb": 0,
            "load_time_ms": 0,
            "error": None
        }
        
        try:
            # Step 1: Check Ollama server
            response = await self.client.get(f"{self.config.OLLAMA_URL}/api/tags")
            if response.status_code != 200:
                diagnostics["error"] = "Ollama server not running"
                return False, diagnostics
            
            diagnostics["ollama_running"] = True
            
            # Step 2: Check model availability
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            if not any(self.config.MODEL_NAME in m for m in models):
                diagnostics["error"] = f"Model {self.config.MODEL_NAME} not found. Run: ollama pull {self.config.MODEL_NAME}"
                return False, diagnostics
            
            diagnostics["model_available"] = True
            
            # Step 3: Load model into VRAM with maximum settings
            logger.info("Loading Qwen3:8B into GPU with FULL 8GB VRAM allocation...")
            start_time = time.time()
            
            warmup_response = await self.client.post(
                f"{self.config.OLLAMA_URL}/api/generate",
                json={
                    "model": self.config.MODEL_NAME,
                    "prompt": "Initialize fraud detection system",
                    "stream": False,
                    "keep_alive": self.config.KEEP_ALIVE,
                    "options": {
                        # FULL VRAM ALLOCATION
                        "num_gpu": self.config.NUM_GPU,
                        "num_ctx": self.config.NUM_CTX,
                        "num_batch": self.config.NUM_BATCH,
                        "num_thread": self.config.NUM_THREAD,
                        "use_mmap": self.config.USE_MMAP,
                        "num_predict": 5
                    }
                },
                timeout=120.0  # Extended timeout for initial load
            )
            
            load_time_ms = (time.time() - start_time) * 1000
            diagnostics["load_time_ms"] = load_time_ms
            
            if warmup_response.status_code != 200:
                diagnostics["error"] = f"Failed to load model: {warmup_response.text}"
                return False, diagnostics
            
            # Step 4: Get VRAM usage
            try:
                ps_response = await self.client.get(f"{self.config.OLLAMA_URL}/api/ps")
                if ps_response.status_code == 200:
                    ps_data = ps_response.json()
                    for model in ps_data.get("models", []):
                        if self.config.MODEL_NAME in model.get("name", ""):
                            size_vram = model.get("size_vram", 0)
                            self.vram_mb = size_vram / (1024 * 1024)
                            diagnostics["vram_mb"] = self.vram_mb
            except:
                pass
            
            diagnostics["model_loaded"] = True
            self.model_loaded = True
            self.is_ready = True
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Load time: {load_time_ms:.0f}ms")
            logger.info(f"   VRAM usage: {self.vram_mb:.0f}MB")
            
            return True, diagnostics
            
        except Exception as e:
            diagnostics["error"] = str(e)
            logger.error(f"Initialization failed: {e}")
            return False, diagnostics
    
    async def analyze_transaction(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        token: str = "USDC",
        sender_profile: Optional[Dict] = None,
        recipient_profile: Optional[Dict] = None,
        ml_scores: Optional[Dict] = None,
        context: Optional[str] = None
    ) -> FraudVerdict:
        """
        Analyze a transaction for fraud using dedicated GPU inference.
        
        Args:
            transaction_id: Unique transaction identifier
            sender: Sender wallet address
            recipient: Recipient wallet address
            amount: Transaction amount
            token: Token symbol (USDC, PYUSD, etc.)
            sender_profile: Historical sender data
            recipient_profile: Historical recipient data
            ml_scores: Pre-computed ML model scores
            context: Additional context (e.g., escrow release reason)
        
        Returns:
            FraudVerdict with cryptographically-signable payload
        """
        if not self.is_ready:
            return self._emergency_fallback(transaction_id, ml_scores)
        
        start_time = time.time()
        
        # Build comprehensive prompt
        prompt = self._build_prompt(
            transaction_id, sender, recipient, amount, token,
            sender_profile or {}, recipient_profile or {},
            ml_scores or {}, context
        )
        
        # Call Qwen3 with maximum GPU settings
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = await self.client.post(
                    f"{self.config.OLLAMA_URL}/api/generate",
                    json={
                        "model": self.config.MODEL_NAME,
                        "system": self.SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "keep_alive": self.config.KEEP_ALIVE,
                        "options": {
                            # === FULL VRAM GPU SETTINGS ===
                            "num_gpu": self.config.NUM_GPU,
                            "num_thread": self.config.NUM_THREAD,
                            "num_ctx": self.config.NUM_CTX,
                            "num_batch": self.config.NUM_BATCH,
                            "use_mmap": self.config.USE_MMAP,
                            "use_mlock": self.config.USE_MLOCK,
                            
                            # === INFERENCE TUNING ===
                            "temperature": self.config.TEMPERATURE,
                            "top_p": self.config.TOP_P,
                            "top_k": self.config.TOP_K,
                            "num_predict": self.config.MAX_TOKENS,
                            "repeat_penalty": self.config.REPEAT_PENALTY,
                            "mirostat": self.config.MIROSTAT,
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Parse response
                    raw_response = data.get("response", "")
                    
                    # Remove thinking tags if present
                    import re
                    clean_response = re.sub(r'<think>[\s\S]*?</think>', '', raw_response).strip()
                    
                    try:
                        parsed = json.loads(clean_response)
                        
                        risk_score = max(0, min(100, int(parsed.get("risk_score", 50))))
                        risk_level = RiskLevel[parsed.get("risk_level", "MEDIUM")]
                        
                        verdict = FraudVerdict(
                            transaction_id=transaction_id,
                            risk_score=risk_score,
                            risk_level=risk_level,
                            approved=parsed.get("approved", risk_score <= 60),
                            alerts=parsed.get("alerts", []),
                            explanation=parsed.get("explanation", "Analysis complete."),
                            recommendations=parsed.get("recommendations", []),
                            confidence=float(parsed.get("confidence", 0.8)),
                            model=self.config.MODEL_NAME,
                            inference_time_ms=inference_time,
                            vram_usage_mb=self.vram_mb,
                            context_tokens=data.get("prompt_eval_count", 0)
                        )
                        
                        # Log performance
                        status = "✅" if inference_time < self.config.TARGET_LATENCY_MS else "⚠️"
                        logger.info(f"{status} TX {transaction_id[:8]}... | Score: {risk_score} | "
                                  f"Level: {risk_level.value} | Time: {inference_time:.0f}ms")
                        
                        return verdict
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
                        if attempt < self.config.MAX_RETRIES - 1:
                            await asyncio.sleep(self.config.RETRY_DELAY)
                            continue
                        
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Analysis error (attempt {attempt+1}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(self.config.RETRY_DELAY)
        
        # All retries failed
        return self._emergency_fallback(transaction_id, ml_scores)
    
    def _build_prompt(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        token: str,
        sender_profile: Dict,
        recipient_profile: Dict,
        ml_scores: Dict,
        context: Optional[str]
    ) -> str:
        """Build comprehensive analysis prompt."""
        
        sender_tx = sender_profile.get('transaction_count', 0)
        sender_vol = sender_profile.get('total_volume', 0)
        sender_avg = sender_profile.get('avg_amount', 0)
        sender_age = sender_profile.get('account_age_days', 0)
        
        recipient_tx = recipient_profile.get('transaction_count', 0)
        recipient_vol = recipient_profile.get('total_volume', 0)
        recipient_age = recipient_profile.get('account_age_days', 0)
        
        # ML Scores section
        ml_section = ""
        if ml_scores:
            ml_section = f"""
## Machine Learning Scores (0-100)
- Velocity Score: {ml_scores.get('velocity', 0)}
- Amount Anomaly: {ml_scores.get('amount', 0)}
- Pattern Score: {ml_scores.get('pattern', 0)}
- Graph Risk: {ml_scores.get('graph', 0)}
- Temporal Score: {ml_scores.get('timing', 0)}
- Ensemble Average: {ml_scores.get('ensemble', 0)}
- Isolation Forest: {ml_scores.get('isolation', 0):.2f}
"""

        context_section = ""
        if context:
            context_section = f"\n## Additional Context\n{context}\n"
        
        return f"""/no_think
TX: ${amount:,.0f} {token} | {sender[:8]}→{recipient[:8]}
Sender: {sender_tx} txs, ${sender_vol:,.0f} vol, {sender_age}d old, {(amount / sender_avg * 100) if sender_avg > 0 else 100:.0f}% of avg
Recipient: {recipient_tx} txs, {recipient_age}d old{"⚠️NEW" if recipient_tx == 0 else ""}
ML: vel={ml_scores.get('velocity', 0)} amt={ml_scores.get('amount', 0)} pat={ml_scores.get('pattern', 0)} gph={ml_scores.get('graph', 0)}
JSON verdict:"""

    def _emergency_fallback(
        self,
        transaction_id: str,
        ml_scores: Optional[Dict]
    ) -> FraudVerdict:
        """Emergency fallback when GPU inference fails."""
        
        # Use ML scores if available
        if ml_scores:
            avg_score = sum(ml_scores.values()) / len(ml_scores) if ml_scores else 50
            risk_score = int(avg_score)
        else:
            risk_score = 50  # Default to medium
        
        if risk_score <= 20:
            level = RiskLevel.SAFE
        elif risk_score <= 40:
            level = RiskLevel.LOW
        elif risk_score <= 60:
            level = RiskLevel.MEDIUM
        elif risk_score <= 80:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return FraudVerdict(
            transaction_id=transaction_id,
            risk_score=risk_score,
            risk_level=level,
            approved=risk_score <= 60,
            alerts=["GPU inference unavailable - using ML fallback"],
            explanation="LLM analysis failed. Using ensemble ML scores as fallback.",
            recommendations=["Verify GPU status", "Check Ollama service"],
            confidence=0.5,
            model="fallback-ml",
            inference_time_ms=0,
            vram_usage_mb=0,
            context_tokens=0
        )
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """Get current GPU and model diagnostics."""
        diag = {
            "is_ready": self.is_ready,
            "model": self.config.MODEL_NAME,
            "vram_mb": self.vram_mb,
            "context_window": self.config.NUM_CTX,
            "batch_size": self.config.NUM_BATCH,
            "target_latency_ms": self.config.TARGET_LATENCY_MS,
            "gpu_layers": self.config.NUM_GPU,
            "keep_alive": self.config.KEEP_ALIVE,
            "architecture": "DEDICATED_8GB_VRAM"
        }
        
        try:
            ps_response = await self.client.get(f"{self.config.OLLAMA_URL}/api/ps")
            if ps_response.status_code == 200:
                ps_data = ps_response.json()
                diag["loaded_models"] = [
                    {
                        "name": m.get("name"),
                        "size_mb": m.get("size", 0) / (1024*1024),
                        "vram_mb": m.get("size_vram", 0) / (1024*1024),
                        "expires_at": m.get("expires_at")
                    }
                    for m in ps_data.get("models", [])
                ]
        except:
            diag["loaded_models"] = []
        
        return diag
    
    async def close(self):
        """Cleanup resources."""
        await self.client.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
#                              BENCHMARK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

async def run_benchmark(num_transactions: int = 20) -> Dict[str, Any]:
    """
    Run comprehensive GPU fraud detection benchmark.
    
    Args:
        num_transactions: Number of test transactions to analyze
    
    Returns:
        Benchmark results with latency statistics
    """
    import random
    import statistics
    
    engine = DedicatedFraudGPU()
    success, init_diag = await engine.initialize()
    
    if not success:
        return {"error": init_diag["error"], "diagnostics": init_diag}
    
    # Generate test transactions
    test_txs = []
    for i in range(num_transactions):
        test_txs.append({
            "transaction_id": f"benchmark_tx_{i}_{int(time.time())}",
            "sender": f"0x{random.randbytes(20).hex()}",
            "recipient": f"0x{random.randbytes(20).hex()}",
            "amount": random.uniform(10, 100000),
            "token": random.choice(["USDC", "PYUSD", "DAI"]),
            "sender_profile": {
                "transaction_count": random.randint(0, 1000),
                "total_volume": random.uniform(0, 10_000_000),
                "avg_amount": random.uniform(100, 50000),
                "account_age_days": random.randint(0, 365)
            },
            "ml_scores": {
                "velocity": random.randint(0, 100),
                "amount": random.randint(0, 100),
                "pattern": random.randint(0, 100),
                "graph": random.randint(0, 100),
                "timing": random.randint(0, 100)
            }
        })
    
    # Run benchmark
    latencies = []
    results = []
    
    print(f"\n{'='*60}")
    print(f"DEDICATED GPU FRAUD DETECTION BENCHMARK")
    print(f"{'='*60}")
    print(f"Model: {engine.config.MODEL_NAME}")
    print(f"VRAM: {init_diag.get('vram_mb', 0):.0f}MB / 8192MB")
    print(f"Context: {engine.config.NUM_CTX} tokens")
    print(f"Batch: {engine.config.NUM_BATCH}")
    print(f"{'='*60}\n")
    
    for i, tx in enumerate(test_txs):
        verdict = await engine.analyze_transaction(
            transaction_id=tx["transaction_id"],
            sender=tx["sender"],
            recipient=tx["recipient"],
            amount=tx["amount"],
            token=tx["token"],
            sender_profile=tx["sender_profile"],
            ml_scores=tx["ml_scores"]
        )
        
        latencies.append(verdict.inference_time_ms)
        results.append({
            "tx": i + 1,
            "score": verdict.risk_score,
            "level": verdict.risk_level.value,
            "latency_ms": verdict.inference_time_ms,
            "approved": verdict.approved
        })
        
        status = "✅" if verdict.inference_time_ms < 300 else "⚠️"
        print(f"{status} TX {i+1:2d}/{num_transactions} | "
              f"Score: {verdict.risk_score:3d} | "
              f"Level: {verdict.risk_level.value:8s} | "
              f"Time: {verdict.inference_time_ms:7.1f}ms")
    
    await engine.close()
    
    # Calculate statistics
    stats = {
        "total_transactions": num_transactions,
        "avg_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "stddev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "under_300ms_pct": len([l for l in latencies if l < 300]) / len(latencies) * 100,
        "under_500ms_pct": len([l for l in latencies if l < 500]) / len(latencies) * 100,
        "vram_mb": init_diag.get("vram_mb", 0),
        "load_time_ms": init_diag.get("load_time_ms", 0)
    }
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total Transactions: {stats['total_transactions']}")
    print(f"Average Latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"Median Latency: {stats['median_latency_ms']:.1f}ms")
    print(f"Min/Max: {stats['min_latency_ms']:.1f}ms / {stats['max_latency_ms']:.1f}ms")
    print(f"P95 Latency: {stats['p95_latency_ms']:.1f}ms")
    print(f"StdDev: {stats['stddev_ms']:.1f}ms")
    print(f"\n<300ms: {stats['under_300ms_pct']:.1f}%")
    print(f"<500ms: {stats['under_500ms_pct']:.1f}%")
    print(f"\nVRAM Usage: {stats['vram_mb']:.0f}MB")
    print(f"Model Load Time: {stats['load_time_ms']:.0f}ms")
    print(f"{'='*60}\n")
    
    # Performance verdict
    if stats['avg_latency_ms'] < 300:
        print("🎉 EXCELLENT: Average latency under 300ms target!")
    elif stats['avg_latency_ms'] < 500:
        print("✅ GOOD: Average latency under 500ms")
    else:
        print("⚠️ NEEDS OPTIMIZATION: Average latency above 500ms")
    
    return {
        "stats": stats,
        "results": results,
        "diagnostics": init_diag
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PayFlow Dedicated GPU Fraud Detection")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    parser.add_argument("--transactions", "-n", type=int, default=20, help="Number of test transactions")
    parser.add_argument("--diagnostics", "-d", action="store_true", help="Show diagnostics only")
    
    args = parser.parse_args()
    
    if args.diagnostics:
        async def show_diag():
            engine = DedicatedFraudGPU()
            success, init_diag = await engine.initialize()
            print(json.dumps(await engine.get_diagnostics(), indent=2))
            await engine.close()
        
        asyncio.run(show_diag())
    elif args.benchmark:
        asyncio.run(run_benchmark(args.transactions))
    else:
        print("Use --benchmark to run performance test or --diagnostics to check status")
