"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PAYFLOW BLAZING FRAUD ENGINE - v4.0 FINAL                              ║
║                                                                                           ║
║   🔥 MAXIMUM GPU ACCELERATION + OPTIMIZED INFERENCE                                      ║
║                                                                                           ║
║   ⚡ INSTANT: <5ms for 65%+ (pure heuristics)                                            ║
║   🚀 TURBO:  <100ms for 25% (ultra-fast AI)                                              ║
║   🧠 SMART:  <300ms for 8% (quick analysis)                                              ║
║   🔬 DEEP:   <800ms for 2% (thinking mode)                                               ║
║                                                                                           ║
║   ═══════════════════════════════════════════════════════════════════════════════════    ║
║   GPU: RTX 4070 (8GB VRAM) - 100% DEDICATED                                              ║
║   MODEL: Qwen3:8B-q4_K_M (5.2GB - ALL LAYERS ON GPU)                                     ║
║   ═══════════════════════════════════════════════════════════════════════════════════    ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import httpx
import json
import time
import logging
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import deque
from functools import lru_cache
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BlazingFraudEngine')


# ═══════════════════════════════════════════════════════════════════════════════
#                           ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisMode(Enum):
    INSTANT = "instant"   # Pure heuristics (<5ms)
    TURBO = "turbo"       # Ultra-fast AI (<100ms)
    SMART = "smart"       # Quick AI (<300ms)
    DEEP = "deep"         # Thinking mode (<800ms)


class RiskLevel(Enum):
    SAFE = (0, 20, "SAFE", "✅")
    LOW = (21, 40, "LOW", "🟢")
    MEDIUM = (41, 60, "MEDIUM", "🟡")
    HIGH = (61, 80, "HIGH", "🟠")
    CRITICAL = (81, 100, "CRITICAL", "🔴")
    
    def __init__(self, min_s: int, max_s: int, label: str, emoji: str):
        self.min_score = min_s
        self.max_score = max_s
        self.label = label
        self.emoji = emoji
    
    @classmethod
    def from_score(cls, score: int) -> 'RiskLevel':
        score = max(0, min(100, score))
        for level in cls:
            if level.min_score <= score <= level.max_score:
                return level
        return cls.CRITICAL


class FraudType(Enum):
    """15 fraud typologies with quick detection flags."""
    RUG_PULL = ("rug_pull", 98, ["massive_outflow", "new_contract"])
    PIG_BUTCHERING = ("pig_butchering", 95, ["trust_pattern", "large_final"])
    MIXER_TUMBLING = ("mixer_tumbling", 92, ["known_mixer"])
    CHAIN_OBFUSCATION = ("chain_obfuscation", 85, ["bridge", "hop"])
    FAKE_TOKEN = ("fake_token", 88, ["honeypot"])
    FLASH_LOAN = ("flash_loan", 80, ["same_block"])
    WASH_TRADING = ("wash_trading", 75, ["self_deal"])
    STRUCTURING = ("structuring", 70, ["just_under"])
    VELOCITY_ATTACK = ("velocity_attack", 65, ["burst"])
    PEEL_CHAIN = ("peel_chain", 60, ["gradual_peel"])
    DUSTING = ("dusting", 55, ["tiny"])
    ADDRESS_POISONING = ("address_poisoning", 50, ["similar_addr"])
    APPROVAL_EXPLOIT = ("approval_exploit", 78, ["unlimited_approval"])
    SIM_SWAP = ("sim_swap", 90, ["recovery_drain"])
    ROMANCE_SCAM = ("romance_scam", 85, ["trust_transfers"])
    
    def __init__(self, code: str, base_score: int, flags: List[str]):
        self.code = code
        self.base_score = base_score
        self.flags = flags


# ═══════════════════════════════════════════════════════════════════════════════
#                           GPU-OPTIMIZED CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class BlazingConfig:
    """Maximum GPU performance configuration."""
    
    OLLAMA_URL = "http://localhost:11434"
    MODEL = "qwen3:8b"  # Uses quantized version
    KEEP_ALIVE = "60m"  # Keep warm for 60 minutes
    
    # Maximum GPU utilization
    GPU_OPTIONS = {
        "num_gpu": 99,           # All layers on GPU
        "main_gpu": 0,           # Use primary GPU
        "num_thread": 8,         # CPU threads for non-GPU ops
        "num_batch": 512,        # Batch size
        "f16_kv": True,          # FP16 for speed
        "low_vram": False,       # We have 8GB dedicated
    }
    
    # Mode-specific inference settings
    INSTANT_CONFIG = None  # No AI needed
    
    TURBO_CONFIG = {
        "num_ctx": 256,          # Minimal context
        "num_predict": 16,       # Just score + confidence
        "temperature": 0.0,      # Deterministic
        "stop": ["\n", ".", "}"],
    }
    
    SMART_CONFIG = {
        "num_ctx": 512,
        "num_predict": 48,
        "temperature": 0.1,
        "top_k": 5,
        "stop": ["\n\n", "---"],
    }
    
    DEEP_CONFIG = {
        "num_ctx": 1024,
        "num_predict": 128,
        "temperature": 0.3,
        "top_k": 20,
        "top_p": 0.85,
    }
    
    # Decision thresholds
    INSTANT_SAFE_MAX = 15       # Score <= 15: instant approve (no AI)
    INSTANT_DANGER_MIN = 90     # Score >= 90: instant block (no AI)
    TURBO_MAX = 40              # Score 16-40: turbo AI
    SMART_MAX = 70              # Score 41-70: smart AI
    # Score 71-89: deep AI
    
    BLOCK_THRESHOLD = 75
    FLAG_THRESHOLD = 50


# ═══════════════════════════════════════════════════════════════════════════════
#                           KNOWN ADDRESSES DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class AddressIntel:
    """Pre-computed address intelligence for instant decisions."""
    
    # Known malicious mixers (instant block)
    MIXERS: Set[str] = {
        "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # Tornado Cash
        "0x722122df12d4e14e13ac3b6895a86e84145b6967",
        "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc",
        "0xa160cdab225685da1d56aa342ad8841c3b53f291",
    }
    
    # Known safe exchanges (instant approve)
    SAFE_EXCHANGES: Set[str] = {
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance
        "0xd551234ae421e3bcba99a0da6d736074f22192ff",  # Binance Hot
        "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase
        "0xeb2629a2734e272bcc07bda959863f316f4bd4cf",  # Coinbase Commerce
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",  # Coinbase 10
        "0xe92d1a43df510f82c66382592a047d288f85226f",  # Kraken
    }
    
    # Blacklisted addresses (dynamic)
    BLACKLIST: Set[str] = set()
    
    # Whitelisted addresses (dynamic)
    WHITELIST: Set[str] = set()
    
    @classmethod
    def check_instant(cls, address: str) -> Optional[Tuple[str, int]]:
        """Check for instant decision (returns action, score)."""
        addr = address.lower()
        
        if addr in cls.MIXERS:
            return ("block", 95)
        if addr in cls.BLACKLIST:
            return ("block", 85)
        if addr in cls.SAFE_EXCHANGES:
            return ("approve", 5)
        if addr in cls.WHITELIST:
            return ("approve", 10)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                           ULTRA-FAST HEURISTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HeuristicsEngine:
    """Lightning-fast heuristic scoring."""
    
    # Amount thresholds
    SMALL_TX = 1_000
    MEDIUM_TX = 10_000
    LARGE_TX = 50_000
    WHALE_TX = 100_000
    
    # Structuring detection (just under $10k)
    STRUCTURING_MIN = 9_000
    STRUCTURING_MAX = 9_999
    
    # Suspicious patterns
    ROUND_AMOUNT_THRESHOLD = 5_000  # Round amounts > $5k are suspicious
    
    def __init__(self):
        self.tx_history: Dict[str, deque] = {}  # sender -> recent tx times
        self.velocity_window = 300  # 5 minutes
        
    def compute_score(self, tx: Dict) -> Tuple[int, List[str], List[FraudType]]:
        """
        Ultra-fast heuristic scoring.
        Returns: (score, reasons, detected_types)
        """
        score = 0
        reasons = []
        detected_types = []
        
        amount = tx.get("amount", 0)
        sender = tx.get("sender", "").lower()
        recipient = tx.get("recipient", "").lower()
        
        # === INSTANT KNOWN ADDRESS CHECK ===
        recipient_check = AddressIntel.check_instant(recipient)
        if recipient_check:
            action, addr_score = recipient_check
            if action == "block":
                return (addr_score, ["Known malicious address"], [FraudType.MIXER_TUMBLING])
            elif action == "approve":
                return (addr_score, ["Known safe address"], [])
        
        # === AMOUNT-BASED SCORING ===
        if amount >= self.WHALE_TX:
            score += 25
            reasons.append(f"Whale transaction: ${amount:,.0f}")
        elif amount >= self.LARGE_TX:
            score += 15
            reasons.append(f"Large transaction: ${amount:,.0f}")
        elif amount >= self.MEDIUM_TX:
            score += 8
            reasons.append(f"Medium transaction: ${amount:,.0f}")
        elif amount < self.SMALL_TX:
            if amount < 1:
                score += 10
                reasons.append("Dust amount")
                detected_types.append(FraudType.DUSTING)
            # Small transactions are low risk
            score += 2
        
        # === STRUCTURING DETECTION ===
        if self.STRUCTURING_MIN <= amount <= self.STRUCTURING_MAX:
            score += 35
            reasons.append(f"Structuring pattern: ${amount:,.2f} (just under $10k)")
            detected_types.append(FraudType.STRUCTURING)
        
        # === ROUND AMOUNT CHECK ===
        if amount >= self.ROUND_AMOUNT_THRESHOLD:
            if amount == int(amount) and amount % 1000 == 0:
                score += 12
                reasons.append(f"Suspicious round amount: ${amount:,.0f}")
        
        # === VELOCITY CHECK ===
        now = time.time()
        if sender not in self.tx_history:
            self.tx_history[sender] = deque(maxlen=20)
        
        # Clean old transactions
        while self.tx_history[sender] and (now - self.tx_history[sender][0]) > self.velocity_window:
            self.tx_history[sender].popleft()
        
        recent_count = len(self.tx_history[sender])
        self.tx_history[sender].append(now)
        
        if recent_count >= 10:
            score += 40
            reasons.append(f"Velocity attack: {recent_count} tx in 5min")
            detected_types.append(FraudType.VELOCITY_ATTACK)
        elif recent_count >= 5:
            score += 20
            reasons.append(f"High frequency: {recent_count} tx in 5min")
        elif recent_count >= 3:
            score += 8
            reasons.append(f"Moderate frequency: {recent_count} tx in 5min")
        
        # === NEW ADDRESS CHECK ===
        tx_count = tx.get("sender_tx_count", 100)
        if tx_count < 5:
            score += 15
            reasons.append(f"New address: only {tx_count} prior transactions")
        elif tx_count < 20:
            score += 5
            reasons.append(f"Young address: {tx_count} transactions")
        
        # === CONTRACT INTERACTION ===
        if tx.get("is_contract", False):
            score += 10
            reasons.append("Contract interaction")
        
        # === TIME-BASED PATTERNS ===
        hour = time.localtime().tm_hour
        if 2 <= hour <= 5:  # 2AM - 5AM local
            score += 5
            reasons.append("Off-hours transaction")
        
        return (min(score, 100), reasons, detected_types)


# ═══════════════════════════════════════════════════════════════════════════════
#                           AI INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AIInferenceEngine:
    """Optimized AI inference for fraud detection."""
    
    def __init__(self, config: BlazingConfig):
        self.config = config
        self.client = httpx.Client(timeout=30.0)
        self.model_ready = False
        
    async def warm_up(self):
        """Pre-load model to GPU."""
        logger.info("🔥 Loading Qwen3:8B to GPU (8GB VRAM)...")
        start = time.perf_counter()
        
        try:
            response = self.client.post(
                f"{self.config.OLLAMA_URL}/api/generate",
                json={
                    "model": self.config.MODEL,
                    "prompt": "Ready",
                    "stream": False,
                    "keep_alive": self.config.KEEP_ALIVE,
                    "options": {
                        **self.config.GPU_OPTIONS,
                        "num_ctx": 256,
                        "num_predict": 1,
                    }
                }
            )
            response.raise_for_status()
            self.model_ready = True
            logger.info(f"   ✅ Model ready in {(time.perf_counter() - start) * 1000:.0f}ms")
        except Exception as e:
            logger.error(f"   ❌ Model load failed: {e}")
            
    def _build_prompt(self, tx: Dict, mode: AnalysisMode, heuristic_score: int) -> str:
        """Build optimized prompts for each mode."""
        
        amount = tx.get("amount", 0)
        sender = tx.get("sender", "")[:10]
        recipient = tx.get("recipient", "")[:10]
        
        if mode == AnalysisMode.TURBO:
            # Minimal prompt for speed
            return f"""Fraud score (0-100) for: ${amount:,.0f} from {sender}... to {recipient}...
Heuristic: {heuristic_score}. Output JSON: {{"score":N,"conf":0.N}}"""

        elif mode == AnalysisMode.SMART:
            return f"""Analyze blockchain transaction for fraud.
Amount: ${amount:,.0f}
Sender: {sender}...
Recipient: {recipient}...
Initial score: {heuristic_score}

Output JSON only: {{"score":N,"confidence":0.N,"type":"fraud_type_or_none"}}"""

        else:  # DEEP
            return f"""/think
Deep fraud analysis required for high-risk transaction.

Transaction Details:
- Amount: ${amount:,.0f}
- Sender: {sender}...
- Recipient: {recipient}...
- Heuristic Score: {heuristic_score}

Analyze for: rug_pull, pig_butchering, mixer, structuring, velocity_attack, wash_trading

Output JSON: {{"score":N,"confidence":0.N,"type":"string","reason":"brief explanation"}}"""

    async def infer(self, tx: Dict, mode: AnalysisMode, heuristic_score: int) -> Dict:
        """Run optimized inference."""
        
        if mode == AnalysisMode.INSTANT:
            return {"score": heuristic_score, "confidence": 0.9}
        
        prompt = self._build_prompt(tx, mode, heuristic_score)
        
        # Select config based on mode
        if mode == AnalysisMode.TURBO:
            options = {**self.config.GPU_OPTIONS, **self.config.TURBO_CONFIG}
        elif mode == AnalysisMode.SMART:
            options = {**self.config.GPU_OPTIONS, **self.config.SMART_CONFIG}
        else:
            options = {**self.config.GPU_OPTIONS, **self.config.DEEP_CONFIG}
        
        start = time.perf_counter()
        
        try:
            response = self.client.post(
                f"{self.config.OLLAMA_URL}/api/generate",
                json={
                    "model": self.config.MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.config.KEEP_ALIVE,
                    "options": options
                }
            )
            response.raise_for_status()
            
            latency = (time.perf_counter() - start) * 1000
            result = response.json()
            text = result.get("response", "")
            
            # Parse JSON from response
            parsed = self._parse_response(text, heuristic_score)
            parsed["latency_ms"] = latency
            parsed["mode"] = mode.value
            
            return parsed
            
        except Exception as e:
            logger.warning(f"AI inference failed: {e}")
            return {
                "score": heuristic_score,
                "confidence": 0.5,
                "error": str(e),
                "mode": mode.value
            }
    
    def _parse_response(self, text: str, fallback_score: int) -> Dict:
        """Parse AI response, extracting JSON."""
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "score": int(data.get("score", fallback_score)),
                    "confidence": float(data.get("confidence", data.get("conf", 0.7))),
                    "type": data.get("type", ""),
                    "reason": data.get("reason", ""),
                }
            except:
                pass
        
        # Try to extract just a number
        numbers = re.findall(r'\b(\d{1,2}|100)\b', text)
        if numbers:
            return {
                "score": int(numbers[0]),
                "confidence": 0.6,
            }
        
        return {
            "score": fallback_score,
            "confidence": 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           BLAZING FRAUD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FraudResult:
    """Fraud analysis result."""
    tx_id: str
    score: int
    risk: RiskLevel
    mode: AnalysisMode
    approved: bool
    blocked: bool
    flagged: bool
    reasons: List[str]
    detected_types: List[str]
    confidence: float
    latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "score": self.score,
            "risk_level": self.risk.label,
            "risk_emoji": self.risk.emoji,
            "mode": self.mode.value,
            "approved": self.approved,
            "blocked": self.blocked,
            "flagged": self.flagged,
            "reasons": self.reasons,
            "detected_types": self.detected_types,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 2),
        }


class BlazingFraudEngine:
    """
    Ultimate fraud detection engine with maximum GPU acceleration.
    
    Architecture:
    1. INSTANT (<5ms): Known addresses + basic heuristics
    2. TURBO (<100ms): Ultra-fast AI for ambiguous low-risk
    3. SMART (<300ms): Quick AI for medium risk
    4. DEEP (<800ms): Full thinking mode for high risk
    """
    
    VERSION = "4.0.0-blazing"
    
    def __init__(self):
        self.config = BlazingConfig()
        self.heuristics = HeuristicsEngine()
        self.ai_engine = AIInferenceEngine(self.config)
        
        # Stats
        self.stats = {
            "total": 0,
            "by_mode": {m.value: 0 for m in AnalysisMode},
            "blocked": 0,
            "flagged": 0,
            "approved": 0,
            "latencies": [],
        }
        
        logger.info(f"🔥 Blazing Fraud Engine {self.VERSION} initialized")
        
    async def initialize(self):
        """Initialize and warm up GPU."""
        await self.ai_engine.warm_up()
        
    def _select_mode(self, heuristic_score: int) -> AnalysisMode:
        """Select analysis mode based on heuristic score."""
        
        if heuristic_score <= self.config.INSTANT_SAFE_MAX:
            return AnalysisMode.INSTANT
        elif heuristic_score >= self.config.INSTANT_DANGER_MIN:
            return AnalysisMode.INSTANT
        elif heuristic_score <= self.config.TURBO_MAX:
            return AnalysisMode.TURBO
        elif heuristic_score <= self.config.SMART_MAX:
            return AnalysisMode.SMART
        else:
            return AnalysisMode.DEEP
    
    async def analyze(self, tx: Dict) -> FraudResult:
        """
        Analyze transaction for fraud.
        
        Args:
            tx: Transaction dict with keys:
                - tx_id: str
                - sender: str
                - recipient: str
                - amount: float
                - sender_tx_count: int (optional)
                - is_contract: bool (optional)
        
        Returns:
            FraudResult with score, risk level, and decisions
        """
        start = time.perf_counter()
        
        # Step 1: Heuristic scoring (< 1ms)
        heuristic_score, reasons, detected_types = self.heuristics.compute_score(tx)
        
        # Step 2: Select analysis mode
        mode = self._select_mode(heuristic_score)
        
        # Step 3: AI inference (if needed)
        final_score = heuristic_score
        confidence = 0.9
        
        if mode != AnalysisMode.INSTANT:
            ai_result = await self.ai_engine.infer(tx, mode, heuristic_score)
            
            # Weighted combination of heuristic and AI scores
            if mode == AnalysisMode.TURBO:
                # 70% heuristic, 30% AI for turbo
                final_score = int(0.7 * heuristic_score + 0.3 * ai_result.get("score", heuristic_score))
            elif mode == AnalysisMode.SMART:
                # 50/50 for smart
                final_score = int(0.5 * heuristic_score + 0.5 * ai_result.get("score", heuristic_score))
            else:
                # 30% heuristic, 70% AI for deep
                final_score = int(0.3 * heuristic_score + 0.7 * ai_result.get("score", heuristic_score))
            
            confidence = ai_result.get("confidence", 0.7)
            
            if ai_result.get("reason"):
                reasons.append(f"AI: {ai_result['reason']}")
            if ai_result.get("type"):
                detected_types.append(ai_result["type"])
        
        # Step 4: Make decisions
        risk = RiskLevel.from_score(final_score)
        blocked = final_score >= self.config.BLOCK_THRESHOLD
        flagged = final_score >= self.config.FLAG_THRESHOLD
        approved = not blocked
        
        latency = (time.perf_counter() - start) * 1000
        
        # Update stats
        self.stats["total"] += 1
        self.stats["by_mode"][mode.value] += 1
        self.stats["latencies"].append(latency)
        if blocked:
            self.stats["blocked"] += 1
        if flagged:
            self.stats["flagged"] += 1
        if approved:
            self.stats["approved"] += 1
        
        return FraudResult(
            tx_id=tx.get("tx_id", "unknown"),
            score=final_score,
            risk=risk,
            mode=mode,
            approved=approved,
            blocked=blocked,
            flagged=flagged,
            reasons=reasons,
            detected_types=[t.code if isinstance(t, FraudType) else str(t) for t in detected_types],
            confidence=confidence,
            latency_ms=latency,
        )
    
    async def analyze_batch(self, transactions: List[Dict]) -> List[FraudResult]:
        """Analyze multiple transactions."""
        return [await self.analyze(tx) for tx in transactions]
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        latencies = self.stats["latencies"]
        if not latencies:
            return self.stats
        
        return {
            **self.stats,
            "latency_avg_ms": sum(latencies) / len(latencies),
            "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
            "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
            "mode_distribution": {
                mode: f"{count}/{self.stats['total']} ({100*count/max(1,self.stats['total']):.1f}%)"
                for mode, count in self.stats["by_mode"].items()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_tests():
    """Run production tests."""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   🔥 PAYFLOW BLAZING FRAUD ENGINE - v4.0 PRODUCTION TEST                                 ║
║                                                                                           ║
║   ⚡ INSTANT: <5ms  |  🚀 TURBO: <100ms  |  🧠 SMART: <300ms  |  🔬 DEEP: <800ms        ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    engine = BlazingFraudEngine()
    await engine.initialize()
    
    # Test transactions covering all modes
    test_transactions = [
        # INSTANT mode tests (known safe/dangerous)
        {
            "tx_id": "instant_safe_01",
            "sender": "0xuser123",
            "recipient": "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance
            "amount": 5000,
        },
        {
            "tx_id": "instant_danger_01",
            "sender": "0xbad_actor",
            "recipient": "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # Tornado Cash
            "amount": 50000,
        },
        # INSTANT mode (low risk heuristics)
        {
            "tx_id": "instant_low_01",
            "sender": "0xalice",
            "recipient": "0xbob",
            "amount": 150,
            "sender_tx_count": 500,
        },
        # TURBO mode (medium-low risk)
        {
            "tx_id": "turbo_01",
            "sender": "0xcharlie",
            "recipient": "0xdave",
            "amount": 8000,
            "sender_tx_count": 50,
        },
        # SMART mode (medium risk)
        {
            "tx_id": "smart_01",
            "sender": "0xsuspect",
            "recipient": "0xunknown",
            "amount": 25000,
            "sender_tx_count": 10,
        },
        # DEEP mode (high risk - structuring)
        {
            "tx_id": "deep_01",
            "sender": "0xlaunderer",
            "recipient": "0xshell",
            "amount": 9950,
            "sender_tx_count": 3,
        },
        # DEEP mode (whale transaction)
        {
            "tx_id": "deep_02",
            "sender": "0xwhale",
            "recipient": "0xnew_addr",
            "amount": 250000,
            "sender_tx_count": 8,
        },
        # Edge case: velocity burst
        {
            "tx_id": "velocity_01",
            "sender": "0xspammer",
            "recipient": "0xtarget",
            "amount": 100,
            "sender_tx_count": 100,
        },
    ]
    
    print("🧪 Running test transactions...\n")
    print("=" * 100)
    print(f"{'TX ID':<20} {'Amount':>12} {'Mode':<8} {'Score':>5} {'Risk':<10} {'Decision':<12} {'Latency':>10}")
    print("=" * 100)
    
    for tx in test_transactions:
        result = await engine.analyze(tx)
        
        mode_emoji = {
            AnalysisMode.INSTANT: "⚡",
            AnalysisMode.TURBO: "🚀",
            AnalysisMode.SMART: "🧠",
            AnalysisMode.DEEP: "🔬",
        }
        
        decision = "🔴BLOCKED" if result.blocked else ("🟡FLAGGED" if result.flagged else "🟢APPROVED")
        
        print(f"{result.tx_id:<20} ${tx['amount']:>10,.0f} {mode_emoji[result.mode]}{result.mode.value:<6} {result.score:>5} {result.risk.emoji}{result.risk.label:<8} {decision:<12} {result.latency_ms:>8.0f}ms")
        
        # Show reasons for high-risk
        if result.score >= 50:
            for reason in result.reasons[:2]:
                print(f"   └─ {reason}")
    
    print("=" * 100)
    
    # Stats
    stats = engine.get_stats()
    print(f"""
📊 ENGINE STATISTICS:
   Version: {engine.VERSION}
   Total Transactions: {stats['total']}
   
   Mode Distribution:
   • INSTANT (heuristic): {stats['mode_distribution']['instant']}
   • TURBO (ultra-fast):  {stats['mode_distribution']['turbo']}
   • SMART (quick AI):    {stats['mode_distribution']['smart']}
   • DEEP (thinking):     {stats['mode_distribution']['deep']}

   Latency:
   • Average: {stats.get('latency_avg_ms', 0):.1f}ms
   • P50:     {stats.get('latency_p50_ms', 0):.1f}ms
   • P95:     {stats.get('latency_p95_ms', 0):.1f}ms

   Decisions:
   • Approved: {stats['approved']}
   • Flagged:  {stats['flagged']}
   • Blocked:  {stats['blocked']}
""")
    
    # Performance targets
    print("\n🎯 PERFORMANCE TARGETS:")
    avg_latency = stats.get('latency_avg_ms', 0)
    instant_pct = stats['by_mode']['instant'] / max(1, stats['total']) * 100
    
    targets = [
        ("Average Latency < 200ms", avg_latency < 200, f"{avg_latency:.1f}ms"),
        ("INSTANT mode > 50%", instant_pct > 50, f"{instant_pct:.1f}%"),
        ("All transactions processed", stats['total'] == len(test_transactions), f"{stats['total']}/{len(test_transactions)}"),
    ]
    
    for target, passed, value in targets:
        status = "✅" if passed else "❌"
        print(f"   {status} {target}: {value}")
    
    print("\n✅ Production test complete!")


if __name__ == "__main__":
    asyncio.run(run_tests())
