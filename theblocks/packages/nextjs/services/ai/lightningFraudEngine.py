"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PAYFLOW ULTIMATE FRAUD ENGINE - v5.0 LIGHTNING                         ║
║                                                                                           ║
║   ⚡ LIGHTNING FAST FRAUD DETECTION WITH GPU ACCELERATION                                ║
║                                                                                           ║
║   MODES:                                                                                  ║
║   ⚡ INSTANT: <2ms  (65% - pure heuristics + known addresses)                            ║
║   🚀 QUICK:   <80ms (30% - minimal AI with 8 tokens max)                                 ║
║   🧠 VERIFY:  <200ms (5% - short AI verification)                                        ║
║                                                                                           ║
║   SECRET: The AI only outputs a SINGLE NUMBER (0-100)!                                   ║
║                                                                                           ║
║   GPU: RTX 4070 (8GB VRAM) - 100% DEDICATED                                              ║
║   MODEL: Qwen3:8B - ALL LAYERS ON GPU                                                    ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import httpx
import time
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LightningFraudEngine')


# ═══════════════════════════════════════════════════════════════════════════════
#                           ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Mode(Enum):
    INSTANT = "instant"
    QUICK = "quick"
    VERIFY = "verify"


class Risk(Enum):
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
    def from_score(cls, s: int) -> 'Risk':
        s = max(0, min(100, s))
        for r in cls:
            if r.min_score <= s <= r.max_score:
                return r
        return cls.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
#                           KNOWN ADDRESSES (INSTANT DECISIONS)
# ═══════════════════════════════════════════════════════════════════════════════

MIXERS = {
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
    "0x722122df12d4e14e13ac3b6895a86e84145b6967",
    "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc",
}

SAFE_EXCHANGES = {
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance
    "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase
    "0xe92d1a43df510f82c66382592a047d288f85226f",  # Kraken
}

BLACKLIST: Set[str] = set()
WHITELIST: Set[str] = set()


# ═══════════════════════════════════════════════════════════════════════════════
#                           HEURISTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class Heuristics:
    """Ultra-fast heuristic scoring."""
    
    def __init__(self):
        self.tx_history: Dict[str, deque] = {}
        
    def score(self, tx: Dict) -> Tuple[int, List[str]]:
        """Compute heuristic score in <1ms."""
        s = 0
        reasons = []
        
        amount = tx.get("amount", 0)
        recipient = tx.get("recipient", "").lower()
        sender = tx.get("sender", "").lower()
        
        # Known addresses (instant)
        if recipient in MIXERS:
            return (95, ["Known mixer address"])
        if recipient in BLACKLIST:
            return (85, ["Blacklisted address"])
        if recipient in SAFE_EXCHANGES:
            return (5, ["Known safe exchange"])
        if recipient in WHITELIST:
            return (10, ["Whitelisted address"])
        
        # Amount scoring
        if amount >= 100_000:
            s += 25
            reasons.append(f"Whale tx: ${amount:,.0f}")
        elif amount >= 50_000:
            s += 15
            reasons.append(f"Large tx: ${amount:,.0f}")
        elif amount >= 10_000:
            s += 8
        elif amount < 1:
            s += 10
            reasons.append("Dust amount")
        
        # Structuring (just under $10k)
        if 9_000 <= amount <= 9_999:
            s += 35
            reasons.append("Structuring pattern")
        
        # Round amounts
        if amount >= 5_000 and amount == int(amount) and amount % 1000 == 0:
            s += 10
            reasons.append("Suspicious round amount")
        
        # Velocity
        now = time.time()
        if sender not in self.tx_history:
            self.tx_history[sender] = deque(maxlen=20)
        
        # Clean old
        while self.tx_history[sender] and (now - self.tx_history[sender][0]) > 300:
            self.tx_history[sender].popleft()
        
        count = len(self.tx_history[sender])
        self.tx_history[sender].append(now)
        
        if count >= 10:
            s += 35
            reasons.append("Velocity attack detected")
        elif count >= 5:
            s += 15
            reasons.append("High frequency")
        
        # New address
        tx_count = tx.get("sender_tx_count", 100)
        if tx_count < 5:
            s += 12
            reasons.append("New address")
        elif tx_count < 20:
            s += 5
        
        return (min(s, 100), reasons)


# ═══════════════════════════════════════════════════════════════════════════════
#                           LIGHTNING AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LightningAI:
    """
    Ultra-optimized AI that outputs ONLY a single number.
    Key insight: We don't need explanations from AI - just the score!
    """
    
    OLLAMA_URL = "http://localhost:11434"
    MODEL = "qwen3:8b"
    
    # Maximum GPU settings
    GPU_OPTS = {
        "num_gpu": 99,
        "num_thread": 8,
        "num_batch": 512,
        "f16_kv": True,
    }
    
    def __init__(self):
        self.client = httpx.Client(timeout=10.0)
        self.ready = False
        
    async def warmup(self):
        """Load model to GPU."""
        logger.info("🔥 Loading Qwen3:8B to GPU...")
        start = time.perf_counter()
        
        try:
            self.client.post(
                f"{self.OLLAMA_URL}/api/generate",
                json={
                    "model": self.MODEL,
                    "prompt": "1",
                    "stream": False,
                    "keep_alive": "60m",
                    "options": {**self.GPU_OPTS, "num_ctx": 128, "num_predict": 1}
                }
            )
            self.ready = True
            logger.info(f"   ✅ Model ready in {(time.perf_counter() - start) * 1000:.0f}ms")
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
    
    async def get_score(self, tx: Dict, heuristic_score: int, mode: Mode) -> int:
        """
        Get AI fraud score. Returns ONLY a number!
        
        The secret sauce: Ask AI to output just one number.
        No JSON, no explanation = ultra-fast inference.
        """
        
        amount = tx.get("amount", 0)
        
        if mode == Mode.QUICK:
            # Minimal prompt, minimal tokens
            prompt = f"/no_think\nFraud score 0-100 for ${amount:,.0f} transfer. Prior:{heuristic_score}. Just the number:"
            max_tokens = 4
            ctx = 128
        else:  # VERIFY
            prompt = f"/no_think\nFraud risk score (0-100) for blockchain transaction:\nAmount: ${amount:,.0f}\nHeuristic: {heuristic_score}\nOutput only the number:"
            max_tokens = 6
            ctx = 256
        
        try:
            start = time.perf_counter()
            
            resp = self.client.post(
                f"{self.OLLAMA_URL}/api/generate",
                json={
                    "model": self.MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "60m",
                    "options": {
                        **self.GPU_OPTS,
                        "num_ctx": ctx,
                        "num_predict": max_tokens,
                        "temperature": 0.0,
                        "stop": ["\n", " ", ".", ","],
                    }
                }
            )
            
            latency = (time.perf_counter() - start) * 1000
            text = resp.json().get("response", "")
            
            # Extract just the number
            nums = re.findall(r'\d+', text)
            if nums:
                ai_score = min(100, max(0, int(nums[0])))
                # Weighted average with heuristics
                if mode == Mode.QUICK:
                    return int(0.6 * heuristic_score + 0.4 * ai_score)
                else:
                    return int(0.4 * heuristic_score + 0.6 * ai_score)
            
            return heuristic_score
            
        except Exception as e:
            logger.warning(f"AI error: {e}")
            return heuristic_score


# ═══════════════════════════════════════════════════════════════════════════════
#                           LIGHTNING FRAUD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    """Fraud analysis result."""
    tx_id: str
    score: int
    risk: Risk
    mode: Mode
    approved: bool
    blocked: bool
    flagged: bool
    reasons: List[str]
    latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "score": self.score,
            "risk": self.risk.label,
            "emoji": self.risk.emoji,
            "mode": self.mode.value,
            "approved": self.approved,
            "blocked": self.blocked,
            "flagged": self.flagged,
            "reasons": self.reasons,
            "latency_ms": round(self.latency_ms, 2),
        }


class LightningFraudEngine:
    """
    Production fraud detection with lightning-fast response.
    
    Key optimization: AI outputs only a single number!
    """
    
    VERSION = "5.0.0-lightning"
    
    # Thresholds
    INSTANT_SAFE = 12      # <= 12: instant approve
    INSTANT_DANGER = 85    # >= 85: instant block
    QUICK_MAX = 45         # 13-45: quick AI
    # > 45: verify AI
    
    BLOCK_AT = 75
    FLAG_AT = 50
    
    def __init__(self):
        self.heuristics = Heuristics()
        self.ai = LightningAI()
        self.stats = {"total": 0, "instant": 0, "quick": 0, "verify": 0, "latencies": []}
        logger.info(f"⚡ Lightning Fraud Engine {self.VERSION} initialized")
        
    async def init(self):
        await self.ai.warmup()
        
    def _select_mode(self, score: int) -> Mode:
        if score <= self.INSTANT_SAFE or score >= self.INSTANT_DANGER:
            return Mode.INSTANT
        elif score <= self.QUICK_MAX:
            return Mode.QUICK
        else:
            return Mode.VERIFY
    
    async def analyze(self, tx: Dict) -> Result:
        """Analyze transaction for fraud."""
        start = time.perf_counter()
        
        # Step 1: Heuristics (<1ms)
        h_score, reasons = self.heuristics.score(tx)
        
        # Step 2: Select mode
        mode = self._select_mode(h_score)
        
        # Step 3: AI if needed
        final_score = h_score
        if mode != Mode.INSTANT:
            final_score = await self.ai.get_score(tx, h_score, mode)
        
        # Step 4: Decisions
        risk = Risk.from_score(final_score)
        blocked = final_score >= self.BLOCK_AT
        flagged = final_score >= self.FLAG_AT
        
        latency = (time.perf_counter() - start) * 1000
        
        # Stats
        self.stats["total"] += 1
        self.stats[mode.value] += 1
        self.stats["latencies"].append(latency)
        
        return Result(
            tx_id=tx.get("tx_id", "unknown"),
            score=final_score,
            risk=risk,
            mode=mode,
            approved=not blocked,
            blocked=blocked,
            flagged=flagged,
            reasons=reasons,
            latency_ms=latency,
        )
    
    def get_stats(self) -> Dict:
        lats = self.stats["latencies"]
        if not lats:
            return self.stats
        
        return {
            **self.stats,
            "avg_ms": sum(lats) / len(lats),
            "p50_ms": sorted(lats)[len(lats) // 2],
            "p95_ms": sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 20 else max(lats),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_test():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   ⚡ PAYFLOW LIGHTNING FRAUD ENGINE - v5.0 PRODUCTION TEST                               ║
║                                                                                           ║
║   ⚡ INSTANT: <2ms  |  🚀 QUICK: <80ms  |  🧠 VERIFY: <200ms                             ║
║                                                                                           ║
║   🔑 SECRET: AI outputs only a SINGLE NUMBER for max speed!                              ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    engine = LightningFraudEngine()
    await engine.init()
    
    # Test transactions
    tests = [
        # INSTANT - known safe
        {"tx_id": "safe_exchange", "sender": "0xuser", "recipient": "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be", "amount": 10000},
        # INSTANT - known mixer
        {"tx_id": "mixer_block", "sender": "0xbad", "recipient": "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b", "amount": 50000},
        # INSTANT - low amount safe
        {"tx_id": "small_safe", "sender": "0xalice", "recipient": "0xbob", "amount": 100, "sender_tx_count": 500},
        {"tx_id": "small_safe2", "sender": "0xcharlie", "recipient": "0xdave", "amount": 250, "sender_tx_count": 200},
        # QUICK - medium risk
        {"tx_id": "medium_1", "sender": "0xeve", "recipient": "0xfrank", "amount": 15000, "sender_tx_count": 30},
        {"tx_id": "medium_2", "sender": "0xgrace", "recipient": "0xhenry", "amount": 8000, "sender_tx_count": 50},
        # VERIFY - higher risk
        {"tx_id": "high_1", "sender": "0xivan", "recipient": "0xjulia", "amount": 35000, "sender_tx_count": 10},
        {"tx_id": "structuring", "sender": "0xkate", "recipient": "0xluke", "amount": 9950, "sender_tx_count": 5},
        # Edge cases
        {"tx_id": "whale", "sender": "0xwhale", "recipient": "0xnew", "amount": 150000, "sender_tx_count": 100},
        {"tx_id": "dust", "sender": "0xdust", "recipient": "0xvictim", "amount": 0.001, "sender_tx_count": 1},
    ]
    
    print("🧪 Running tests...\n")
    print("=" * 95)
    print(f"{'TX ID':<16} {'Amount':>12} {'Mode':<8} {'Score':>5} {'Risk':<10} {'Decision':<12} {'Latency':>8}")
    print("=" * 95)
    
    for tx in tests:
        r = await engine.analyze(tx)
        
        mode_emoji = {"instant": "⚡", "quick": "🚀", "verify": "🧠"}
        decision = "🔴BLOCKED" if r.blocked else ("🟡FLAG" if r.flagged else "🟢OK")
        
        print(f"{r.tx_id:<16} ${tx['amount']:>10,.2f} {mode_emoji[r.mode.value]}{r.mode.value:<5} {r.score:>5} {r.risk.emoji}{r.risk.label:<8} {decision:<12} {r.latency_ms:>6.0f}ms")
    
    print("=" * 95)
    
    stats = engine.get_stats()
    total = stats["total"]
    
    print(f"""
📊 STATISTICS:
   Total: {total}
   
   Mode Distribution:
   • INSTANT: {stats['instant']}/{total} ({100*stats['instant']/total:.0f}%)
   • QUICK:   {stats['quick']}/{total} ({100*stats['quick']/total:.0f}%)
   • VERIFY:  {stats['verify']}/{total} ({100*stats['verify']/total:.0f}%)

   Latency:
   • Average: {stats['avg_ms']:.1f}ms
   • P50:     {stats['p50_ms']:.1f}ms
   • P95:     {stats['p95_ms']:.1f}ms
""")
    
    # Targets
    print("🎯 TARGETS:")
    avg = stats['avg_ms']
    instant_pct = 100 * stats['instant'] / total
    
    checks = [
        ("Average < 100ms", avg < 100, f"{avg:.1f}ms"),
        ("INSTANT > 40%", instant_pct > 40, f"{instant_pct:.0f}%"),
        ("P95 < 300ms", stats['p95_ms'] < 300, f"{stats['p95_ms']:.0f}ms"),
    ]
    
    for name, ok, val in checks:
        print(f"   {'✅' if ok else '❌'} {name}: {val}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(run_test())
