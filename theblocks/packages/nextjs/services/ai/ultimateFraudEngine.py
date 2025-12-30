"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW ULTIMATE FRAUD DETECTION ENGINE                               ║
║                                                                                           ║
║   🚀 FULL THROTTLE GPU ACCELERATION FOR RTX 4070 (8GB VRAM)                              ║
║   🧠 QWEN3:8B WITH THINKING MODE FOR MAXIMUM ACCURACY                                     ║
║   ⚡ QUANTIZED MODEL FOR OPTIMAL SPEED/QUALITY BALANCE                                    ║
║                                                                                           ║
║   ═══════════════════════════════════════════════════════════════════════════════════    ║
║   PERFORMANCE TARGETS:                                                                    ║
║   • Response Time: <300ms (target) | <500ms (max)                                        ║
║   • Accuracy: 95%+ detection rate                                                         ║
║   • False Positive Rate: <5%                                                              ║
║   • Throughput: 100+ tx/minute                                                            ║
║   ═══════════════════════════════════════════════════════════════════════════════════    ║
║                                                                                           ║
║   15 FRAUD TYPOLOGIES DETECTED:                                                           ║
║   1. Rug Pulls           6. Flash Loans       11. Dusting                                ║
║   2. Pig Butchering      7. Wash Trading      12. Address Poisoning                      ║
║   3. Mixer/Tumbling      8. Structuring       13. Approval Exploits                      ║
║   4. Chain Obfuscation   9. Velocity Attack   14. SIM Swap                               ║
║   5. Fake Tokens        10. Peel Chains       15. Romance Scams                          ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import httpx
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UltimateFraudEngine')


# ═══════════════════════════════════════════════════════════════════════════════
#                     GPU CONFIGURATION - FULL THROTTLE RTX 4070
# ═══════════════════════════════════════════════════════════════════════════════

class GPUConfig:
    """
    🚀 MAXIMUM PERFORMANCE GPU SETTINGS FOR RTX 4070 8GB
    
    Now that chatbot is moved to Gemini Cloud, the full 8GB VRAM
    is dedicated to fraud detection for ultimate performance.
    """
    
    # Ollama endpoint
    OLLAMA_URL = "http://localhost:11434"
    MODEL = "qwen3:8b"
    
    # === FULL THROTTLE GPU ACCELERATION ===
    GPU_LAYERS = 99  # ALL layers on GPU (no CPU offload)
    
    # === VRAM OPTIMIZATION (8GB DEDICATED) ===
    NUM_CTX = 4096  # Full context window (increased from 2048)
    NUM_BATCH = 512  # Maximum batch size for prompt processing
    
    # === PARALLEL PROCESSING ===
    NUM_THREAD = 8  # CPU threads for non-GPU operations
    
    # === MEMORY MANAGEMENT ===
    USE_MMAP = True  # Memory-mapped model loading
    USE_MLOCK = False  # Don't lock in RAM (use VRAM)
    
    # === INFERENCE SPEED (TUNED FOR ACCURACY) ===
    TEMPERATURE_THINKING = 0.6  # Balanced for reasoning quality
    TEMPERATURE_FAST = 0.3  # Deterministic for quick decisions
    TOP_K = 40
    TOP_P = 0.9
    REPEAT_PENALTY = 1.15  # Prevent repetitive reasoning loops
    
    # === THINKING MODE SETTINGS ===
    THINKING_BUDGET_TOKENS = 1024  # Max thinking tokens
    OUTPUT_TOKENS = 256  # Focused output
    
    # === MIROSTAT (DISABLED FOR SPEED) ===
    MIROSTAT = 0
    
    # === KEEP ALIVE (30 MINUTES) ===
    KEEP_ALIVE = "30m"
    
    @classmethod
    def get_thinking_options(cls) -> Dict[str, Any]:
        """Get GPU options optimized for thinking mode (accuracy focus)."""
        return {
            "num_gpu": cls.GPU_LAYERS,
            "num_ctx": cls.NUM_CTX,
            "num_batch": cls.NUM_BATCH,
            "num_thread": cls.NUM_THREAD,
            "use_mmap": cls.USE_MMAP,
            "use_mlock": cls.USE_MLOCK,
            "temperature": cls.TEMPERATURE_THINKING,
            "top_k": cls.TOP_K,
            "top_p": cls.TOP_P,
            "repeat_penalty": cls.REPEAT_PENALTY,
            "mirostat": cls.MIROSTAT,
            "num_predict": cls.THINKING_BUDGET_TOKENS + cls.OUTPUT_TOKENS,
        }
    
    @classmethod
    def get_fast_options(cls) -> Dict[str, Any]:
        """Get GPU options optimized for speed (low-risk transactions)."""
        return {
            "num_gpu": cls.GPU_LAYERS,
            "num_ctx": 2048,  # Smaller context for speed
            "num_batch": cls.NUM_BATCH,
            "num_thread": cls.NUM_THREAD,
            "use_mmap": cls.USE_MMAP,
            "use_mlock": cls.USE_MLOCK,
            "temperature": cls.TEMPERATURE_FAST,
            "top_k": 20,  # Tighter sampling
            "top_p": 0.8,
            "repeat_penalty": 1.1,
            "mirostat": cls.MIROSTAT,
            "num_predict": 128,  # Short response
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           FRAUD TYPOLOGY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class FraudType(Enum):
    """15 fraud typologies with market impact data."""
    RUG_PULL = ("rug_pull", 8000, 96)
    PIG_BUTCHERING = ("pig_butchering", 7500, 94)
    MIXER_TUMBLING = ("mixer_tumbling", 5600, 98)
    CHAIN_OBFUSCATION = ("chain_obfuscation", 4300, 93)
    FAKE_TOKEN = ("fake_token", 2800, 97)
    FLASH_LOAN = ("flash_loan", 1900, 91)
    WASH_TRADING = ("wash_trading", 1500, 95)
    STRUCTURING = ("structuring", 1200, 99)
    VELOCITY_ATTACK = ("velocity_attack", 900, 94)
    PEEL_CHAIN = ("peel_chain", 700, 92)
    DUSTING = ("dusting", 500, 96)
    ADDRESS_POISONING = ("address_poisoning", 400, 97)
    APPROVAL_EXPLOIT = ("approval_exploit", 300, 93)
    SIM_SWAP = ("sim_swap", 200, 89)
    ROMANCE_SCAM = ("romance_scam", 200, 88)
    
    def __init__(self, code: str, market_impact_m: int, detection_target: int):
        self.code = code
        self.market_impact_m = market_impact_m
        self.detection_target = detection_target


class RiskLevel(Enum):
    """Risk classification levels."""
    SAFE = (0, 20, "SAFE", "✅")
    LOW = (21, 40, "LOW", "🟢")
    MEDIUM = (41, 60, "MEDIUM", "🟡")
    HIGH = (61, 80, "HIGH", "🟠")
    CRITICAL = (81, 100, "CRITICAL", "🔴")
    
    def __init__(self, min_score: int, max_score: int, label: str, emoji: str):
        self.min_score = min_score
        self.max_score = max_score
        self.label = label
        self.emoji = emoji
    
    @classmethod
    def from_score(cls, score: int) -> 'RiskLevel':
        for level in cls:
            if level.min_score <= score <= level.max_score:
                return level
        return cls.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
#                           DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    """Transaction data structure."""
    tx_id: str
    sender: str
    recipient: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    token: str = "USDC"
    chain: str = "ethereum"
    gas_price: Optional[float] = None
    block_number: Optional[int] = None
    
    # Enriched data (populated by engine)
    sender_history_count: int = 0
    recipient_history_count: int = 0
    sender_age_days: float = 0
    recipient_age_days: float = 0
    is_contract_recipient: bool = False
    sender_avg_amount: float = 0
    time_since_last_tx: float = 0


@dataclass
class FraudAnalysis:
    """Complete fraud analysis result."""
    tx_id: str
    timestamp: float
    
    # Scores (0-100)
    overall_score: int
    risk_level: RiskLevel
    
    # Component scores
    velocity_score: int = 0
    amount_score: int = 0
    pattern_score: int = 0
    graph_score: int = 0
    timing_score: int = 0
    typology_score: int = 0
    
    # Detected typologies
    detected_typologies: List[Tuple[FraudType, float]] = field(default_factory=list)
    primary_typology: Optional[FraudType] = None
    
    # AI Analysis
    ai_reasoning: str = ""
    ai_confidence: float = 0.0
    thinking_used: bool = False
    
    # Decisions
    approved: bool = True
    flagged: bool = False
    blocked: bool = False
    
    # Explanations
    explanations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Performance
    analysis_time_ms: float = 0
    model_inference_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.label,
            "risk_emoji": self.risk_level.emoji,
            "component_scores": {
                "velocity": self.velocity_score,
                "amount": self.amount_score,
                "pattern": self.pattern_score,
                "graph": self.graph_score,
                "timing": self.timing_score,
                "typology": self.typology_score,
            },
            "detected_typologies": [
                {"type": t.code, "confidence": round(c, 4)}
                for t, c in self.detected_typologies
            ],
            "primary_typology": self.primary_typology.code if self.primary_typology else None,
            "ai_reasoning": self.ai_reasoning,
            "ai_confidence": round(self.ai_confidence, 4),
            "thinking_used": self.thinking_used,
            "decisions": {
                "approved": self.approved,
                "flagged": self.flagged,
                "blocked": self.blocked,
            },
            "explanations": self.explanations,
            "recommendations": self.recommendations,
            "performance": {
                "total_ms": round(self.analysis_time_ms, 2),
                "model_ms": round(self.model_inference_time_ms, 2),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           WALLET PROFILER
# ═══════════════════════════════════════════════════════════════════════════════

class WalletProfiler:
    """Maintains behavioral profiles for wallet addresses."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.blacklist: set = set()
        self.whitelist: set = set()
        self.known_mixers: set = {
            "0x8589427373d6d84e98730d7795d8f6f8731fda16",  # Tornado Cash
            "0x722122df12d4e14e13ac3b6895a86e84145b6967",  # Tornado Router
            "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # Tornado Proxy
        }
        self.known_exchanges: set = {
            "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
            "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance US
            "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Coinbase
        }
    
    def get_profile(self, address: str) -> Dict[str, Any]:
        """Get or create wallet profile."""
        addr = address.lower()
        if addr not in self.profiles:
            self.profiles[addr] = {
                "address": addr,
                "transaction_count": 0,
                "total_volume": 0.0,
                "avg_amount": 0.0,
                "amounts": [],
                "timestamps": [],
                "counterparties": set(),
                "risk_history": [],
                "is_blacklisted": addr in self.blacklist,
                "is_whitelisted": addr in self.whitelist,
                "is_mixer": addr in self.known_mixers,
                "is_exchange": addr in self.known_exchanges,
                "created_at": time.time(),
            }
        return self.profiles[addr]
    
    def update_profile(self, address: str, tx: Transaction, analysis: FraudAnalysis):
        """Update profile after transaction analysis."""
        profile = self.get_profile(address)
        profile["transaction_count"] += 1
        profile["total_volume"] += tx.amount
        profile["avg_amount"] = profile["total_volume"] / profile["transaction_count"]
        profile["amounts"].append(tx.amount)
        profile["timestamps"].append(tx.timestamp)
        profile["counterparties"].add(tx.recipient if tx.sender.lower() == address.lower() else tx.sender)
        profile["risk_history"].append({
            "score": analysis.overall_score,
            "timestamp": tx.timestamp,
            "tx_id": tx.tx_id,
        })
        
        # Keep only last 100 items
        if len(profile["amounts"]) > 100:
            profile["amounts"] = profile["amounts"][-100:]
            profile["timestamps"] = profile["timestamps"][-100:]
            profile["risk_history"] = profile["risk_history"][-100:]


# ═══════════════════════════════════════════════════════════════════════════════
#                     THINKING MODE PROMPT ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class PromptEngineering:
    """Advanced prompt engineering for fraud detection with thinking mode."""
    
    SYSTEM_PROMPT = """You are an EXPERT blockchain fraud detection AI with deep knowledge of:
- 15 major fraud typologies ($35B+ market impact)
- Behavioral pattern analysis
- Transaction graph analytics
- Temporal anomaly detection
- DeFi attack vectors

Your task: Analyze transactions and provide PRECISE risk assessments.

CRITICAL: Lives and fortunes depend on your accuracy. Think carefully.
"""

    THINKING_PROMPT = """<think>
Analyze this transaction step-by-step for fraud indicators:

**TRANSACTION DATA:**
- TX ID: {tx_id}
- Amount: ${amount:,.2f} {token}
- Sender: {sender}
- Recipient: {recipient}
- Timestamp: {timestamp}

**SENDER PROFILE:**
- History: {sender_history_count} transactions
- Avg Amount: ${sender_avg_amount:,.2f}
- Account Age: {sender_age_days:.1f} days
- Time Since Last TX: {time_since_last_tx:.1f} hours

**RECIPIENT PROFILE:**
- History: {recipient_history_count} transactions
- Account Age: {recipient_age_days:.1f} days
- Is Contract: {is_contract_recipient}

**ANALYSIS CHECKLIST:**
1. Amount Anomaly: Is ${amount:,.2f} unusual vs sender's average ${sender_avg_amount:,.2f}?
2. Velocity: Is {time_since_last_tx:.1f} hours unusual transaction frequency?
3. New Recipient: Is recipient new ({recipient_history_count} txs)?
4. Wallet Age: Sender {sender_age_days:.1f} days, Recipient {recipient_age_days:.1f} days old
5. Structuring: Is amount near reporting thresholds ($10K, $50K)?
6. Mixer Pattern: Multiple small transactions to obfuscate?
7. Flash Loan Indicators: Same-block borrow/repay pattern?
8. Rug Pull Signs: Large outflow from new contract?

Consider each of the 15 typologies:
{typology_list}

Think through each indicator carefully...
</think>

Based on my analysis, here is the JSON fraud assessment:
"""

    FAST_PROMPT = """Analyze this transaction for fraud (JSON only):

TX: ${amount:,.2f} from {sender} to {recipient}
Sender: {sender_history_count} txs, avg ${sender_avg_amount:,.2f}
Recipient: {recipient_history_count} txs, age {recipient_age_days:.0f}d

Output JSON only:
{{"risk_score": 0-100, "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL", "primary_threat": "...", "confidence": 0-1}}"""

    TYPOLOGY_LIST = """
1. RUG_PULL: Large outflow from new contract, LP removal
2. PIG_BUTCHERING: Gradual trust-building, large final transfer
3. MIXER_TUMBLING: Known mixer addresses, obfuscation patterns
4. CHAIN_OBFUSCATION: Bridge usage, chain hopping
5. FAKE_TOKEN: Honeypot contracts, fake liquidity
6. FLASH_LOAN: Same-block borrow/repay, price manipulation
7. WASH_TRADING: Self-dealing, circular transactions
8. STRUCTURING: Amounts near $10K/$50K thresholds
9. VELOCITY_ATTACK: Burst of transactions in short time
10. PEEL_CHAIN: Gradual peeling of funds to new addresses
11. DUSTING: Tiny amounts to many addresses
12. ADDRESS_POISONING: Similar-looking addresses
13. APPROVAL_EXPLOIT: Unlimited token approvals
14. SIM_SWAP: Account recovery after compromise
15. ROMANCE_SCAM: Trust-based gradual transfers
"""

    @classmethod
    def get_thinking_prompt(cls, tx: Transaction) -> str:
        """Generate thinking mode prompt for deep analysis."""
        return cls.THINKING_PROMPT.format(
            tx_id=tx.tx_id,
            amount=tx.amount,
            token=tx.token,
            sender=tx.sender[:10] + "..." + tx.sender[-6:] if len(tx.sender) > 20 else tx.sender,
            recipient=tx.recipient[:10] + "..." + tx.recipient[-6:] if len(tx.recipient) > 20 else tx.recipient,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tx.timestamp)),
            sender_history_count=tx.sender_history_count,
            sender_avg_amount=tx.sender_avg_amount,
            sender_age_days=tx.sender_age_days,
            time_since_last_tx=tx.time_since_last_tx,
            recipient_history_count=tx.recipient_history_count,
            recipient_age_days=tx.recipient_age_days,
            is_contract_recipient=tx.is_contract_recipient,
            typology_list=cls.TYPOLOGY_LIST,
        )
    
    @classmethod
    def get_fast_prompt(cls, tx: Transaction) -> str:
        """Generate fast prompt for quick assessment."""
        return cls.FAST_PROMPT.format(
            amount=tx.amount,
            sender=tx.sender[:8] + "...",
            recipient=tx.recipient[:8] + "...",
            sender_history_count=tx.sender_history_count,
            sender_avg_amount=tx.sender_avg_amount,
            recipient_history_count=tx.recipient_history_count,
            recipient_age_days=tx.recipient_age_days,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#                     ULTIMATE FRAUD DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class UltimateFraudEngine:
    """
    🚀 ULTIMATE FRAUD DETECTION ENGINE
    
    Full GPU acceleration with Qwen3:8B thinking mode for maximum accuracy.
    Now with dedicated 8GB VRAM (chatbot moved to Gemini Cloud).
    """
    
    VERSION = "3.0.0-ultimate"
    
    def __init__(self):
        self.profiler = WalletProfiler()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.model_loaded = False
        self.total_analyses = 0
        self.total_blocked = 0
        self.total_flagged = 0
        self.latency_history: List[float] = []
        
        # Thresholds (tuned for accuracy)
        self.BLOCK_THRESHOLD = 75  # Block if score >= 75
        self.FLAG_THRESHOLD = 50   # Flag for review if score >= 50
        self.THINKING_THRESHOLD = 40  # Use thinking mode if initial score >= 40
        
        logger.info(f"🚀 Ultimate Fraud Engine v{self.VERSION} initialized")
        logger.info(f"   GPU Config: {GPUConfig.GPU_LAYERS} layers, {GPUConfig.NUM_CTX} context")
    
    async def initialize(self):
        """Initialize HTTP client and warm up model."""
        self.http_client = httpx.AsyncClient(timeout=60.0)
        await self._warm_up_model()
    
    async def _warm_up_model(self):
        """Load model to GPU VRAM."""
        logger.info("🔥 Warming up Qwen3:8B on GPU (full 8GB VRAM)...")
        
        try:
            start = time.time()
            response = await self.http_client.post(
                f"{GPUConfig.OLLAMA_URL}/api/generate",
                json={
                    "model": GPUConfig.MODEL,
                    "prompt": "Initialize fraud detection mode.",
                    "stream": False,
                    "keep_alive": GPUConfig.KEEP_ALIVE,
                    "options": {
                        **GPUConfig.get_thinking_options(),
                        "num_predict": 1,
                    }
                }
            )
            
            if response.status_code == 200:
                elapsed = (time.time() - start) * 1000
                self.model_loaded = True
                logger.info(f"   ✅ Model loaded in {elapsed:.0f}ms")
            else:
                logger.error(f"   ❌ Warm-up failed: {response.text}")
                
        except Exception as e:
            logger.error(f"   ❌ Warm-up error: {e}")
    
    async def analyze(self, tx: Transaction, force_thinking: bool = False) -> FraudAnalysis:
        """
        🎯 MAIN ANALYSIS FUNCTION
        
        Uses adaptive thinking mode:
        - Fast path: Low-risk transactions (< 40 initial score)
        - Thinking path: Higher risk or ambiguous transactions
        
        Args:
            tx: Transaction to analyze
            force_thinking: Force thinking mode regardless of initial score
            
        Returns:
            Complete FraudAnalysis result
        """
        start_time = time.time()
        
        # Create initial analysis
        analysis = FraudAnalysis(
            tx_id=tx.tx_id,
            timestamp=tx.timestamp,
            overall_score=0,
            risk_level=RiskLevel.SAFE,
        )
        
        try:
            # Step 1: Enrich transaction with profile data
            self._enrich_transaction(tx)
            
            # Step 2: Fast heuristic pre-screening
            initial_score = self._heuristic_screening(tx, analysis)
            
            # Step 3: Decide on analysis path
            use_thinking = force_thinking or initial_score >= self.THINKING_THRESHOLD
            
            # Step 4: AI Analysis
            model_start = time.time()
            
            if use_thinking:
                await self._thinking_analysis(tx, analysis)
            else:
                await self._fast_analysis(tx, analysis)
            
            analysis.model_inference_time_ms = (time.time() - model_start) * 1000
            analysis.thinking_used = use_thinking
            
            # Step 5: Final scoring and decisions
            self._finalize_analysis(analysis)
            
            # Step 6: Update profiles
            self.profiler.update_profile(tx.sender, tx, analysis)
            self.profiler.update_profile(tx.recipient, tx, analysis)
            
            # Update stats
            self.total_analyses += 1
            if analysis.blocked:
                self.total_blocked += 1
            if analysis.flagged:
                self.total_flagged += 1
            
        except Exception as e:
            logger.error(f"Analysis error for {tx.tx_id}: {e}")
            analysis.explanations.append(f"Analysis error: {str(e)}")
            analysis.overall_score = 50  # Default to medium risk on error
            analysis.risk_level = RiskLevel.MEDIUM
            analysis.flagged = True
        
        analysis.analysis_time_ms = (time.time() - start_time) * 1000
        self.latency_history.append(analysis.analysis_time_ms)
        
        return analysis
    
    def _enrich_transaction(self, tx: Transaction):
        """Enrich transaction with profile data."""
        sender_profile = self.profiler.get_profile(tx.sender)
        recipient_profile = self.profiler.get_profile(tx.recipient)
        
        tx.sender_history_count = sender_profile["transaction_count"]
        tx.recipient_history_count = recipient_profile["transaction_count"]
        tx.sender_avg_amount = sender_profile["avg_amount"]
        tx.sender_age_days = (time.time() - sender_profile["created_at"]) / 86400
        tx.recipient_age_days = (time.time() - recipient_profile["created_at"]) / 86400
        
        if sender_profile["timestamps"]:
            tx.time_since_last_tx = (tx.timestamp - sender_profile["timestamps"][-1]) / 3600
        else:
            tx.time_since_last_tx = 0
    
    def _heuristic_screening(self, tx: Transaction, analysis: FraudAnalysis) -> int:
        """Fast heuristic pre-screening (no AI, ~1ms)."""
        score = 0
        
        # Blacklist check (instant block)
        sender_profile = self.profiler.get_profile(tx.sender)
        recipient_profile = self.profiler.get_profile(tx.recipient)
        
        if sender_profile["is_blacklisted"] or recipient_profile["is_blacklisted"]:
            score += 90
            analysis.explanations.append("🚨 Blacklisted address detected")
        
        # Mixer check
        if recipient_profile["is_mixer"]:
            score += 70
            analysis.explanations.append("🌀 Transfer to known mixer/tumbler")
            analysis.detected_typologies.append((FraudType.MIXER_TUMBLING, 0.95))
        
        # Amount anomaly
        if tx.sender_avg_amount > 0:
            deviation = abs(tx.amount - tx.sender_avg_amount) / tx.sender_avg_amount
            if deviation > 10:  # 10x deviation
                score += 25
                analysis.velocity_score += 25
                analysis.explanations.append(f"⚠️ Amount {deviation:.1f}x higher than average")
        
        # New recipient with large amount
        if tx.recipient_history_count == 0 and tx.amount > 10000:
            score += 20
            analysis.explanations.append("⚠️ Large transfer to new wallet")
        
        # Structuring detection (near reporting thresholds)
        thresholds = [9900, 9950, 9990, 49900, 49950, 49990]
        for threshold in thresholds:
            if abs(tx.amount - threshold) < 100:
                score += 35
                analysis.explanations.append(f"⚠️ Amount near ${threshold} threshold (structuring)")
                analysis.detected_typologies.append((FraudType.STRUCTURING, 0.8))
                break
        
        # Velocity check (rapid transactions)
        if tx.time_since_last_tx > 0 and tx.time_since_last_tx < 0.1:  # < 6 minutes
            score += 20
            analysis.velocity_score += 30
            analysis.explanations.append("⚡ Rapid transaction velocity")
        
        # Very new accounts (< 1 day)
        if tx.sender_age_days < 1 and tx.amount > 5000:
            score += 15
            analysis.explanations.append("🆕 New account large transfer")
        
        return min(score, 100)
    
    async def _thinking_analysis(self, tx: Transaction, analysis: FraudAnalysis):
        """Deep analysis with thinking mode for accuracy."""
        prompt = PromptEngineering.get_thinking_prompt(tx)
        
        try:
            response = await self.http_client.post(
                f"{GPUConfig.OLLAMA_URL}/api/generate",
                json={
                    "model": GPUConfig.MODEL,
                    "prompt": prompt,
                    "system": PromptEngineering.SYSTEM_PROMPT,
                    "stream": False,
                    "keep_alive": GPUConfig.KEEP_ALIVE,
                    "format": "json",
                    "options": GPUConfig.get_thinking_options(),
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self._parse_ai_response(data.get("response", ""), analysis, thinking=True)
            else:
                logger.error(f"Thinking analysis failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Thinking analysis error: {e}")
    
    async def _fast_analysis(self, tx: Transaction, analysis: FraudAnalysis):
        """Fast analysis for low-risk transactions."""
        prompt = PromptEngineering.get_fast_prompt(tx)
        
        try:
            response = await self.http_client.post(
                f"{GPUConfig.OLLAMA_URL}/api/generate",
                json={
                    "model": GPUConfig.MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": GPUConfig.KEEP_ALIVE,
                    "format": "json",
                    "options": GPUConfig.get_fast_options(),
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self._parse_ai_response(data.get("response", ""), analysis, thinking=False)
            else:
                logger.error(f"Fast analysis failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Fast analysis error: {e}")
    
    def _parse_ai_response(self, response: str, analysis: FraudAnalysis, thinking: bool):
        """Parse AI response and update analysis."""
        # Extract thinking content if present
        thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if thinking_match:
            analysis.ai_reasoning = thinking_match.group(1).strip()
            response = response.replace(thinking_match.group(0), "")
        
        # Try to parse JSON
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                if "risk_score" in data:
                    analysis.overall_score = max(
                        analysis.overall_score,
                        int(data["risk_score"])
                    )
                
                if "confidence" in data:
                    analysis.ai_confidence = float(data["confidence"])
                
                if "primary_threat" in data and data["primary_threat"]:
                    threat = data["primary_threat"].upper().replace(" ", "_")
                    for ft in FraudType:
                        if ft.code.upper() == threat or ft.name == threat:
                            analysis.primary_typology = ft
                            analysis.detected_typologies.append((ft, analysis.ai_confidence))
                            break
                            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from AI response")
    
    def _finalize_analysis(self, analysis: FraudAnalysis):
        """Finalize scoring and make decisions."""
        # Aggregate component scores
        component_avg = (
            analysis.velocity_score +
            analysis.amount_score +
            analysis.pattern_score +
            analysis.graph_score +
            analysis.timing_score
        ) / 5
        
        # Combine with AI score (weighted)
        if analysis.ai_confidence > 0:
            analysis.overall_score = int(
                0.4 * component_avg +
                0.6 * analysis.overall_score
            )
        else:
            analysis.overall_score = int(component_avg)
        
        # Boost score based on detected typologies
        for typology, confidence in analysis.detected_typologies:
            if confidence > 0.8:
                analysis.typology_score += 20
            elif confidence > 0.5:
                analysis.typology_score += 10
        
        analysis.overall_score = min(100, analysis.overall_score + analysis.typology_score // 2)
        
        # Set risk level
        analysis.risk_level = RiskLevel.from_score(analysis.overall_score)
        
        # Make decisions
        if analysis.overall_score >= self.BLOCK_THRESHOLD:
            analysis.blocked = True
            analysis.approved = False
            analysis.explanations.append(f"🛑 BLOCKED: Risk score {analysis.overall_score} exceeds threshold")
        elif analysis.overall_score >= self.FLAG_THRESHOLD:
            analysis.flagged = True
            analysis.approved = True  # Approved but flagged for review
            analysis.explanations.append(f"⚠️ FLAGGED: Risk score {analysis.overall_score} requires review")
        else:
            analysis.approved = True
            analysis.explanations.append(f"✅ APPROVED: Risk score {analysis.overall_score} within safe range")
        
        # Add recommendations
        if analysis.blocked:
            analysis.recommendations.append("Reject transaction immediately")
            analysis.recommendations.append("Report to compliance team")
            analysis.recommendations.append("Consider adding sender to watchlist")
        elif analysis.flagged:
            analysis.recommendations.append("Require additional verification")
            analysis.recommendations.append("Monitor subsequent transactions")
            analysis.recommendations.append("Consider transaction hold period")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "version": self.VERSION,
            "total_analyses": self.total_analyses,
            "total_blocked": self.total_blocked,
            "total_flagged": self.total_flagged,
            "block_rate": self.total_blocked / max(1, self.total_analyses),
            "flag_rate": self.total_flagged / max(1, self.total_analyses),
            "profiles_tracked": len(self.profiler.profiles),
            "blacklist_size": len(self.profiler.blacklist),
            "whitelist_size": len(self.profiler.whitelist),
            "avg_latency_ms": np.mean(self.latency_history[-100:]) if self.latency_history else 0,
            "p95_latency_ms": np.percentile(self.latency_history[-100:], 95) if len(self.latency_history) >= 5 else 0,
            "p99_latency_ms": np.percentile(self.latency_history[-100:], 99) if len(self.latency_history) >= 5 else 0,
            "model_loaded": self.model_loaded,
            "gpu_config": {
                "layers": GPUConfig.GPU_LAYERS,
                "context": GPUConfig.NUM_CTX,
                "batch": GPUConfig.NUM_BATCH,
            }
        }
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
#                           SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[UltimateFraudEngine] = None

async def get_engine() -> UltimateFraudEngine:
    """Get or create the fraud engine singleton."""
    global _engine
    if _engine is None:
        _engine = UltimateFraudEngine()
        await _engine.initialize()
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
#                           QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    async def test():
        engine = await get_engine()
        
        # Test transaction
        tx = Transaction(
            tx_id="test_001",
            sender="0xAlice123456789",
            recipient="0xBob987654321",
            amount=50000.0,
            token="USDC",
        )
        
        print("\n" + "=" * 70)
        print("       ULTIMATE FRAUD ENGINE TEST")
        print("=" * 70)
        
        result = await engine.analyze(tx, force_thinking=True)
        
        print(f"\n🎯 Transaction: ${tx.amount:,.2f} USDC")
        print(f"📊 Risk Score: {result.overall_score}/100")
        print(f"🏷️ Risk Level: {result.risk_level.emoji} {result.risk_level.label}")
        print(f"⏱️ Analysis Time: {result.analysis_time_ms:.0f}ms")
        print(f"🧠 Thinking Used: {result.thinking_used}")
        
        if result.ai_reasoning:
            print(f"\n💭 AI Reasoning:\n{result.ai_reasoning[:500]}...")
        
        print(f"\n📋 Explanations:")
        for exp in result.explanations:
            print(f"   • {exp}")
        
        print("\n" + json.dumps(engine.get_stats(), indent=2))
        
        await engine.close()
    
    asyncio.run(test())
