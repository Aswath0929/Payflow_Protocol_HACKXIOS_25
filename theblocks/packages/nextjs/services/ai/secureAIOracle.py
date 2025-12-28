"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW SECURE AI ORACLE SERVICE                                  ║
║                                                                                       ║
║   Production-Grade AI-Powered Fraud Detection with Cryptographic Verification        ║
║                                                                                       ║
║   Features:                                                                           ║
║   • 100% LOCAL AI: Qwen3 LLM + Neural Network + ML (No cloud API needed!)            ║
║   • Qwen3 Local LLM: Running on RTX 4070 GPU via Ollama                              ║
║   • Cryptographic Signatures: All AI decisions are signed for on-chain verification ║
║   • Real-time WebSocket: Live transaction monitoring with <50ms latency             ║
║   • Privacy-Preserving: Zero-knowledge compatible compliance checking               ║
║   • Multi-Model Ensemble: Neural Net + ML + LLM for maximum accuracy                ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import secrets

# Cryptographic imports
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

# ML imports
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Local Neural Network (100% Offline)
from localNeuralNetwork import LocalNeuralNetworkEngine, NeuralNetworkPrediction

# Qwen3 Local LLM (100% Offline - RTX 4070 GPU)
from localLLMAnalyzer import Qwen3LocalAnalyzer, Qwen3Config

# Qwen3 Local LLM (100% Offline - RTX 4070 GPU)
from localLLMAnalyzer import Qwen3LocalAnalyzer, Qwen3Config

# API imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SecureAIOracle')

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Configuration from environment variables."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Not needed for Qwen3
    ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY", "")
    API_SECRET_KEY = os.getenv("API_SECRET_KEY", secrets.token_hex(32))
    
    # Thresholds
    BLOCK_THRESHOLD = 80
    REVIEW_THRESHOLD = 60
    MONITOR_THRESHOLD = 40
    
    # Model settings (Qwen3 Local LLM - No API key needed!)
    USE_OPENAI = False  # Disabled - we use Qwen3 locally
    USE_QWEN3 = True    # Use local Qwen3 model
    QWEN3_MODEL = "qwen3:8b"  # Best for RTX 4070 8GB
    
    # Performance
    MAX_ANALYSIS_TIME_MS = 100  # Target latency
    CACHE_TTL_SECONDS = 300     # 5 minute cache

# ═══════════════════════════════════════════════════════════════════════════════
#                              TYPES & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AlertType(Enum):
    VELOCITY_ANOMALY = "velocity_anomaly"
    AMOUNT_ANOMALY = "amount_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"
    MIXING_DETECTED = "mixing_detected"
    SANCTIONED_INTERACTION = "sanctioned_interaction"
    DUST_ATTACK = "dust_attack"
    SYBIL_ATTACK = "sybil_attack"
    FLASH_LOAN_ATTACK = "flash_loan_attack"
    WASH_TRADING = "wash_trading"
    LAYERING = "layering"
    AI_FLAGGED = "ai_flagged"

@dataclass
class WalletProfile:
    """Behavioral profile for a wallet address."""
    address: str
    transaction_count: int = 0
    total_volume: float = 0.0
    avg_amount: float = 0.0
    std_amount: float = 0.0
    avg_frequency_seconds: float = 86400.0
    last_transaction_time: float = 0.0
    
    # Feature vectors for ML
    amounts: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    counterparties: List[str] = field(default_factory=list)
    
    # Risk state
    current_risk_score: int = 0
    peak_risk_score: int = 0
    is_blacklisted: bool = False
    is_whitelisted: bool = False
    
    # ML features
    feature_vector: List[float] = field(default_factory=list)
    anomaly_score: float = 0.0
    cluster_id: int = -1

@dataclass
class SignedAnalysis:
    """AI analysis with cryptographic signature for on-chain verification."""
    transaction_id: str
    sender: str
    recipient: str
    amount: float
    timestamp: int
    
    # Risk scores (0-100)
    overall_score: int
    velocity_score: int
    amount_score: int
    pattern_score: int
    graph_score: int
    timing_score: int
    ai_score: int  # GPT-4 analysis score
    
    # Neural Network outputs (PRIMARY)
    neural_net_score: int  # Local neural network risk score
    neural_net_confidence: float  # Prediction confidence
    neural_net_risk_level: str  # Neural net's risk classification
    neural_net_is_anomaly: bool  # Autoencoder anomaly flag
    neural_net_explanation: str  # Neural network reasoning
    
    # ML model outputs
    isolation_forest_score: float
    cluster_anomaly: bool
    
    # Classification
    risk_level: str
    approved: bool
    flagged: bool
    blocked: bool
    
    # AI explanation (combined)
    ai_explanation: str
    alerts: List[str]
    
    # Cryptographic signature
    signature: str
    signer_address: str
    message_hash: str
    
    # Metadata
    model_version: str
    analysis_time_ms: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ═══════════════════════════════════════════════════════════════════════════════
#                         LOCAL ML MODELS (Fast Path)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalMLEngine:
    """
    Local ML models for fast fraud detection.
    These provide immediate results while GPT-4 does deep analysis.
    """
    
    def __init__(self):
        # Isolation Forest for anomaly detection
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # DBSCAN for cluster-based anomaly detection
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Training data buffer
        self.training_buffer: List[np.ndarray] = []
        self.is_trained = False
        self.min_training_samples = 50
        
        # Known patterns (pre-loaded threat intelligence)
        self.known_bad_patterns = self._load_threat_patterns()
        
        logger.info("LocalMLEngine initialized")
    
    def _load_threat_patterns(self) -> Dict:
        """Load known fraud patterns from threat intelligence."""
        return {
            "structuring_amounts": [2999, 9999, 14999, 49999],  # Just below thresholds
            "dust_threshold": 0.01,
            "velocity_burst_threshold": 10,  # Transactions per minute
            "round_trip_window": 3600,  # 1 hour
            "mixing_counterparty_threshold": 0.9,  # 90% unique counterparties
        }
    
    def extract_features(self, profile: WalletProfile, amount: float, timestamp: float) -> np.ndarray:
        """Extract ML features from wallet profile and transaction."""
        features = []
        
        # Amount features
        if profile.amounts:
            amounts = np.array(profile.amounts[-100:])  # Last 100 transactions
            features.extend([
                amount,
                np.mean(amounts) if len(amounts) > 0 else amount,
                np.std(amounts) if len(amounts) > 1 else 0,
                (amount - np.mean(amounts)) / (np.std(amounts) + 1e-6) if len(amounts) > 1 else 0,  # Z-score
                np.percentile(amounts, 25) if len(amounts) > 0 else amount,
                np.percentile(amounts, 75) if len(amounts) > 0 else amount,
            ])
        else:
            features.extend([amount, amount, 0, 0, amount, amount])
        
        # Velocity features
        if profile.timestamps:
            timestamps = np.array(profile.timestamps[-20:])
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                features.extend([
                    np.mean(intervals),
                    np.std(intervals),
                    np.min(intervals),
                    timestamp - profile.last_transaction_time,
                ])
            else:
                features.extend([86400, 0, 86400, 86400])
        else:
            features.extend([86400, 0, 86400, 86400])
        
        # Counterparty diversity
        if profile.counterparties:
            recent = profile.counterparties[-20:]
            unique_ratio = len(set(recent)) / len(recent)
            features.append(unique_ratio)
        else:
            features.append(0)
        
        # Transaction count features
        features.extend([
            profile.transaction_count,
            np.log1p(profile.total_volume),
        ])
        
        return np.array(features)
    
    def train_incremental(self, features: np.ndarray):
        """Incrementally train models with new data."""
        self.training_buffer.append(features)
        
        if len(self.training_buffer) >= self.min_training_samples:
            X = np.array(self.training_buffer)
            X_scaled = self.scaler.fit_transform(X)
            self.isolation_forest.fit(X_scaled)
            self.is_trained = True
            logger.info(f"Models trained on {len(self.training_buffer)} samples")
    
    def predict_anomaly(self, features: np.ndarray) -> Tuple[float, bool]:
        """
        Predict if transaction is anomalous.
        Returns (anomaly_score, is_anomaly)
        """
        if not self.is_trained:
            # Fallback to heuristics if not enough training data
            return 0.5, False
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Isolation Forest score (-1 = anomaly, 1 = normal)
        if_score = self.isolation_forest.score_samples(features_scaled)[0]
        
        # Convert to 0-1 range (0 = normal, 1 = anomaly)
        anomaly_score = 1 - (if_score + 1) / 2
        is_anomaly = anomaly_score > 0.6
        
        return float(anomaly_score), is_anomaly

# ═══════════════════════════════════════════════════════════════════════════════
#                         QWEN3 LOCAL LLM INTEGRATION (Deep Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

class GPT4Analyzer:
    """
    UPGRADED: Now uses Qwen3 Local LLM (100% Offline - RTX 4070 GPU)!
    
    Provides GPT-4 level reasoning for fraud detection,
    running entirely on your local GPU. No cloud API needed.
    
    Model: Qwen3 (Latest 2025 release from Alibaba)
    VRAM: ~5GB on RTX 4070 (8GB available)
    Inference: <500ms per analysis
    """
    
    SYSTEM_PROMPT = """You are an expert financial crime analyst specializing in cryptocurrency and stablecoin fraud detection. 
You analyze transactions for money laundering, terrorist financing, sanctions evasion, and other financial crimes.

Your analysis should consider:
1. Transaction patterns and velocity
2. Amount anomalies and structuring
3. Counterparty relationships and mixing services
4. Timing patterns and behavioral deviations
5. Known fraud typologies (layering, smurfing, round-tripping)

Respond in JSON format with:
{
  "risk_score": 0-100,
  "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL",
  "approved": true/false,
  "alerts": ["list of specific concerns"],
  "explanation": "detailed analysis",
  "recommendations": ["action items for compliance team"]
}

Be precise, specific, and cite the exact patterns that concern you."""

    def __init__(self):
        # PRIMARY: Qwen3 Local LLM (Offline)
        self.qwen3 = Qwen3LocalAnalyzer()
        self.enabled = False
        self.is_local = True
        
        # Initialize Qwen3 in background
        asyncio.create_task(self._init_qwen3())
        
        logger.info("GPT4Analyzer upgraded to Qwen3 Local LLM")
        logger.info(f"  Model: {Qwen3Config.MODEL_NAME}")
        logger.info("  Running 100% offline on your GPU!")
    
    async def _init_qwen3(self):
        """Initialize Qwen3 asynchronously."""
        try:
            available = await self.qwen3.check_availability()
            if available:
                await self.qwen3.warm_up()
                self.enabled = True
                logger.info("✅ Qwen3 Local LLM ready!")
            else:
                logger.warning("⚠️ Qwen3 not available. Run: ollama pull qwen3:8b")
                self.enabled = False
        except Exception as e:
            logger.warning(f"Qwen3 init failed: {e}")
            self.enabled = False
    
    async def analyze(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        sender_profile: WalletProfile,
        recipient_profile: WalletProfile,
        local_scores: Dict[str, int]
    ) -> Dict:
        """
        Deep analysis using Qwen3 Local LLM (100% Offline).
        """
        if not self.enabled:
            # Try to re-enable
            await self._init_qwen3()
            if not self.enabled:
                return self._fallback_analysis(local_scores)
        
        # Convert WalletProfile to dict for Qwen3
        sender_dict = {
            "transaction_count": sender_profile.transaction_count,
            "total_volume": sender_profile.total_volume,
            "avg_amount": sender_profile.avg_amount,
            "current_risk_score": sender_profile.current_risk_score,
        }
        
        recipient_dict = {
            "transaction_count": recipient_profile.transaction_count,
            "total_volume": recipient_profile.total_volume,
        }
        
        try:
            result = await self.qwen3.analyze(
                transaction_id=transaction_id,
                sender=sender,
                recipient=recipient,
                amount=amount,
                sender_profile=sender_dict,
                recipient_profile=recipient_dict,
                local_scores=local_scores
            )
            
            return {
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "approved": result.approved,
                "alerts": result.alerts,
                "explanation": result.explanation,
                "recommendations": result.recommendations,
                "confidence": result.confidence,
                "model_used": result.model_used,
                "is_local": result.is_local,
                "inference_time_ms": result.inference_time_ms,
            }
        except Exception as e:
            logger.error(f"Qwen3 analysis failed: {e}")
            return self._fallback_analysis(local_scores)
    
    def _fallback_analysis(self, local_scores: Dict[str, int]) -> Dict:
        """Fallback when Qwen3 is unavailable - uses ML models only."""
        # Calculate numeric scores only
        numeric_scores = [v for v in local_scores.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        avg_score = sum(numeric_scores) / max(len(numeric_scores), 1)
        
        return {
            "risk_score": int(avg_score),
            "risk_level": self._score_to_level(int(avg_score)),
            "approved": avg_score < Config.BLOCK_THRESHOLD,
            "alerts": ["Qwen3 LLM unavailable - using ML fallback"],
            "explanation": "Analysis performed using Neural Network + ML models (Qwen3 LLM offline). Run: ollama serve && ollama pull qwen3:8b",
            "recommendations": ["Enable Qwen3 for detailed LLM analysis"],
            "confidence": 0.6,
            "model_used": "fallback",
            "is_local": True,
            "inference_time_ms": 0,
        }
    
    def _score_to_level(self, score: int) -> str:
        if score <= 20: return "SAFE"
        if score <= 40: return "LOW"
        if score <= 60: return "MEDIUM"
        if score <= 80: return "HIGH"
        return "CRITICAL"

# ═══════════════════════════════════════════════════════════════════════════════
#                         CRYPTOGRAPHIC SIGNER
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoSigner:
    """
    Signs AI analysis results for on-chain verification.
    Uses Ethereum's ECDSA signatures compatible with Solidity's ecrecover.
    """
    
    def __init__(self, private_key: str = None):
        if private_key:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
            logger.info(f"CryptoSigner initialized with address: {self.address}")
        else:
            # Generate new key for development
            self.account = Account.create()
            self.address = self.account.address
            logger.warning(f"Generated new oracle key: {self.address}")
            logger.warning(f"Private key (save this!): {self.account.key.hex()}")
    
    def sign_analysis(self, analysis: Dict) -> Tuple[str, str]:
        """
        Sign analysis result for on-chain verification.
        Returns (signature, message_hash)
        """
        # Create deterministic message from analysis
        message = self._create_message(analysis)
        message_hash = Web3.keccak(text=message).hex()
        
        # Sign with EIP-191
        signable = encode_defunct(text=message)
        signed = self.account.sign_message(signable)
        
        return signed.signature.hex(), message_hash
    
    def _create_message(self, analysis: Dict) -> str:
        """Create deterministic message for signing."""
        # Include only fields needed for on-chain verification
        return json.dumps({
            "transaction_id": analysis.get("transaction_id"),
            "overall_score": analysis.get("overall_score"),
            "approved": analysis.get("approved"),
            "blocked": analysis.get("blocked"),
            "timestamp": analysis.get("timestamp"),
        }, sort_keys=True)
    
    def verify_signature(self, message: str, signature: str) -> str:
        """Verify signature and return signer address."""
        signable = encode_defunct(text=message)
        return Account.recover_message(signable, signature=signature)

# ═══════════════════════════════════════════════════════════════════════════════
#                         MAIN ORACLE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SecureAIOracle:
    """
    Production-grade AI Oracle combining:
    1. LOCAL NEURAL NETWORK for 100% offline detection (PRIMARY)
    2. Local ML (Isolation Forest + DBSCAN) for anomaly detection
    3. GPT-4 for intelligent deep analysis (SECONDARY/ENHANCEMENT)
    4. Cryptographic signatures for on-chain verification
    
    The system works completely offline using neural networks.
    GPT-4 is OPTIONAL and only enhances the analysis when available.
    """
    
    MODEL_VERSION = "PayFlow-SecureAI-v2.1.0-NeuralNet"
    
    def __init__(self):
        # PRIMARY: Local Neural Network (100% Offline)
        self.neural_net = LocalNeuralNetworkEngine()
        
        # SECONDARY: Traditional ML Engine
        self.ml_engine = LocalMLEngine()
        
        # Qwen3 Local LLM (100% Offline - RTX 4070 GPU)
        self.gpt4 = GPT4Analyzer()  # Actually uses Qwen3 now!
        
        # Cryptographic signer
        self.signer = CryptoSigner(Config.ORACLE_PRIVATE_KEY if Config.ORACLE_PRIVATE_KEY else None)
        
        # Wallet profiles
        self.profiles: Dict[str, WalletProfile] = {}
        
        # Caches
        self.analysis_cache: Dict[str, SignedAnalysis] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "total_blocked": 0,
            "total_flagged": 0,
            "avg_latency_ms": 0,
            "qwen3_calls": 0,  # Renamed from gpt4_calls
            "cache_hits": 0,
            "neural_net_predictions": 0,
        }
        
        # Blacklist/Whitelist
        self.blacklist: set = set()
        self.whitelist: set = set()
        
        # Known bad actors (would come from Chainalysis in production)
        self.known_bad_actors: set = set()
        
        logger.info(f"SecureAIOracle initialized - Version: {self.MODEL_VERSION}")
        logger.info(f"Oracle signer address: {self.signer.address}")
        logger.info(f"Neural Network: ENABLED (100% Offline)")
        logger.info(f"Qwen3 Local LLM: {'ENABLED (RTX 4070 GPU)' if self.gpt4.enabled else 'STARTING... Run: ollama serve && ollama pull qwen3:8b'}")
    
    def get_or_create_profile(self, address: str) -> WalletProfile:
        """Get or create wallet profile."""
        addr = address.lower()
        if addr not in self.profiles:
            self.profiles[addr] = WalletProfile(address=addr)
        return self.profiles[addr]
    
    async def analyze_transaction(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: Optional[int] = None,
        use_gpt4: bool = True
    ) -> SignedAnalysis:
        """
        Comprehensive transaction analysis with AI + ML + Signatures.
        
        Pipeline:
        1. Check cache
        2. Local ML scoring (fast)
        3. GPT-4 deep analysis (if enabled)
        4. Combine scores
        5. Sign result
        6. Return signed analysis
        """
        start_time = time.time()
        timestamp = timestamp or int(time.time())
        
        # Check cache
        cache_key = f"{transaction_id}:{sender}:{recipient}:{amount}"
        if cache_key in self.analysis_cache:
            cache_time = self.cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < Config.CACHE_TTL_SECONDS:
                self.stats["cache_hits"] += 1
                return self.analysis_cache[cache_key]
        
        # Get profiles
        sender_profile = self.get_or_create_profile(sender)
        recipient_profile = self.get_or_create_profile(recipient)
        
        # Quick blacklist check
        if sender.lower() in self.blacklist or recipient.lower() in self.blacklist:
            return self._create_blocked_analysis(
                transaction_id, sender, recipient, amount, timestamp,
                "Address is blacklisted", start_time
            )
        
        # ═══ PHASE 1: NEURAL NETWORK ANALYSIS (100% Offline - PRIMARY) ═══
        neural_result: NeuralNetworkPrediction = self.neural_net.predict(
            sender=sender,
            recipient=recipient,
            amount=amount,
            timestamp=timestamp
        )
        self.stats["neural_net_predictions"] += 1
        
        logger.info(f"Neural Network: Risk={neural_result.risk_score}, Level={neural_result.risk_level}")
        
        # ═══ PHASE 2: Local ML Scoring (Fast Path - SECONDARY) ═══
        local_scores = await self._local_ml_analysis(
            sender_profile, recipient_profile, amount, timestamp
        )
        
        # Add neural network scores to local_scores
        local_scores["neural_network"] = neural_result.risk_score
        local_scores["nn_anomaly"] = neural_result.is_anomaly
        
        # ═══ PHASE 3: Qwen3 Local LLM Analysis (100% Offline - RTX 4070) ═══
        qwen3_result = None
        if use_gpt4 and self.gpt4.enabled:
            # Only call Qwen3 for non-trivial transactions OR when neural net flags it
            preliminary_score = sum([v for k, v in local_scores.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]) / len(local_scores)
            if neural_result.is_anomaly or preliminary_score > 30 or amount > 5000:
                qwen3_result = await self.gpt4.analyze(
                    transaction_id, sender, recipient, amount,
                    sender_profile, recipient_profile, local_scores
                )
                self.stats["qwen3_calls"] += 1
        
        # ═══ PHASE 4: Combine Scores (Neural Net weighted heavily) ═══
        final_analysis = self._combine_scores_with_neural_net(
            transaction_id, sender, recipient, amount, timestamp,
            neural_result, local_scores, qwen3_result
        )
        
        # ═══ PHASE 4: Sign Result ═══
        signature, message_hash = self.signer.sign_analysis(final_analysis)
        
        # ═══ PHASE 5: Create Signed Analysis ═══
        analysis_time = (time.time() - start_time) * 1000
        
        signed_analysis = SignedAnalysis(
            transaction_id=transaction_id,
            sender=sender,
            recipient=recipient,
            amount=amount,
            timestamp=timestamp,
            overall_score=final_analysis["overall_score"],
            velocity_score=local_scores["velocity"],
            amount_score=local_scores["amount"],
            pattern_score=local_scores["pattern"],
            graph_score=local_scores["graph"],
            timing_score=local_scores["timing"],
            ai_score=qwen3_result["risk_score"] if qwen3_result else 0,
            # Neural Network outputs (PRIMARY)
            neural_net_score=neural_result.risk_score,
            neural_net_confidence=neural_result.confidence,
            neural_net_risk_level=neural_result.risk_level,
            neural_net_is_anomaly=neural_result.is_anomaly,
            neural_net_explanation=neural_result.explanation,
            # Traditional ML outputs
            isolation_forest_score=local_scores["isolation_forest"],
            cluster_anomaly=local_scores.get("cluster_anomaly", False),
            risk_level=final_analysis["risk_level"],
            approved=final_analysis["approved"],
            flagged=final_analysis["flagged"],
            blocked=final_analysis["blocked"],
            ai_explanation=self._build_combined_explanation(neural_result, qwen3_result),
            alerts=final_analysis["alerts"],
            signature=signature,
            signer_address=self.signer.address,
            message_hash=message_hash,
            model_version=self.MODEL_VERSION,
            analysis_time_ms=analysis_time
        )
        
        # Update statistics
        self.stats["total_analyses"] += 1
        if signed_analysis.blocked:
            self.stats["total_blocked"] += 1
        if signed_analysis.flagged:
            self.stats["total_flagged"] += 1
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (self.stats["total_analyses"] - 1) + analysis_time)
            / self.stats["total_analyses"]
        )
        
        # Update profile
        self._update_profile(sender_profile, amount, timestamp, recipient)
        
        # Cache result
        self.analysis_cache[cache_key] = signed_analysis
        self.cache_timestamps[cache_key] = time.time()
        
        # Train ML models incrementally
        features = self.ml_engine.extract_features(sender_profile, amount, timestamp)
        self.ml_engine.train_incremental(features)
        
        logger.info(
            f"Transaction {transaction_id[:8]}... analyzed: "
            f"Score={signed_analysis.overall_score}, "
            f"Level={signed_analysis.risk_level}, "
            f"Time={analysis_time:.1f}ms, "
            f"GPT4={gpt4_result is not None}"
        )
        
        return signed_analysis
    
    async def _local_ml_analysis(
        self,
        sender_profile: WalletProfile,
        recipient_profile: WalletProfile,
        amount: float,
        timestamp: float
    ) -> Dict[str, Any]:
        """Fast local ML analysis."""
        
        scores = {
            "velocity": 0,
            "amount": 0,
            "pattern": 0,
            "graph": 0,
            "timing": 0,
            "isolation_forest": 0.0,
            "cluster_anomaly": False,
        }
        
        # Velocity analysis
        if sender_profile.timestamps:
            recent = [t for t in sender_profile.timestamps if t > timestamp - 3600]
            velocity = len(recent)
            expected = max(sender_profile.transaction_count / max(1, (timestamp - sender_profile.timestamps[0]) / 3600), 0.1)
            velocity_ratio = velocity / expected if expected > 0 else 0
            
            if velocity_ratio > 10:
                scores["velocity"] = 90
            elif velocity_ratio > 5:
                scores["velocity"] = 70
            elif velocity_ratio > 3:
                scores["velocity"] = 50
            elif velocity_ratio > 2:
                scores["velocity"] = 30
        
        # Amount analysis
        if sender_profile.amounts:
            amounts = np.array(sender_profile.amounts)
            mean, std = np.mean(amounts), np.std(amounts)
            z_score = (amount - mean) / (std + 1e-6) if std > 0 else 0
            
            if abs(z_score) > 4:
                scores["amount"] = 80
            elif abs(z_score) > 3:
                scores["amount"] = 60
            elif abs(z_score) > 2:
                scores["amount"] = 40
            
            # Structuring detection
            for threshold in [3000, 10000, 15000, 50000]:
                if threshold * 0.9 <= amount < threshold:
                    scores["amount"] = max(scores["amount"], 70)
        
        # Pattern analysis (mixing detection)
        if len(sender_profile.counterparties) >= 10:
            recent = sender_profile.counterparties[-20:]
            unique_ratio = len(set(recent)) / len(recent)
            if unique_ratio > 0.9:
                scores["pattern"] = 75
            elif unique_ratio > 0.8:
                scores["pattern"] = 55
        
        # Graph analysis (bad actor proximity)
        if sender_profile.address in self.known_bad_actors:
            scores["graph"] = 100
        elif recipient_profile.address in self.known_bad_actors:
            scores["graph"] = 95
        
        # Timing analysis
        hour = datetime.fromtimestamp(timestamp).hour
        if hour in range(0, 5):  # Late night
            scores["timing"] = 20
        
        # ML model prediction
        features = self.ml_engine.extract_features(sender_profile, amount, timestamp)
        anomaly_score, is_anomaly = self.ml_engine.predict_anomaly(features)
        scores["isolation_forest"] = anomaly_score
        scores["cluster_anomaly"] = is_anomaly
        
        return scores
    
    def _combine_scores_with_neural_net(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: int,
        neural_result: NeuralNetworkPrediction,
        local_scores: Dict,
        gpt4_result: Optional[Dict]
    ) -> Dict:
        """
        Combine Neural Network + Local ML + GPT-4 scores.
        Neural Network is PRIMARY (50% weight).
        Local ML is SECONDARY (30% weight).
        GPT-4 is ENHANCEMENT (20% weight) - only when available.
        """
        
        # Weight allocation - Neural Network is PRIMARY
        weights = {
            "neural_network": 0.50,  # PRIMARY - Local Neural Net
            "velocity": 0.08,
            "amount": 0.08,
            "pattern": 0.06,
            "graph": 0.04,
            "timing": 0.02,
            "isolation_forest": 0.02,
            "gpt4": 0.20,  # OPTIONAL enhancement
        }
        
        # Calculate weighted score
        score = 0
        
        # PRIMARY: Neural Network (50%)
        score += neural_result.risk_score * weights["neural_network"]
        
        # SECONDARY: Traditional ML (30%)
        score += local_scores["velocity"] * weights["velocity"]
        score += local_scores["amount"] * weights["amount"]
        score += local_scores["pattern"] * weights["pattern"]
        score += local_scores["graph"] * weights["graph"]
        score += local_scores["timing"] * weights["timing"]
        score += local_scores["isolation_forest"] * 100 * weights["isolation_forest"]
        
        # OPTIONAL: GPT-4 Enhancement (20%)
        if gpt4_result:
            score += gpt4_result["risk_score"] * weights["gpt4"]
        else:
            # Redistribute GPT-4 weight to Neural Network (stays offline)
            score = score / (1 - weights["gpt4"])
        
        overall_score = int(min(100, max(0, score)))
        
        # Use Neural Network's risk level as primary, but can escalate
        if overall_score <= 20:
            risk_level = "SAFE"
        elif overall_score <= 40:
            risk_level = "LOW"
        elif overall_score <= 60:
            risk_level = "MEDIUM"
        elif overall_score <= 80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Collect alerts (Neural Network flags are PRIMARY)
        alerts = list(neural_result.flags) if neural_result.flags else []
        
        if local_scores["velocity"] >= 50:
            alerts.append(AlertType.VELOCITY_ANOMALY.value)
        if local_scores["amount"] >= 50:
            alerts.append(AlertType.AMOUNT_ANOMALY.value)
        if local_scores["pattern"] >= 50:
            alerts.append(AlertType.MIXING_DETECTED.value)
        if local_scores["graph"] >= 50:
            alerts.append(AlertType.SANCTIONED_INTERACTION.value)
        if local_scores.get("cluster_anomaly"):
            alerts.append(AlertType.PATTERN_ANOMALY.value)
        if neural_result.is_anomaly:
            alerts.append(AlertType.AI_FLAGGED.value)
        if gpt4_result and gpt4_result.get("alerts"):
            alerts.extend(gpt4_result["alerts"][:3])
        
        return {
            "transaction_id": transaction_id,
            "overall_score": overall_score,
            "risk_level": risk_level,
            "approved": overall_score < Config.BLOCK_THRESHOLD,
            "flagged": Config.REVIEW_THRESHOLD <= overall_score < Config.BLOCK_THRESHOLD,
            "blocked": overall_score >= Config.BLOCK_THRESHOLD,
            "alerts": list(set(alerts)),
            "timestamp": timestamp,
            "neural_network_risk_level": neural_result.risk_level,
            "neural_network_confidence": neural_result.confidence,
        }
    
    def _combine_scores(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: int,
        local_scores: Dict,
        gpt4_result: Optional[Dict]
    ) -> Dict:
        """Combine local ML and GPT-4 scores (legacy fallback)."""
        
        # Weight allocation
        weights = {
            "velocity": 0.20,
            "amount": 0.20,
            "pattern": 0.15,
            "graph": 0.15,
            "timing": 0.05,
            "isolation_forest": 0.10,
            "gpt4": 0.15,
        }
        
        # Calculate weighted score
        score = 0
        score += local_scores["velocity"] * weights["velocity"]
        score += local_scores["amount"] * weights["amount"]
        score += local_scores["pattern"] * weights["pattern"]
        score += local_scores["graph"] * weights["graph"]
        score += local_scores["timing"] * weights["timing"]
        score += local_scores["isolation_forest"] * 100 * weights["isolation_forest"]
        
        if gpt4_result:
            score += gpt4_result["risk_score"] * weights["gpt4"]
        else:
            # Redistribute GPT-4 weight to local scores
            score = score / (1 - weights["gpt4"])
        
        overall_score = int(min(100, max(0, score)))
        
        # Determine risk level
        if overall_score <= 20:
            risk_level = "SAFE"
        elif overall_score <= 40:
            risk_level = "LOW"
        elif overall_score <= 60:
            risk_level = "MEDIUM"
        elif overall_score <= 80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Collect alerts
        alerts = []
        if local_scores["velocity"] >= 50:
            alerts.append(AlertType.VELOCITY_ANOMALY.value)
        if local_scores["amount"] >= 50:
            alerts.append(AlertType.AMOUNT_ANOMALY.value)
        if local_scores["pattern"] >= 50:
            alerts.append(AlertType.MIXING_DETECTED.value)
        if local_scores["graph"] >= 50:
            alerts.append(AlertType.SANCTIONED_INTERACTION.value)
        if local_scores["cluster_anomaly"]:
            alerts.append(AlertType.PATTERN_ANOMALY.value)
        if gpt4_result and gpt4_result.get("alerts"):
            alerts.extend(gpt4_result["alerts"][:3])
        
        return {
            "transaction_id": transaction_id,
            "overall_score": overall_score,
            "risk_level": risk_level,
            "approved": overall_score < Config.BLOCK_THRESHOLD,
            "flagged": Config.REVIEW_THRESHOLD <= overall_score < Config.BLOCK_THRESHOLD,
            "blocked": overall_score >= Config.BLOCK_THRESHOLD,
            "alerts": list(set(alerts)),
            "timestamp": timestamp,
        }
    
    def _create_blocked_analysis(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: int,
        reason: str,
        start_time: float
    ) -> SignedAnalysis:
        """Create a blocked analysis for blacklisted addresses."""
        analysis_time = (time.time() - start_time) * 1000
        
        analysis_dict = {
            "transaction_id": transaction_id,
            "overall_score": 100,
            "approved": False,
            "blocked": True,
            "timestamp": timestamp,
        }
        
        signature, message_hash = self.signer.sign_analysis(analysis_dict)
        
        self.stats["total_analyses"] += 1
        self.stats["total_blocked"] += 1
        
        return SignedAnalysis(
            transaction_id=transaction_id,
            sender=sender,
            recipient=recipient,
            amount=amount,
            timestamp=timestamp,
            overall_score=100,
            velocity_score=0,
            amount_score=0,
            pattern_score=0,
            graph_score=0,
            timing_score=0,
            ai_score=0,
            isolation_forest_score=1.0,
            cluster_anomaly=True,
            risk_level="CRITICAL",
            approved=False,
            flagged=False,
            blocked=True,
            ai_explanation=reason,
            alerts=["BLACKLISTED"],
            signature=signature,
            signer_address=self.signer.address,
            message_hash=message_hash,
            model_version=self.MODEL_VERSION,
            analysis_time_ms=analysis_time
        )
    
    def _build_combined_explanation(
        self,
        neural_result: NeuralNetworkPrediction,
        gpt4_result: Optional[Dict]
    ) -> str:
        """
        Build combined explanation from Neural Network (primary) + GPT-4 (optional).
        """
        explanations = []
        
        # Neural Network explanation (PRIMARY - always present)
        explanations.append(f"🧠 Local Neural Network ({neural_result.confidence:.0%} confidence): {neural_result.explanation}")
        
        # GPT-4 explanation (OPTIONAL enhancement)
        if gpt4_result and gpt4_result.get("explanation"):
            explanations.append(f"🤖 GPT-4 Enhancement: {gpt4_result['explanation']}")
        else:
            explanations.append("📡 Mode: 100% Offline (Neural Network + Local ML only)")
        
        return " | ".join(explanations)
    
    def _update_profile(self, profile: WalletProfile, amount: float, timestamp: float, counterparty: str):
        """Update wallet profile with new transaction."""
        profile.amounts.append(amount)
        profile.timestamps.append(timestamp)
        profile.counterparties.append(counterparty.lower())
        
        profile.transaction_count += 1
        profile.total_volume += amount
        profile.avg_amount = profile.total_volume / profile.transaction_count
        
        if len(profile.amounts) > 1:
            profile.std_amount = float(np.std(profile.amounts))
        
        if len(profile.timestamps) > 1:
            intervals = np.diff(profile.timestamps)
            profile.avg_frequency_seconds = float(np.mean(intervals))
        
        profile.last_transaction_time = timestamp
    
    def blacklist_address(self, address: str, reason: str = ""):
        """Add address to blacklist."""
        self.blacklist.add(address.lower())
        logger.warning(f"Address blacklisted: {address[:10]}... Reason: {reason}")
    
    def whitelist_address(self, address: str):
        """Add address to whitelist."""
        self.whitelist.add(address.lower())
    
    def add_known_bad_actor(self, address: str):
        """Add known bad actor."""
        self.known_bad_actors.add(address.lower())
        self.blacklist_address(address, "Known bad actor")
    
    def get_statistics(self) -> Dict:
        """Get oracle statistics."""
        return {
            **self.stats,
            "total_profiles": len(self.profiles),
            "blacklist_size": len(self.blacklist),
            "whitelist_size": len(self.whitelist),
            "model_version": self.MODEL_VERSION,
            "oracle_address": self.signer.address,
            "gpt4_enabled": self.gpt4.enabled,
            "ml_trained": self.ml_engine.is_trained,
        }

# ═══════════════════════════════════════════════════════════════════════════════
#                         SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_oracle: Optional[SecureAIOracle] = None

def get_oracle() -> SecureAIOracle:
    """Get singleton oracle instance."""
    global _oracle
    if _oracle is None:
        _oracle = SecureAIOracle()
    return _oracle
