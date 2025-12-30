"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW 15-TYPOLOGY FRAUD DETECTION ENGINE                        ║
║                                                                                       ║
║   Comprehensive Fraud Pattern Detection Based on 2025 Research                       ║
║   NOW WITH GNN FUSION - GraphSAGE-enhanced typology detection                        ║
║                                                                                       ║
║   15 Fraud Patterns by Market Impact:                                                ║
║   ┌─────────────────────────────────────────────────────────────────────────────┐    ║
║   │  1. Rug Pulls ............. $8B+ losses ..... 96% detection target          │    ║
║   │  2. Pig Butchering ........ $7.5B losses .... 94% detection target          │    ║
║   │  3. Mixer/Tumbling ........ $5.6B volume .... 98% detection target          │    ║
║   │  4. Chain Obfuscation ..... $4.3B volume .... 93% detection target          │    ║
║   │  5. Fake Tokens ........... $2.8B losses .... 97% detection target          │    ║
║   │  6. Flash Loan Attacks .... $1.9B losses .... 91% detection target          │    ║
║   │  7. Wash Trading .......... $1.5B volume .... 95% detection target          │    ║
║   │  8. Structuring/Smurfing .. $1.2B volume .... 99% detection target          │    ║
║   │  9. Velocity Attacks ...... $0.9B volume .... 94% detection target          │    ║
║   │ 10. Peel Chains ........... $0.7B volume .... 92% detection target          │    ║
║   │ 11. Dusting Attacks ....... $0.5B volume .... 96% detection target          │    ║
║   │ 12. Address Poisoning ..... $0.4B losses .... 97% detection target          │    ║
║   │ 13. Approval Exploits ..... $0.3B losses .... 93% detection target          │    ║
║   │ 14. SIM Swap Fraud ........ $0.2B losses .... 89% detection target          │    ║
║   │ 15. Romance Scams ......... $0.2B losses .... 88% detection target          │    ║
║   └─────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                       ║
║   GNN FUSION:                                                                         ║
║   • GraphSAGE embeddings enhance rule-based detection                               ║
║   • 64-dim graph features fused with 34-dim transaction features                    ║
║   • Neighbor-aware pattern recognition for chain detection                          ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import defaultdict
import math

# Import GNN Engine for fusion (with fallback)
try:
    from .graphNeuralNetwork import get_gnn_engine, GNNPrediction, GNNFraudTypology
    GNN_AVAILABLE = True
except ImportError:
    try:
        from graphNeuralNetwork import get_gnn_engine, GNNPrediction, GNNFraudTypology
        GNN_AVAILABLE = True
    except ImportError:
        GNN_AVAILABLE = False
        GNNPrediction = None


# ═══════════════════════════════════════════════════════════════════════════════
#                              FRAUD TYPOLOGY ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class FraudTypology(Enum):
    """15 fraud typologies ordered by market impact."""
    RUG_PULL = ("rug_pull", "Rug Pull", 8000)              # $8B
    PIG_BUTCHERING = ("pig_butchering", "Pig Butchering", 7500)  # $7.5B
    MIXER_TUMBLING = ("mixer_tumbling", "Mixer/Tumbling", 5600)  # $5.6B
    CHAIN_OBFUSCATION = ("chain_obfuscation", "Chain Obfuscation", 4300)
    FAKE_TOKEN = ("fake_token", "Fake Token Scam", 2800)
    FLASH_LOAN = ("flash_loan", "Flash Loan Attack", 1900)
    WASH_TRADING = ("wash_trading", "Wash Trading", 1500)
    STRUCTURING = ("structuring", "Structuring/Smurfing", 1200)
    VELOCITY_ATTACK = ("velocity_attack", "Velocity Attack", 900)
    PEEL_CHAIN = ("peel_chain", "Peel Chain", 700)
    DUSTING = ("dusting", "Dusting Attack", 500)
    ADDRESS_POISONING = ("address_poisoning", "Address Poisoning", 400)
    APPROVAL_EXPLOIT = ("approval_exploit", "Approval Exploit", 300)
    SIM_SWAP = ("sim_swap", "SIM Swap Related", 200)
    ROMANCE_SCAM = ("romance_scam", "Romance Scam", 200)
    
    def __init__(self, code: str, display_name: str, market_impact_m: int):
        self.code = code
        self.display_name = display_name
        self.market_impact_m = market_impact_m  # In millions USD


# ═══════════════════════════════════════════════════════════════════════════════
#                              DETECTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypologyDetection:
    """Result of a single typology detection."""
    typology: FraudTypology
    confidence: float  # 0.0 - 1.0
    triggered_rules: List[str]
    evidence: Dict[str, Any]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "typology": self.typology.code,
            "typology_name": self.typology.display_name,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "triggered_rules": self.triggered_rules,
            "evidence": self.evidence,
            "market_impact_reference": f"${self.typology.market_impact_m}M+ market"
        }


@dataclass
class TypologyAnalysisResult:
    """Complete result of 15-typology analysis with GNN fusion."""
    detected_typologies: List[TypologyDetection]
    primary_typology: Optional[TypologyDetection]
    aggregate_risk_score: float  # 0-100
    analysis_time_ms: float
    transaction_id: str
    # GNN Fusion Results (NEW)
    gnn_enabled: bool = False
    gnn_predicted_typology: str = "Unknown"
    gnn_typology_confidence: float = 0.0
    gnn_has_graph_context: bool = False
    gnn_neighbor_count: int = 0
    gnn_boost_applied: bool = False  # True if GNN boosted/modified detection
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "aggregate_risk_score": round(self.aggregate_risk_score, 2),
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "primary_typology": self.primary_typology.to_dict() if self.primary_typology else None,
            "all_detected_typologies": [t.to_dict() for t in self.detected_typologies],
            "typology_count": len(self.detected_typologies),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              15-TYPOLOGY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class FraudTypologyDetector:
    """
    Expert-level fraud detection with 15 distinct typology detectors.
    NOW WITH GNN FUSION - GraphSAGE enhances pattern detection.
    
    Each detector implements:
    - Rule-based pattern matching
    - Statistical anomaly detection
    - Graph analysis via GNN embeddings (NEW)
    - Confidence scoring with GNN boost
    
    Target: 95%+ aggregate detection rate, <2% false positives
    
    GNN FUSION Architecture:
    - GraphSAGE provides 15-class typology prediction
    - GNN confidence boosts matching rule-based detections
    - Novel typologies from GNN are added to detection list
    - Aggregate score weighted by GNN contribution
    """
    
    # ═══ AML THRESHOLDS ═══
    AML_REPORTING_THRESHOLDS = {
        "US": 10000,    # Bank Secrecy Act
        "EU": 10000,    # MiCA
        "FATF": 10000,  # Travel Rule
    }
    
    # ═══ KNOWN BAD ENTITIES ═══
    KNOWN_MIXERS = {
        "0x722122df12d4e14e13ac3b6895a86e84145b6967",
        "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
        "0x23773e65ed146a459791799d01336db287f25334",
        "0xa160cdab225685da1d56aa342ad8841c3b53f291",
    }
    
    KNOWN_FLASH_LOAN_PROVIDERS = {
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # Aave V2
        "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",  # Aave V3
        "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",  # Uniswap V2 Factory
    }
    
    STRUCTURING_AMOUNTS = [3000, 5000, 9999, 9998, 9997, 9500, 14999, 49999]
    
    # GNN Typology Index to FraudTypology Mapping
    GNN_TYPOLOGY_MAP = {
        0: FraudTypology.RUG_PULL,
        1: FraudTypology.PIG_BUTCHERING,
        2: FraudTypology.MIXER_TUMBLING,
        3: FraudTypology.CHAIN_OBFUSCATION,
        4: FraudTypology.FAKE_TOKEN,
        5: FraudTypology.FLASH_LOAN,
        6: FraudTypology.WASH_TRADING,
        7: FraudTypology.STRUCTURING,
        8: FraudTypology.VELOCITY_ATTACK,
        9: FraudTypology.PEEL_CHAIN,
        10: FraudTypology.DUSTING,
        11: FraudTypology.ADDRESS_POISONING,
        12: FraudTypology.APPROVAL_EXPLOIT,
        13: FraudTypology.SIM_SWAP,
        14: FraudTypology.ROMANCE_SCAM,
    }
    
    def __init__(self):
        self.transaction_history: Dict[str, List[Dict]] = defaultdict(list)
        self.address_graph: Dict[str, Set[str]] = defaultdict(set)
        self.risk_scores: Dict[str, float] = {}
        
        # GNN Engine for fusion (NEW)
        self.gnn_engine = None
        self.gnn_enabled = False
        self._initialize_gnn()
    
    def _initialize_gnn(self):
        """Initialize GNN Engine for enhanced typology detection."""
        if not GNN_AVAILABLE:
            return
        
        try:
            self.gnn_engine = get_gnn_engine()
            self.gnn_enabled = self.gnn_engine.enabled
        except Exception as e:
            self.gnn_enabled = False
    
    def _get_gnn_prediction(
        self,
        tx_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: int,
        features: np.ndarray
    ):
        """Get GNN typology prediction."""
        if not self.gnn_enabled or self.gnn_engine is None:
            return None
        
        try:
            # Use first 13 features for GNN (matching LocalNeuralNetwork)
            standard_features = features[:13] if len(features) >= 13 else np.pad(features, (0, 13 - len(features)))
            
            return self.gnn_engine.predict(
                tx_id=tx_id,
                sender=sender,
                recipient=recipient,
                amount=amount,
                timestamp=float(timestamp),
                standard_features=standard_features
            )
        except Exception:
            return None
        
    def analyze_all_typologies(
        self,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: int,
        features: np.ndarray,  # 34-feature vector
        metadata: Dict[str, Any] = None
    ) -> TypologyAnalysisResult:
        """
        Run all 15 fraud typology detectors on a transaction.
        Returns aggregated analysis result.
        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Generate transaction ID
        tx_id = hashlib.sha256(
            f"{sender}{recipient}{amount}{timestamp}".encode()
        ).hexdigest()[:16]
        
        # Store transaction for pattern analysis
        self._store_transaction(sender, recipient, amount, timestamp)
        
        # Run all 15 detectors
        detections: List[TypologyDetection] = []
        
        # 1. Rug Pull Detection ($8B market)
        rug_pull = self._detect_rug_pull(sender, recipient, amount, features, metadata)
        if rug_pull.confidence > 0.3:
            detections.append(rug_pull)
        
        # 2. Pig Butchering Detection ($7.5B market)
        pig_butchering = self._detect_pig_butchering(sender, recipient, amount, features, metadata)
        if pig_butchering.confidence > 0.3:
            detections.append(pig_butchering)
        
        # 3. Mixer/Tumbling Detection ($5.6B market)
        mixer = self._detect_mixer_tumbling(sender, recipient, amount, features)
        if mixer.confidence > 0.3:
            detections.append(mixer)
        
        # 4. Chain Obfuscation Detection ($4.3B market)
        chain_obf = self._detect_chain_obfuscation(sender, recipient, amount, features)
        if chain_obf.confidence > 0.3:
            detections.append(chain_obf)
        
        # 5. Fake Token Detection ($2.8B market)
        fake_token = self._detect_fake_token(sender, recipient, amount, features, metadata)
        if fake_token.confidence > 0.3:
            detections.append(fake_token)
        
        # 6. Flash Loan Attack Detection ($1.9B market)
        flash_loan = self._detect_flash_loan(sender, recipient, amount, features, metadata)
        if flash_loan.confidence > 0.3:
            detections.append(flash_loan)
        
        # 7. Wash Trading Detection ($1.5B market)
        wash_trade = self._detect_wash_trading(sender, recipient, amount, features)
        if wash_trade.confidence > 0.3:
            detections.append(wash_trade)
        
        # 8. Structuring/Smurfing Detection ($1.2B market)
        structuring = self._detect_structuring(sender, recipient, amount, features)
        if structuring.confidence > 0.3:
            detections.append(structuring)
        
        # 9. Velocity Attack Detection ($0.9B market)
        velocity = self._detect_velocity_attack(sender, recipient, amount, timestamp, features)
        if velocity.confidence > 0.3:
            detections.append(velocity)
        
        # 10. Peel Chain Detection ($0.7B market)
        peel_chain = self._detect_peel_chain(sender, recipient, amount, features)
        if peel_chain.confidence > 0.3:
            detections.append(peel_chain)
        
        # 11. Dusting Attack Detection ($0.5B market)
        dusting = self._detect_dusting(sender, recipient, amount, features)
        if dusting.confidence > 0.3:
            detections.append(dusting)
        
        # 12. Address Poisoning Detection ($0.4B market)
        poisoning = self._detect_address_poisoning(sender, recipient, amount, features, metadata)
        if poisoning.confidence > 0.3:
            detections.append(poisoning)
        
        # 13. Approval Exploit Detection ($0.3B market)
        approval = self._detect_approval_exploit(sender, recipient, amount, features, metadata)
        if approval.confidence > 0.3:
            detections.append(approval)
        
        # 14. SIM Swap Related Detection ($0.2B market)
        sim_swap = self._detect_sim_swap_related(sender, recipient, amount, features, metadata)
        if sim_swap.confidence > 0.3:
            detections.append(sim_swap)
        
        # 15. Romance Scam Pattern Detection ($0.2B market)
        romance = self._detect_romance_scam_pattern(sender, recipient, amount, features, metadata)
        if romance.confidence > 0.3:
            detections.append(romance)
        
        # ═══ GNN FUSION (NEW) ═══
        # Get GNN typology prediction to boost/add detections
        gnn_pred = self._get_gnn_prediction(tx_id, sender, recipient, amount, timestamp, features)
        
        gnn_enabled = gnn_pred is not None
        gnn_predicted_typology = "Unknown"
        gnn_typology_confidence = 0.0
        gnn_has_graph_context = False
        gnn_neighbor_count = 0
        gnn_boost_applied = False
        
        if gnn_pred is not None:
            gnn_predicted_typology = gnn_pred.typology_name
            gnn_typology_confidence = gnn_pred.typology_confidence
            gnn_has_graph_context = gnn_pred.has_graph_context
            gnn_neighbor_count = gnn_pred.num_neighbors
            
            # Apply GNN boost if confidence is high
            if gnn_typology_confidence > 0.5 and gnn_pred.primary_typology < 15:
                gnn_typology = self.GNN_TYPOLOGY_MAP.get(gnn_pred.primary_typology)
                
                if gnn_typology:
                    # Check if this typology was already detected
                    existing_detection = next(
                        (d for d in detections if d.typology == gnn_typology), None
                    )
                    
                    if existing_detection:
                        # Boost confidence of existing detection
                        boost_factor = 1.0 + (gnn_typology_confidence * 0.3)  # Up to 30% boost
                        existing_detection.confidence = min(1.0, existing_detection.confidence * boost_factor)
                        existing_detection.triggered_rules.append(
                            f"GNN_BOOST: GraphSAGE confirmed pattern (conf: {gnn_typology_confidence:.2f})"
                        )
                        existing_detection.evidence["gnn_confidence"] = gnn_typology_confidence
                        existing_detection.evidence["gnn_neighbors"] = gnn_neighbor_count
                        gnn_boost_applied = True
                    elif gnn_typology_confidence > 0.65:
                        # Add new detection from GNN (high confidence only)
                        gnn_detection = TypologyDetection(
                            typology=gnn_typology,
                            confidence=gnn_typology_confidence * 0.8,  # Slightly discount pure GNN
                            triggered_rules=[
                                f"GNN_DETECTION: GraphSAGE identified pattern from transaction graph",
                                f"Graph context: {gnn_neighbor_count} neighbor transactions analyzed"
                            ],
                            evidence={
                                "gnn_confidence": gnn_typology_confidence,
                                "gnn_neighbors": gnn_neighbor_count,
                                "detection_source": "graph_neural_network"
                            },
                            severity=self._confidence_to_severity(gnn_typology_confidence * 0.8)
                        )
                        detections.append(gnn_detection)
                        gnn_boost_applied = True
        
        # Calculate aggregate risk score (with GNN contribution)
        aggregate_risk = self._calculate_aggregate_risk(detections, features)
        
        # Boost aggregate risk if GNN has high confidence
        if gnn_enabled and gnn_has_graph_context and gnn_typology_confidence > 0.6:
            gnn_risk_boost = gnn_typology_confidence * 10  # Up to 10 points
            aggregate_risk = min(100, aggregate_risk + gnn_risk_boost)
        
        # Determine primary typology (highest confidence + market impact)
        primary = None
        if detections:
            detections.sort(key=lambda x: x.confidence * x.typology.market_impact_m, reverse=True)
            primary = detections[0]
        
        analysis_time = (time.time() - start_time) * 1000
        
        return TypologyAnalysisResult(
            detected_typologies=detections,
            primary_typology=primary,
            aggregate_risk_score=aggregate_risk,
            analysis_time_ms=analysis_time,
            transaction_id=tx_id,
            # GNN Fusion Results
            gnn_enabled=gnn_enabled,
            gnn_predicted_typology=gnn_predicted_typology,
            gnn_typology_confidence=gnn_typology_confidence,
            gnn_has_graph_context=gnn_has_graph_context,
            gnn_neighbor_count=gnn_neighbor_count,
            gnn_boost_applied=gnn_boost_applied,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                              INDIVIDUAL DETECTORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_rug_pull(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Rug Pull patterns ($8B market).
        
        Signals:
        - Large outflow from token contract/liquidity pool
        - New token with sudden large transfers
        - All funds moving to single address
        - Dev wallet draining liquidity
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Feature indices
        account_age = features[7]  # account_age_days
        total_transfers = features[8]  # total_transfers
        rug_pull_likelihood = features[32]  # rug_pull_likelihood
        
        # Rule 1: High rug pull likelihood from feature engine
        if rug_pull_likelihood > 60:
            confidence += 0.35
            rules.append("High rug pull likelihood score")
            evidence["rug_likelihood"] = float(rug_pull_likelihood)
        
        # Rule 2: New address with large transfer
        if account_age < 7 and amount > 50000:
            confidence += 0.25
            rules.append("New address large withdrawal")
            evidence["account_age_days"] = int(account_age)
        
        # Rule 3: Draining pattern (large % of known holdings)
        sender_history = self.transaction_history.get(sender.lower(), [])
        if sender_history:
            total_volume = sum(tx["amount"] for tx in sender_history)
            if amount > total_volume * 0.5:
                confidence += 0.20
                rules.append("Draining >50% of known volume")
                evidence["drain_percentage"] = round(amount / total_volume * 100, 2)
        
        # Rule 4: Token contract flag
        if metadata.get("is_token_contract"):
            confidence += 0.15
            rules.append("Token contract interaction")
        
        # Rule 5: Liquidity removal
        if metadata.get("is_liquidity_removal"):
            confidence += 0.30
            rules.append("Liquidity removal detected")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.RUG_PULL,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_pig_butchering(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Pig Butchering scam patterns ($7.5B market).
        
        Signals:
        - Escalating transaction amounts over time
        - Regular small "test" transactions followed by large ones
        - Pattern of trust-building → large withdrawal
        - Recipient is known scam consolidator
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        sender_history = self.transaction_history.get(sender.lower(), [])
        
        # Rule 1: Escalating amounts pattern
        if len(sender_history) >= 3:
            amounts = [tx["amount"] for tx in sender_history[-10:]]
            if self._is_escalating_pattern(amounts):
                confidence += 0.35
                rules.append("Escalating transaction pattern")
                evidence["amount_progression"] = amounts[-5:]
        
        # Rule 2: Test transaction followed by large transfer
        if len(sender_history) >= 2:
            last_amount = sender_history[-1]["amount"] if sender_history else 0
            if last_amount < 100 and amount > 10000:
                confidence += 0.30
                rules.append("Test transaction → large transfer")
                evidence["test_amount"] = last_amount
                evidence["main_amount"] = amount
        
        # Rule 3: Regular timing pattern (suggests ongoing relationship)
        timing_consistency = features[20]  # transaction_timing_consistency
        if timing_consistency < 1.0 and len(sender_history) > 5:
            confidence += 0.20
            rules.append("Consistent timing pattern (trust-building)")
        
        # Rule 4: Known scammer association
        scammer_flag = features[25]  # known_scammer_association
        if scammer_flag > 0.5:
            confidence += 0.25
            rules.append("Recipient linked to scammer network")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.PIG_BUTCHERING,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_mixer_tumbling(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Mixer/Tumbling service usage ($5.6B market).
        
        Signals:
        - Direct mixer contract interaction
        - Equal-sized deposit/withdrawal pattern
        - Time-delayed matched transactions
        - Known mixer address
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Feature check
        mixer_interaction = features[12]  # mixer_service_interaction
        
        # Rule 1: Direct mixer interaction
        if mixer_interaction > 0.5:
            confidence += 0.60
            rules.append("Direct mixer contract interaction")
            evidence["mixer_flag"] = True
        
        # Rule 2: Known mixer address
        if recipient.lower() in self.KNOWN_MIXERS or sender.lower() in self.KNOWN_MIXERS:
            confidence += 0.35
            rules.append("Known mixer address")
            evidence["known_mixer"] = True
        
        # Rule 3: Equal-sized "coinjoin" pattern
        if self._is_common_mixer_amount(amount):
            confidence += 0.20
            rules.append("Common mixer denomination")
            evidence["standard_amount"] = True
        
        # Rule 4: Time entropy suggests programmatic behavior
        time_entropy = features[15]  # time_entropy
        if time_entropy > 4.0:  # High entropy = randomized timing
            confidence += 0.15
            rules.append("Randomized transaction timing")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.MIXER_TUMBLING,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_chain_obfuscation(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Chain Obfuscation patterns ($4.3B market).
        
        Signals:
        - Cross-chain bridge usage
        - Multi-hop through DEXes
        - Token swaps to obscure trail
        - Chain-hopping pattern
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Feature checks
        cross_chain = features[24]  # cross_chain_movement
        flow_type = features[23]  # flow_destination_type
        
        # Rule 1: Cross-chain movement
        if cross_chain > 0.5:
            confidence += 0.40
            rules.append("Cross-chain bridge transaction")
            evidence["cross_chain"] = True
        
        # Rule 2: Multiple hops in short time
        sender_history = self.transaction_history.get(sender.lower(), [])
        recent_txs = [tx for tx in sender_history if tx.get("timestamp", 0) > time.time() - 3600]
        if len(recent_txs) > 5:
            unique_recipients = len(set(tx.get("recipient", "") for tx in recent_txs))
            if unique_recipients > 3:
                confidence += 0.30
                rules.append("Multi-hop obfuscation pattern")
                evidence["hops_per_hour"] = len(recent_txs)
        
        # Rule 3: High transaction pattern complexity
        pattern_type = features[17]  # transaction_pattern_type numeric
        if pattern_type > 0.6:  # Branching or star pattern
            confidence += 0.20
            rules.append("Complex transaction pattern")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.CHAIN_OBFUSCATION,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_fake_token(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Fake Token scams ($2.8B market).
        
        Signals:
        - New token with misleading name
        - Honeypot contract (can buy, can't sell)
        - Concentrated token holdings
        - Zero liquidity or locked selling
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Metadata checks (would come from token analysis service)
        token_age_days = metadata.get("token_age_days", 365)
        is_honeypot = metadata.get("is_honeypot", False)
        holder_concentration = metadata.get("holder_concentration", 0)
        
        # Rule 1: Honeypot detection
        if is_honeypot:
            confidence += 0.50
            rules.append("Honeypot contract detected")
            evidence["honeypot"] = True
        
        # Rule 2: Very new token
        if token_age_days < 7:
            confidence += 0.25
            rules.append("Token less than 7 days old")
            evidence["token_age_days"] = token_age_days
        
        # Rule 3: High holder concentration (whale risk)
        if holder_concentration > 0.8:  # 80%+ held by few wallets
            confidence += 0.30
            rules.append("High holder concentration")
            evidence["top_holder_percentage"] = holder_concentration
        
        # Rule 4: Contract interaction on new account
        account_age = features[7]
        contract_flag = features[4]  # smart_contract_interaction
        if account_age < 30 and contract_flag > 0.5:
            confidence += 0.15
            rules.append("New account interacting with contract")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.FAKE_TOKEN,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_flash_loan(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Flash Loan Attack patterns ($1.9B market).
        
        Signals:
        - Interaction with flash loan providers
        - Large uncollateralized borrow
        - Same-block repayment
        - Oracle manipulation pattern
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Rule 1: Known flash loan provider
        if recipient.lower() in self.KNOWN_FLASH_LOAN_PROVIDERS:
            confidence += 0.40
            rules.append("Flash loan provider interaction")
            evidence["flash_loan_provider"] = True
        
        # Rule 2: Large amount with zero collateral
        if amount > 100000 and metadata.get("is_uncollateralized", False):
            confidence += 0.35
            rules.append("Large uncollateralized transaction")
        
        # Rule 3: Same-block complexity
        if metadata.get("same_block_tx_count", 0) > 3:
            confidence += 0.25
            rules.append("Multiple same-block transactions")
            evidence["same_block_txs"] = metadata.get("same_block_tx_count")
        
        # Rule 4: DEX arbitrage pattern
        if metadata.get("involves_multiple_dexes", False):
            confidence += 0.20
            rules.append("Multi-DEX interaction pattern")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.FLASH_LOAN,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_wash_trading(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Wash Trading patterns ($1.5B market).
        
        Signals:
        - Circular transaction flow (A→B→C→A)
        - Self-dealing through intermediaries
        - Volume inflation pattern
        - Same amounts returning
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        wash_trading_score = features[33]  # wash_trading_score
        
        # Rule 1: Feature engine detection
        if wash_trading_score > 50:
            confidence += 0.40
            rules.append("Circular flow detected")
            evidence["wash_score"] = float(wash_trading_score)
        
        # Rule 2: Check for A→B→A pattern
        if recipient.lower() in self.address_graph.get(sender.lower(), set()):
            if sender.lower() in self.address_graph.get(recipient.lower(), set()):
                confidence += 0.35
                rules.append("Direct round-trip detected")
                evidence["round_trip"] = True
        
        # Rule 3: Same amount returning
        sender_history = self.transaction_history.get(sender.lower(), [])
        received = [tx for tx in sender_history if tx.get("direction") == "in"]
        for tx in received:
            if abs(tx.get("amount", 0) - amount) < amount * 0.01:  # Within 1%
                confidence += 0.25
                rules.append("Matching amount return")
                break
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.WASH_TRADING,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_structuring(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Structuring/Smurfing patterns ($1.2B market).
        
        AML Compliance Target: 99% detection rate
        
        Signals:
        - Amounts just below reporting thresholds
        - Multiple transactions to stay under limits
        - Pattern of $9,999 or similar
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Rule 1: Just below threshold amounts
        for threshold in [10000, 15000, 50000]:
            lower_bound = threshold * 0.94
            if lower_bound <= amount < threshold:
                confidence += 0.45
                rules.append(f"Amount {amount:.2f} just below ${threshold} threshold")
                evidence["near_threshold"] = threshold
                evidence["gap"] = round(threshold - amount, 2)
                break
        
        # Rule 2: Pattern amounts
        if any(abs(amount - sa) < 5 for sa in self.STRUCTURING_AMOUNTS):
            confidence += 0.35
            rules.append("Known structuring amount pattern")
        
        # Rule 3: Multiple near-threshold transactions
        sender_history = self.transaction_history.get(sender.lower(), [])
        recent_24h = [tx for tx in sender_history if tx.get("timestamp", 0) > time.time() - 86400]
        structuring_count = sum(1 for tx in recent_24h if 9000 <= tx.get("amount", 0) < 10000)
        
        if structuring_count >= 2:
            confidence += 0.30
            rules.append(f"Multiple structuring transactions in 24h: {structuring_count}")
            evidence["structuring_count_24h"] = structuring_count
        
        # Rule 4: Cumulative 24h total exceeds threshold
        total_24h = sum(tx.get("amount", 0) for tx in recent_24h) + amount
        if 10000 <= total_24h < 50000 and structuring_count >= 2:
            confidence += 0.20
            rules.append("Cumulative 24h total suggests threshold avoidance")
            evidence["cumulative_24h"] = round(total_24h, 2)
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.STRUCTURING,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_velocity_attack(
        self, sender: str, recipient: str, amount: float,
        timestamp: int, features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Velocity Attack patterns ($0.9B market).
        
        Signals:
        - Abnormally high transaction frequency
        - Rapid-fire transactions
        - Account draining pattern
        - Automated withdrawal behavior
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        tx_count_24h = features[5]  # transaction_count_24h
        tx_frequency = features[6]  # tx_frequency_per_day
        
        # Rule 1: Very high 24h transaction count
        if tx_count_24h > 50:
            confidence += 0.40
            rules.append(f"High velocity: {int(tx_count_24h)} transactions in 24h")
            evidence["tx_count_24h"] = int(tx_count_24h)
        
        # Rule 2: Transaction frequency anomaly
        if tx_frequency > 100:
            confidence += 0.30
            rules.append("Abnormal transaction frequency")
            evidence["tx_frequency"] = round(tx_frequency, 2)
        
        # Rule 3: Recent burst detection
        sender_history = self.transaction_history.get(sender.lower(), [])
        recent_1h = [tx for tx in sender_history if tx.get("timestamp", 0) > timestamp - 3600]
        if len(recent_1h) > 20:
            confidence += 0.35
            rules.append(f"Burst: {len(recent_1h)} transactions in last hour")
            evidence["burst_1h"] = len(recent_1h)
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.VELOCITY_ATTACK,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_peel_chain(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Peel Chain patterns ($0.7B market).
        
        Signals:
        - Systematic amount reduction
        - Linear transaction chain
        - Each hop reduces amount by peeling off
        - Common in ransomware cash-out
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        peel_indicator = features[31]  # peel_chain_indicator
        pattern_type = features[17]  # transaction_pattern_type
        
        # Rule 1: Feature engine detection
        if peel_indicator > 0.5:
            confidence += 0.40
            rules.append("Systematic amount reduction detected")
            evidence["peel_indicator"] = float(peel_indicator)
        
        # Rule 2: Linear pattern
        if pattern_type < 0.4:  # Linear or normal (not branching)
            sender_history = self.transaction_history.get(sender.lower(), [])
            if len(sender_history) >= 3:
                amounts = [tx["amount"] for tx in sender_history[-5:]]
                if self._is_decreasing_pattern(amounts):
                    confidence += 0.35
                    rules.append("Decreasing amounts in chain")
                    evidence["amount_pattern"] = amounts
        
        # Rule 3: Similar % reduction at each step
        sender_history = self.transaction_history.get(sender.lower(), [])
        if len(sender_history) >= 2:
            ratios = []
            for i in range(1, min(5, len(sender_history))):
                prev = sender_history[-(i+1)]["amount"]
                curr = sender_history[-i]["amount"]
                if prev > 0:
                    ratios.append(curr / prev)
            
            if ratios and max(ratios) - min(ratios) < 0.1:  # Consistent reduction
                confidence += 0.25
                rules.append("Consistent peeling ratio")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.PEEL_CHAIN,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_dusting(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray
    ) -> TypologyDetection:
        """
        Detect Dusting Attack patterns ($0.5B market).
        
        Signals:
        - Tiny amounts (< $1) sent to many addresses
        - Used for wallet tracing
        - Unsolicited tiny deposits
        - Often precedes targeted attacks
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        is_dust = features[30]  # is_dust_attack
        
        # Rule 1: Dust amount
        if amount < 1.0:
            confidence += 0.30
            rules.append(f"Dust amount: ${amount:.4f}")
            evidence["dust_amount"] = amount
        
        # Rule 2: Feature engine dust detection
        if is_dust > 0.5:
            confidence += 0.35
            rules.append("Pattern of dust transactions")
            evidence["dust_pattern"] = True
        
        # Rule 3: Many recipients from same sender
        sender_lower = sender.lower()
        recipients = self.address_graph.get(sender_lower, set())
        if len(recipients) > 50:
            confidence += 0.30
            rules.append(f"Sending to {len(recipients)} unique addresses")
            evidence["recipient_count"] = len(recipients)
        
        # Rule 4: All dust amounts
        sender_history = self.transaction_history.get(sender_lower, [])
        dust_ratio = sum(1 for tx in sender_history if tx.get("amount", 0) < 1) / max(1, len(sender_history))
        if dust_ratio > 0.9:
            confidence += 0.25
            rules.append("90%+ transactions are dust")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.DUSTING,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_address_poisoning(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Address Poisoning attacks ($0.4B market).
        
        Signals:
        - Similar address to known recipient
        - Zero-value or dust transaction
        - Appears in transaction history to confuse user
        - First 4 and last 4 chars match legitimate address
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Rule 1: Dust or zero value
        if amount < 0.01:
            confidence += 0.25
            rules.append("Zero/dust value transaction")
        
        # Rule 2: Similar address detection (would check against user's known addresses)
        known_addresses = metadata.get("user_known_addresses", [])
        for known in known_addresses:
            if self._is_address_similar(recipient, known):
                confidence += 0.50
                rules.append("Recipient similar to known address")
                evidence["similar_to"] = known[:10] + "..."
                evidence["poisoned"] = recipient[:10] + "..."
                break
        
        # Rule 3: New sender we haven't seen before
        if sender.lower() not in self.transaction_history:
            confidence += 0.20
            rules.append("First transaction from this sender")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.ADDRESS_POISONING,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_approval_exploit(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Approval Exploit patterns ($0.3B market).
        
        Signals:
        - Unlimited approval usage
        - Draining after long dormant period
        - Unknown contract using approval
        - Multiple approvals being exploited
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Rule 1: Using old approval
        approval_age_days = metadata.get("approval_age_days", 0)
        if approval_age_days > 90 and amount > 1000:
            confidence += 0.40
            rules.append(f"Using {approval_age_days}-day old approval")
            evidence["approval_age"] = approval_age_days
        
        # Rule 2: Unlimited approval
        if metadata.get("is_unlimited_approval", False):
            confidence += 0.30
            rules.append("Unlimited token approval")
        
        # Rule 3: Unknown contract draining
        contract_flag = features[4]  # smart_contract_interaction
        if contract_flag > 0.5 and metadata.get("contract_verified", True) is False:
            confidence += 0.35
            rules.append("Unverified contract interaction")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.APPROVAL_EXPLOIT,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_sim_swap_related(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect SIM Swap related patterns ($0.2B market).
        
        Signals:
        - Account behavior change
        - Sudden withdrawal after dormancy
        - New device/location indicators
        - 2FA bypass patterns
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        # Rule 1: Dormant account suddenly active
        account_age = features[7]  # account_age_days
        recent_activity = features[5]  # transaction_count_24h
        
        sender_history = self.transaction_history.get(sender.lower(), [])
        if len(sender_history) >= 1 and account_age > 30:
            days_since_last = (time.time() - sender_history[-1].get("timestamp", 0)) / 86400
            if days_since_last > 30 and amount > 5000:
                confidence += 0.40
                rules.append(f"Account dormant {int(days_since_last)} days, now large transfer")
                evidence["dormancy_days"] = int(days_since_last)
        
        # Rule 2: Behavior change (sudden high velocity)
        if recent_activity > 10 and len(sender_history) < 5:
            confidence += 0.30
            rules.append("Sudden activity spike on low-history account")
        
        # Rule 3: New device indicator (from metadata)
        if metadata.get("new_device", False):
            confidence += 0.25
            rules.append("New device detected")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.SIM_SWAP,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    def _detect_romance_scam_pattern(
        self, sender: str, recipient: str, amount: float,
        features: np.ndarray, metadata: Dict
    ) -> TypologyDetection:
        """
        Detect Romance Scam patterns ($0.2B market).
        
        Signals:
        - Escalating amounts (trust building)
        - Regular timing (relationship pattern)
        - Single recipient focus
        - Increasing frequency
        """
        confidence = 0.0
        rules = []
        evidence = {}
        
        sender_history = self.transaction_history.get(sender.lower(), [])
        
        # Rule 1: Single recipient focus
        if sender_history:
            recipient_counts = {}
            for tx in sender_history:
                r = tx.get("recipient", "")
                recipient_counts[r] = recipient_counts.get(r, 0) + 1
            
            if len(recipient_counts) == 1 and len(sender_history) > 5:
                confidence += 0.35
                rules.append("Single recipient focus over multiple transactions")
                evidence["transaction_count"] = len(sender_history)
        
        # Rule 2: Escalating pattern
        if len(sender_history) >= 3:
            amounts = [tx["amount"] for tx in sender_history]
            if self._is_escalating_pattern(amounts):
                confidence += 0.30
                rules.append("Trust-building escalation pattern")
        
        # Rule 3: Regular timing
        timing_consistency = features[20]  # transaction_timing_consistency
        if timing_consistency < 1.0:
            confidence += 0.20
            rules.append("Regular transaction timing")
        
        # Rule 4: Known romance scam destination
        if metadata.get("is_known_romance_scam_address", False):
            confidence += 0.40
            rules.append("Known romance scam address")
        
        severity = self._confidence_to_severity(confidence)
        
        return TypologyDetection(
            typology=FraudTypology.ROMANCE_SCAM,
            confidence=min(1.0, confidence),
            triggered_rules=rules,
            evidence=evidence,
            severity=severity,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                              HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _store_transaction(
        self, sender: str, recipient: str, amount: float, timestamp: int
    ):
        """Store transaction for pattern analysis."""
        sender_lower = sender.lower()
        recipient_lower = recipient.lower()
        
        self.transaction_history[sender_lower].append({
            "recipient": recipient_lower,
            "amount": amount,
            "timestamp": timestamp,
            "direction": "out",
        })
        
        self.transaction_history[recipient_lower].append({
            "sender": sender_lower,
            "amount": amount,
            "timestamp": timestamp,
            "direction": "in",
        })
        
        self.address_graph[sender_lower].add(recipient_lower)
    
    def _is_escalating_pattern(self, amounts: List[float]) -> bool:
        """Check if amounts show escalating pattern."""
        if len(amounts) < 3:
            return False
        
        increases = sum(1 for i in range(len(amounts)-1) if amounts[i+1] > amounts[i] * 1.1)
        return increases >= len(amounts) * 0.6
    
    def _is_decreasing_pattern(self, amounts: List[float]) -> bool:
        """Check if amounts show decreasing pattern (peel chain)."""
        if len(amounts) < 3:
            return False
        
        decreases = sum(1 for i in range(len(amounts)-1) if amounts[i+1] < amounts[i])
        return decreases >= len(amounts) * 0.7
    
    def _is_common_mixer_amount(self, amount: float) -> bool:
        """Check if amount matches common mixer denominations."""
        common = [0.1, 1.0, 10.0, 100.0, 0.01, 0.001]
        return any(abs(amount - c) < 0.001 for c in common)
    
    def _is_address_similar(self, addr1: str, addr2: str) -> bool:
        """Check if addresses are similar (poisoning pattern)."""
        a1 = addr1.lower()
        a2 = addr2.lower()
        
        if a1 == a2:
            return False
        
        # Check first 4 and last 4 characters match
        return (a1[:6] == a2[:6] and a1[-4:] == a2[-4:]) or \
               (a1[:4] == a2[:4] and a1[-6:] == a2[-6:])
    
    def _confidence_to_severity(self, confidence: float) -> str:
        """Convert confidence to severity level."""
        if confidence >= 0.75:
            return "CRITICAL"
        elif confidence >= 0.50:
            return "HIGH"
        elif confidence >= 0.35:
            return "MEDIUM"
        return "LOW"
    
    def _calculate_aggregate_risk(
        self, 
        detections: List[TypologyDetection],
        features: np.ndarray
    ) -> float:
        """Calculate aggregate risk score from all detections."""
        if not detections:
            # Base risk from features
            mixer_flag = features[12]
            bad_address = features[14]
            scammer_assoc = features[25]
            
            base_risk = (mixer_flag * 40 + bad_address * 50 + scammer_assoc * 30) / 3
            return min(30.0, base_risk)
        
        # Weighted by market impact and confidence
        total_weighted = 0
        total_weight = 0
        
        for d in detections:
            weight = math.log10(d.typology.market_impact_m + 1)  # Log of market impact
            total_weighted += d.confidence * 100 * weight
            total_weight += weight
        
        base_score = total_weighted / total_weight if total_weight > 0 else 0
        
        # Boost for multiple typology detections
        if len(detections) >= 3:
            base_score *= 1.2
        elif len(detections) >= 2:
            base_score *= 1.1
        
        return min(100.0, base_score)


# ═══════════════════════════════════════════════════════════════════════════════
#                              STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("15-TYPOLOGY FRAUD DETECTION ENGINE - TEST")
    print("=" * 80)
    
    detector = FraudTypologyDetector()
    
    # Create mock feature vector (34 features)
    mock_features = np.zeros(34, dtype=np.float32)
    mock_features[0] = 9999.0   # amount
    mock_features[7] = 5        # account_age_days
    mock_features[12] = 0.0     # mixer_service_interaction
    mock_features[32] = 70.0    # rug_pull_likelihood
    mock_features[33] = 40.0    # wash_trading_score
    
    # Test transaction
    result = detector.analyze_all_typologies(
        sender="0xNewAddress1234567890123456789012345678",
        recipient="0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
        amount=9999.0,
        timestamp=int(time.time()),
        features=mock_features,
        metadata={
            "is_token_contract": True,
            "token_age_days": 3,
        }
    )
    
    print(f"\nTransaction ID: {result.transaction_id}")
    print(f"Aggregate Risk Score: {result.aggregate_risk_score:.1f}/100")
    print(f"Analysis Time: {result.analysis_time_ms:.2f}ms")
    print(f"Typologies Detected: {len(result.detected_typologies)}")
    
    if result.primary_typology:
        print(f"\nPrimary Typology: {result.primary_typology.typology.display_name}")
        print(f"Confidence: {result.primary_typology.confidence:.2%}")
        print(f"Severity: {result.primary_typology.severity}")
        print("Triggered Rules:")
        for rule in result.primary_typology.triggered_rules:
            print(f"  • {rule}")
    
    print("\nAll Detected Typologies:")
    for t in result.detected_typologies:
        print(f"  [{t.severity:8}] {t.typology.display_name:25} ({t.confidence:.1%})")
    
    print("\n" + "=" * 80)
