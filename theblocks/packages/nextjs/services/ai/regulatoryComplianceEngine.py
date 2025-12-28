"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW REGULATORY COMPLIANCE ENGINE                              ║
║                                                                                       ║
║   Industry-Grade AML/KYC Compliance for Stablecoin Transactions                      ║
║                                                                                       ║
║   Regulatory Frameworks Supported:                                                    ║
║   ┌─────────────────────────────────────────────────────────────────────────────┐    ║
║   │                                                                             │    ║
║   │  🇺🇸 UNITED STATES                                                          │    ║
║   │     • GENIUS Act (2025) - Stablecoin-specific regulations                  │    ║
║   │     • Bank Secrecy Act (BSA) - $10,000 reporting threshold                 │    ║
║   │     • FinCEN Travel Rule - $3,000 threshold for crypto                     │    ║
║   │     • OFAC Sanctions - SDN list screening                                  │    ║
║   │                                                                             │    ║
║   │  🇪🇺 EUROPEAN UNION                                                          │    ║
║   │     • MiCA (Markets in Crypto-Assets) - Full framework 2024                │    ║
║   │     • AMLD6 - 6th Anti-Money Laundering Directive                          │    ║
║   │     • Travel Rule - €1,000 threshold                                       │    ║
║   │                                                                             │    ║
║   │  🌍 INTERNATIONAL                                                            │    ║
║   │     • FATF Travel Rule - $1,000/€1,000 threshold                           │    ║
║   │     • FATF Recommendations - 40 recommendations                            │    ║
║   │     • Wolfsberg Principles - Correspondent banking                         │    ║
║   │                                                                             │    ║
║   └─────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                       ║
║   Features:                                                                           ║
║   • Real-time sanctions screening (OFAC SDN simulation)                              ║
║   • Automatic SAR/STR generation                                                     ║
║   • Travel Rule compliance check                                                     ║
║   • Jurisdiction detection                                                           ║
║   • Compliance scoring and recommendations                                           ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime, timezone
import re


# ═══════════════════════════════════════════════════════════════════════════════
#                              REGULATORY FRAMEWORKS
# ═══════════════════════════════════════════════════════════════════════════════

class Jurisdiction(Enum):
    """Supported jurisdictions."""
    US = ("US", "United States", "🇺🇸")
    EU = ("EU", "European Union", "🇪🇺")
    UK = ("UK", "United Kingdom", "🇬🇧")
    SG = ("SG", "Singapore", "🇸🇬")
    HK = ("HK", "Hong Kong", "🇭🇰")
    JP = ("JP", "Japan", "🇯🇵")
    GLOBAL = ("GLOBAL", "International", "🌍")
    
    def __init__(self, code: str, name: str, emoji: str):
        self.code_ = code
        self.full_name = name
        self.emoji = emoji


class ComplianceLevel(Enum):
    """Compliance status levels."""
    COMPLIANT = ("compliant", "✅", "Transaction meets all regulatory requirements")
    NEEDS_REVIEW = ("needs_review", "🟡", "Manual review recommended")
    ENHANCED_DUE_DILIGENCE = ("edd", "🟠", "Enhanced due diligence required")
    BLOCKED = ("blocked", "🔴", "Transaction blocked - regulatory violation")
    SANCTIONS_HIT = ("sanctions_hit", "🚨", "Potential sanctions violation - STOP")
    
    def __init__(self, code: str, emoji: str, description: str):
        self.code_ = code
        self.emoji = emoji
        self.description = description


class ReportType(Enum):
    """Types of regulatory reports."""
    SAR = "Suspicious Activity Report"
    STR = "Suspicious Transaction Report"
    CTR = "Currency Transaction Report"
    TRAR = "Travel Rule Advisory Report"


# ═══════════════════════════════════════════════════════════════════════════════
#                              REGULATORY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegulatoryThresholds:
    """Regulatory thresholds by jurisdiction."""
    
    # US Thresholds
    US_CTR_THRESHOLD: float = 10000.0  # Currency Transaction Report
    US_TRAVEL_RULE_THRESHOLD: float = 3000.0  # FinCEN crypto travel rule
    US_STRUCTURING_THRESHOLD: float = 10000.0  # BSA structuring
    
    # EU/MiCA Thresholds
    EU_TRAVEL_RULE_THRESHOLD: float = 1000.0  # €1,000 EUR
    EU_HIGH_VALUE_THRESHOLD: float = 10000.0  # Enhanced due diligence
    
    # FATF Thresholds
    FATF_TRAVEL_RULE_THRESHOLD: float = 1000.0  # $1,000 USD equivalent
    
    # Singapore MAS
    SG_THRESHOLD: float = 5000.0  # SGD equivalent
    
    # General
    HIGH_RISK_AMOUNT: float = 50000.0  # Always flag
    CRITICAL_AMOUNT: float = 100000.0  # Always block pending review


# ═══════════════════════════════════════════════════════════════════════════════
#                              COMPLIANCE RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SanctionsScreenResult:
    """Result of sanctions screening."""
    is_sanctioned: bool
    matched_list: Optional[str]
    matched_name: Optional[str]
    match_score: float
    screening_time_ms: float


@dataclass
class TravelRuleCheck:
    """Travel Rule compliance check result."""
    applies: bool
    threshold: float
    jurisdiction: Jurisdiction
    required_fields: List[str]
    missing_fields: List[str]
    is_compliant: bool


@dataclass
class RegulatoryReport:
    """Generated regulatory report."""
    report_type: ReportType
    report_id: str
    timestamp: str
    transaction_id: str
    jurisdiction: Jurisdiction
    
    # Transaction details
    sender_address: str
    recipient_address: str
    amount: float
    currency: str
    
    # Risk information
    risk_score: float
    risk_indicators: List[str]
    
    # Narrative
    narrative: str
    
    # Recommendation
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_type": self.report_type.value,
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "transaction_id": self.transaction_id,
            "jurisdiction": self.jurisdiction.full_name,
            "transaction": {
                "sender": self.sender_address,
                "recipient": self.recipient_address,
                "amount": self.amount,
                "currency": self.currency,
            },
            "risk": {
                "score": self.risk_score,
                "indicators": self.risk_indicators,
            },
            "narrative": self.narrative,
            "recommended_action": self.recommended_action,
        }


@dataclass
class ComplianceResult:
    """Complete compliance check result."""
    
    # Overall status
    compliance_level: ComplianceLevel
    compliance_score: float  # 0-100 (100 = fully compliant)
    
    # Jurisdiction
    primary_jurisdiction: Jurisdiction
    applicable_regulations: List[str]
    
    # Checks performed
    sanctions_check: SanctionsScreenResult
    travel_rule_check: TravelRuleCheck
    
    # Threshold analysis
    exceeds_ctr_threshold: bool
    structuring_detected: bool
    
    # Reports generated
    reports_required: List[ReportType]
    generated_reports: List[RegulatoryReport]
    
    # Recommendations
    required_actions: List[str]
    recommendations: List[str]
    
    # Timing
    check_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compliance_level": self.compliance_level.code_,
            "compliance_emoji": self.compliance_level.emoji,
            "compliance_score": round(self.compliance_score, 2),
            "description": self.compliance_level.description,
            "primary_jurisdiction": self.primary_jurisdiction.full_name,
            "applicable_regulations": self.applicable_regulations,
            "sanctions_check": {
                "is_sanctioned": self.sanctions_check.is_sanctioned,
                "matched_list": self.sanctions_check.matched_list,
                "match_score": round(self.sanctions_check.match_score, 4),
                "screening_time_ms": round(self.sanctions_check.screening_time_ms, 2),
            },
            "travel_rule": {
                "applies": self.travel_rule_check.applies,
                "threshold": self.travel_rule_check.threshold,
                "is_compliant": self.travel_rule_check.is_compliant,
                "missing_fields": self.travel_rule_check.missing_fields,
            },
            "threshold_analysis": {
                "exceeds_ctr_threshold": self.exceeds_ctr_threshold,
                "structuring_detected": self.structuring_detected,
            },
            "reports_required": [r.value for r in self.reports_required],
            "required_actions": self.required_actions,
            "recommendations": self.recommendations,
            "check_time_ms": round(self.check_time_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              SANCTIONS DATABASE (SIMULATION)
# ═══════════════════════════════════════════════════════════════════════════════

class SanctionsDatabase:
    """
    Simulated sanctions database (OFAC SDN, UN, EU lists).
    
    In production, would integrate with:
    - OFAC SDN API
    - Chainalysis Sanctions API
    - TRM Labs
    - Elliptic
    """
    
    # Simulated sanctioned addresses (Tornado Cash, OFAC-designated)
    SANCTIONED_ADDRESSES = {
        "0x722122df12d4e14e13ac3b6895a86e84145b6967": "OFAC SDN - Tornado Cash",
        "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b": "OFAC SDN - Tornado Cash",
        "0x23773e65ed146a459791799d01336db287f25334": "OFAC SDN - Tornado Cash",
        "0xa160cdab225685da1d56aa342ad8841c3b53f291": "OFAC SDN - Tornado Cash",
        "0x8589427373d6d84e98730d7795d8f6f8731fda16": "OFAC SDN - Lazarus Group Related",
        "0x098b716b8aaf21512996dc57eb0615e2383e2f96": "OFAC SDN - Lazarus Group Related",
        "0x8576acc5c05d6ce88f4e49bf65bdf0c62f91353c": "OFAC SDN - Blender.io Mixer",
        "0xfec8a60023265364d066a1212fde3930f6ae8da7": "OFAC SDN - Sinbad Mixer",
    }
    
    # High-risk jurisdictions (FATF grey/black list simulation)
    HIGH_RISK_JURISDICTIONS = {
        "KP": "North Korea",
        "IR": "Iran",
        "SY": "Syria",
        "CU": "Cuba",
        "MM": "Myanmar",
        "RU": "Russia (partial)",
    }
    
    @classmethod
    def screen_address(cls, address: str) -> Tuple[bool, Optional[str], float]:
        """
        Screen an address against sanctions lists.
        Returns: (is_sanctioned, list_name, match_score)
        """
        addr_lower = address.lower()
        
        # Direct match
        if addr_lower in cls.SANCTIONED_ADDRESSES:
            return True, cls.SANCTIONED_ADDRESSES[addr_lower], 1.0
        
        # Fuzzy match (first/last 4 chars - address poisoning check)
        for sanctioned in cls.SANCTIONED_ADDRESSES:
            if addr_lower[:6] == sanctioned[:6] and addr_lower[-4:] == sanctioned[-4:]:
                return True, f"Fuzzy match: {cls.SANCTIONED_ADDRESSES[sanctioned]}", 0.85
        
        return False, None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#                              COMPLIANCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RegulatoryComplianceEngine:
    """
    Industry-grade regulatory compliance engine for stablecoin transactions.
    
    Features:
    - Multi-jurisdiction support (US, EU, FATF)
    - Real-time sanctions screening
    - Travel Rule compliance
    - Automatic SAR generation
    - Structuring detection
    """
    
    def __init__(self):
        self.thresholds = RegulatoryThresholds()
        self.transaction_history: Dict[str, List[Dict]] = {}
        
    def check_compliance(
        self,
        sender: str,
        recipient: str,
        amount: float,
        currency: str = "USDC",
        risk_score: float = 0.0,
        fraud_typologies: List[str] = None,
        sender_info: Dict[str, Any] = None,
        recipient_info: Dict[str, Any] = None,
        jurisdiction: Jurisdiction = None,
    ) -> ComplianceResult:
        """
        Perform comprehensive compliance check on a transaction.
        """
        start_time = time.time()
        fraud_typologies = fraud_typologies or []
        sender_info = sender_info or {}
        recipient_info = recipient_info or {}
        
        # Determine jurisdiction
        if jurisdiction is None:
            jurisdiction = self._detect_jurisdiction(sender_info, recipient_info)
        
        # 1. Sanctions Screening
        sanctions_result = self._screen_sanctions(sender, recipient)
        
        # 2. Travel Rule Check
        travel_rule_result = self._check_travel_rule(
            amount, jurisdiction, sender_info, recipient_info
        )
        
        # 3. Threshold Analysis
        exceeds_ctr = amount >= self.thresholds.US_CTR_THRESHOLD
        structuring = self._detect_structuring(sender, amount)
        
        # 4. Determine applicable regulations
        regulations = self._get_applicable_regulations(jurisdiction, amount)
        
        # 5. Determine compliance level
        compliance_level, compliance_score = self._calculate_compliance_level(
            sanctions_result,
            travel_rule_result,
            exceeds_ctr,
            structuring,
            risk_score,
            fraud_typologies,
        )
        
        # 6. Determine required reports
        reports_required = self._determine_required_reports(
            compliance_level,
            exceeds_ctr,
            risk_score,
            fraud_typologies,
            jurisdiction,
        )
        
        # 7. Generate reports
        generated_reports = []
        tx_id = hashlib.sha256(f"{sender}{recipient}{amount}{time.time()}".encode()).hexdigest()[:16]
        
        for report_type in reports_required:
            report = self._generate_report(
                report_type, tx_id, jurisdiction,
                sender, recipient, amount, currency,
                risk_score, fraud_typologies,
            )
            generated_reports.append(report)
        
        # 8. Generate recommendations
        required_actions, recommendations = self._generate_recommendations(
            compliance_level,
            sanctions_result,
            travel_rule_result,
            exceeds_ctr,
            structuring,
            risk_score,
        )
        
        check_time = (time.time() - start_time) * 1000
        
        return ComplianceResult(
            compliance_level=compliance_level,
            compliance_score=compliance_score,
            primary_jurisdiction=jurisdiction,
            applicable_regulations=regulations,
            sanctions_check=sanctions_result,
            travel_rule_check=travel_rule_result,
            exceeds_ctr_threshold=exceeds_ctr,
            structuring_detected=structuring,
            reports_required=reports_required,
            generated_reports=generated_reports,
            required_actions=required_actions,
            recommendations=recommendations,
            check_time_ms=check_time,
        )
    
    def _detect_jurisdiction(
        self,
        sender_info: Dict,
        recipient_info: Dict
    ) -> Jurisdiction:
        """Detect primary jurisdiction for the transaction."""
        # Check sender/recipient country
        sender_country = sender_info.get("country", "").upper()
        recipient_country = recipient_info.get("country", "").upper()
        
        # Priority: US > EU > UK > Others
        if sender_country == "US" or recipient_country == "US":
            return Jurisdiction.US
        elif sender_country in ["DE", "FR", "IT", "ES", "NL"] or \
             recipient_country in ["DE", "FR", "IT", "ES", "NL"]:
            return Jurisdiction.EU
        elif sender_country == "GB" or recipient_country == "GB":
            return Jurisdiction.UK
        elif sender_country == "SG" or recipient_country == "SG":
            return Jurisdiction.SG
        
        return Jurisdiction.GLOBAL
    
    def _screen_sanctions(
        self,
        sender: str,
        recipient: str
    ) -> SanctionsScreenResult:
        """Screen both addresses against sanctions lists."""
        start = time.time()
        
        # Screen sender
        sender_sanctioned, sender_list, sender_score = SanctionsDatabase.screen_address(sender)
        
        # Screen recipient
        recipient_sanctioned, recipient_list, recipient_score = SanctionsDatabase.screen_address(recipient)
        
        screening_time = (time.time() - start) * 1000
        
        is_sanctioned = sender_sanctioned or recipient_sanctioned
        matched_list = sender_list or recipient_list
        match_score = max(sender_score, recipient_score)
        
        return SanctionsScreenResult(
            is_sanctioned=is_sanctioned,
            matched_list=matched_list,
            matched_name=None,  # Would come from sanctions API
            match_score=match_score,
            screening_time_ms=screening_time,
        )
    
    def _check_travel_rule(
        self,
        amount: float,
        jurisdiction: Jurisdiction,
        sender_info: Dict,
        recipient_info: Dict
    ) -> TravelRuleCheck:
        """Check Travel Rule compliance."""
        # Determine threshold based on jurisdiction
        if jurisdiction == Jurisdiction.US:
            threshold = self.thresholds.US_TRAVEL_RULE_THRESHOLD
        elif jurisdiction == Jurisdiction.EU:
            threshold = self.thresholds.EU_TRAVEL_RULE_THRESHOLD
        else:
            threshold = self.thresholds.FATF_TRAVEL_RULE_THRESHOLD
        
        applies = amount >= threshold
        
        # Required fields for Travel Rule
        required_fields = [
            "originator_name",
            "originator_address",
            "beneficiary_name",
            "beneficiary_address",
        ]
        
        if jurisdiction == Jurisdiction.US and amount >= self.thresholds.US_CTR_THRESHOLD:
            required_fields.extend(["originator_account", "originator_institution"])
        
        # Check which fields are missing
        missing = []
        for field in required_fields:
            if field.startswith("originator"):
                key = field.replace("originator_", "")
                if key not in sender_info:
                    missing.append(field)
            else:
                key = field.replace("beneficiary_", "")
                if key not in recipient_info:
                    missing.append(field)
        
        is_compliant = len(missing) == 0 or not applies
        
        return TravelRuleCheck(
            applies=applies,
            threshold=threshold,
            jurisdiction=jurisdiction,
            required_fields=required_fields,
            missing_fields=missing,
            is_compliant=is_compliant,
        )
    
    def _detect_structuring(self, sender: str, amount: float) -> bool:
        """Detect potential structuring (smurfing) activity."""
        sender_lower = sender.lower()
        
        # Get transaction history
        history = self.transaction_history.get(sender_lower, [])
        
        # Store current transaction
        history.append({
            "amount": amount,
            "timestamp": time.time(),
        })
        self.transaction_history[sender_lower] = history[-100:]  # Keep last 100
        
        # Check for structuring patterns
        recent_24h = [tx for tx in history if tx["timestamp"] > time.time() - 86400]
        
        # Count transactions just below threshold
        structuring_count = 0
        for tx in recent_24h:
            tx_amount = tx["amount"]
            if 9000 <= tx_amount < 10000:
                structuring_count += 1
        
        # Structuring if multiple near-threshold transactions
        if structuring_count >= 2:
            return True
        
        # Check cumulative amount
        total_24h = sum(tx["amount"] for tx in recent_24h)
        if total_24h >= 10000 and structuring_count >= 1:
            return True
        
        return False
    
    def _get_applicable_regulations(
        self,
        jurisdiction: Jurisdiction,
        amount: float
    ) -> List[str]:
        """Get list of applicable regulations."""
        regulations = []
        
        if jurisdiction == Jurisdiction.US:
            regulations.append("Bank Secrecy Act (BSA)")
            regulations.append("FinCEN Crypto Travel Rule")
            if amount >= 10000:
                regulations.append("Currency Transaction Reporting")
            regulations.append("OFAC Sanctions")
            regulations.append("GENIUS Act (2025)")
        
        elif jurisdiction == Jurisdiction.EU:
            regulations.append("MiCA (Markets in Crypto-Assets)")
            regulations.append("AMLD6 (6th AML Directive)")
            if amount >= 1000:
                regulations.append("EU Travel Rule")
        
        elif jurisdiction == Jurisdiction.UK:
            regulations.append("FCA Cryptoasset Regulations")
            regulations.append("MLR 2017 (Money Laundering Regulations)")
        
        # Always applicable
        regulations.append("FATF Recommendations")
        regulations.append("FATF Travel Rule")
        
        return regulations
    
    def _calculate_compliance_level(
        self,
        sanctions_result: SanctionsScreenResult,
        travel_rule_result: TravelRuleCheck,
        exceeds_ctr: bool,
        structuring: bool,
        risk_score: float,
        fraud_typologies: List[str],
    ) -> Tuple[ComplianceLevel, float]:
        """Calculate overall compliance level and score."""
        
        # Immediate blockers
        if sanctions_result.is_sanctioned:
            return ComplianceLevel.SANCTIONS_HIT, 0.0
        
        if "mixer_tumbling" in fraud_typologies and sanctions_result.match_score > 0.5:
            return ComplianceLevel.SANCTIONS_HIT, 5.0
        
        # High risk indicators
        score = 100.0
        
        if structuring:
            score -= 40
        
        if not travel_rule_result.is_compliant and travel_rule_result.applies:
            score -= 25
        
        if exceeds_ctr:
            score -= 10  # Not necessarily bad, just needs reporting
        
        if risk_score > 70:
            score -= 30
        elif risk_score > 50:
            score -= 20
        elif risk_score > 30:
            score -= 10
        
        # Fraud typology deductions
        high_risk_typologies = ["rug_pull", "mixer_tumbling", "pig_butchering"]
        for typology in fraud_typologies:
            if typology in high_risk_typologies:
                score -= 15
            else:
                score -= 5
        
        score = max(0, score)
        
        # Determine level
        if score >= 80:
            return ComplianceLevel.COMPLIANT, score
        elif score >= 60:
            return ComplianceLevel.NEEDS_REVIEW, score
        elif score >= 30:
            return ComplianceLevel.ENHANCED_DUE_DILIGENCE, score
        else:
            return ComplianceLevel.BLOCKED, score
    
    def _determine_required_reports(
        self,
        compliance_level: ComplianceLevel,
        exceeds_ctr: bool,
        risk_score: float,
        fraud_typologies: List[str],
        jurisdiction: Jurisdiction,
    ) -> List[ReportType]:
        """Determine which regulatory reports are required."""
        reports = []
        
        # CTR for large transactions (US)
        if exceeds_ctr and jurisdiction == Jurisdiction.US:
            reports.append(ReportType.CTR)
        
        # SAR for suspicious activity
        if risk_score > 50 or len(fraud_typologies) > 0:
            if jurisdiction == Jurisdiction.US:
                reports.append(ReportType.SAR)
            else:
                reports.append(ReportType.STR)
        
        # Travel Rule advisory
        if compliance_level in [ComplianceLevel.NEEDS_REVIEW, ComplianceLevel.ENHANCED_DUE_DILIGENCE]:
            reports.append(ReportType.TRAR)
        
        return reports
    
    def _generate_report(
        self,
        report_type: ReportType,
        tx_id: str,
        jurisdiction: Jurisdiction,
        sender: str,
        recipient: str,
        amount: float,
        currency: str,
        risk_score: float,
        fraud_typologies: List[str],
    ) -> RegulatoryReport:
        """Generate a regulatory report."""
        
        # Generate report ID
        report_id = f"{report_type.name}-{tx_id[:8]}-{int(time.time())}"
        
        # Generate narrative
        narrative = self._generate_narrative(
            report_type, amount, risk_score, fraud_typologies
        )
        
        # Determine recommended action
        if risk_score > 70:
            action = "BLOCK transaction pending investigation"
        elif risk_score > 50:
            action = "FLAG for enhanced due diligence review"
        else:
            action = "MONITOR and retain records per regulatory requirements"
        
        return RegulatoryReport(
            report_type=report_type,
            report_id=report_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transaction_id=tx_id,
            jurisdiction=jurisdiction,
            sender_address=sender,
            recipient_address=recipient,
            amount=amount,
            currency=currency,
            risk_score=risk_score,
            risk_indicators=fraud_typologies,
            narrative=narrative,
            recommended_action=action,
        )
    
    def _generate_narrative(
        self,
        report_type: ReportType,
        amount: float,
        risk_score: float,
        fraud_typologies: List[str],
    ) -> str:
        """Generate narrative for regulatory report."""
        
        parts = []
        
        parts.append(f"Transaction of ${amount:,.2f} flagged for {report_type.value}.")
        parts.append(f"AI-generated risk score: {risk_score:.1f}/100.")
        
        if fraud_typologies:
            typology_str = ", ".join(t.replace("_", " ").title() for t in fraud_typologies)
            parts.append(f"Detected fraud patterns: {typology_str}.")
        
        if "structuring" in fraud_typologies:
            parts.append("Transaction appears to be structured to avoid reporting thresholds (BSA violation).")
        
        if "mixer_tumbling" in fraud_typologies:
            parts.append("Transaction involves potential cryptocurrency mixing service - may indicate attempt to obscure fund origins.")
        
        if "rug_pull" in fraud_typologies:
            parts.append("Pattern consistent with potential exit scam/rug pull activity.")
        
        parts.append("Recommend compliance review and potential law enforcement referral if warranted.")
        
        return " ".join(parts)
    
    def _generate_recommendations(
        self,
        compliance_level: ComplianceLevel,
        sanctions_result: SanctionsScreenResult,
        travel_rule_result: TravelRuleCheck,
        exceeds_ctr: bool,
        structuring: bool,
        risk_score: float,
    ) -> Tuple[List[str], List[str]]:
        """Generate required actions and recommendations."""
        
        required = []
        recommendations = []
        
        if sanctions_result.is_sanctioned:
            required.append("IMMEDIATELY block transaction - potential sanctions violation")
            required.append("File SAR/STR within 30 days")
            required.append("Preserve all transaction records")
            required.append("Consider law enforcement notification")
        
        if structuring:
            required.append("File SAR for potential structuring activity")
            required.append("Review sender's transaction history for past 90 days")
        
        if exceeds_ctr:
            required.append("File Currency Transaction Report (CTR) within 15 days")
        
        if not travel_rule_result.is_compliant and travel_rule_result.applies:
            required.append(f"Obtain missing Travel Rule information: {', '.join(travel_rule_result.missing_fields)}")
        
        # Recommendations
        if risk_score > 50:
            recommendations.append("Consider enhanced due diligence for sender and recipient")
            recommendations.append("Review transaction against KYC information")
        
        if compliance_level == ComplianceLevel.NEEDS_REVIEW:
            recommendations.append("Manual review by compliance officer recommended")
        
        if compliance_level == ComplianceLevel.ENHANCED_DUE_DILIGENCE:
            recommendations.append("Perform enhanced due diligence before processing")
            recommendations.append("Document rationale for any approval decision")
        
        recommendations.append("Retain all records for minimum 5 years per regulatory requirements")
        
        return required, recommendations


# ═══════════════════════════════════════════════════════════════════════════════
#                              STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("REGULATORY COMPLIANCE ENGINE - TEST")
    print("GENIUS Act | MiCA | FATF | OFAC")
    print("=" * 80)
    
    engine = RegulatoryComplianceEngine()
    
    # Test cases
    test_cases = [
        {
            "name": "Normal Transaction",
            "sender": "0xAlice123456789012345678901234567890123456",
            "recipient": "0xBob123456789012345678901234567890123456",
            "amount": 500.0,
            "risk_score": 15.0,
            "fraud_typologies": [],
        },
        {
            "name": "Large CTR Transaction",
            "sender": "0xAlice123456789012345678901234567890123456",
            "recipient": "0xBob123456789012345678901234567890123456",
            "amount": 15000.0,
            "risk_score": 25.0,
            "fraud_typologies": [],
        },
        {
            "name": "Structuring Pattern",
            "sender": "0xStructurer12345678901234567890123456789",
            "recipient": "0xRecipient12345678901234567890123456789",
            "amount": 9999.0,
            "risk_score": 65.0,
            "fraud_typologies": ["structuring"],
        },
        {
            "name": "Sanctions Hit (Tornado Cash)",
            "sender": "0xUser1234567890123456789012345678901234567",
            "recipient": "0x722122df12d4e14e13ac3b6895a86e84145b6967",
            "amount": 10000.0,
            "risk_score": 95.0,
            "fraud_typologies": ["mixer_tumbling"],
        },
    ]
    
    for case in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Test: {case['name']}")
        print(f"{'─' * 80}")
        
        result = engine.check_compliance(
            sender=case["sender"],
            recipient=case["recipient"],
            amount=case["amount"],
            risk_score=case["risk_score"],
            fraud_typologies=case["fraud_typologies"],
        )
        
        print(f"Compliance Level: {result.compliance_level.emoji} {result.compliance_level.code_.upper()}")
        print(f"Compliance Score: {result.compliance_score:.1f}/100")
        print(f"Jurisdiction: {result.primary_jurisdiction.emoji} {result.primary_jurisdiction.full_name}")
        
        print(f"\nSanctions Screening:")
        print(f"  Is Sanctioned: {result.sanctions_check.is_sanctioned}")
        if result.sanctions_check.is_sanctioned:
            print(f"  Matched List: {result.sanctions_check.matched_list}")
        print(f"  Screening Time: {result.sanctions_check.screening_time_ms:.2f}ms")
        
        print(f"\nTravel Rule:")
        print(f"  Applies: {result.travel_rule_check.applies}")
        print(f"  Threshold: ${result.travel_rule_check.threshold:,.2f}")
        print(f"  Compliant: {result.travel_rule_check.is_compliant}")
        
        print(f"\nReports Required: {[r.value for r in result.reports_required]}")
        
        if result.required_actions:
            print(f"\nRequired Actions:")
            for action in result.required_actions[:3]:
                print(f"  • {action}")
        
        print(f"\nTotal Check Time: {result.check_time_ms:.2f}ms")
    
    print("\n" + "=" * 80)
