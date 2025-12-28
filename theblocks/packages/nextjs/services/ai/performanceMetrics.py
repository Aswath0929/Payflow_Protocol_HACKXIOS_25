"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW PERFORMANCE METRICS DASHBOARD                             ║
║                                                                                       ║
║   Real-time tracking of:                                                              ║
║   • Latency (<300ms for Visa requirement)                                            ║
║   • False Positive Rate (<2% for PayPal requirement)                                 ║
║   • Detection Accuracy (>98% target)                                                 ║
║   • Model Agreement & Confidence                                                     ║
║   • Compliance Coverage                                                              ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import deque
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
#                              JUDGE REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JudgeRequirement:
    """Requirements from hackathon judges."""
    name: str
    role: str
    company: str
    metric: str
    target: float
    unit: str
    priority: str
    
    def is_met(self, value: float) -> bool:
        if self.metric == "latency":
            return value < self.target
        elif self.metric == "false_positive_rate":
            return value < self.target
        elif self.metric in ["accuracy", "detection_rate"]:
            return value > self.target
        return True


# Define judge requirements
JUDGE_REQUIREMENTS = [
    JudgeRequirement(
        name="Mayank", role="Developer Experience", company="Visa",
        metric="latency", target=300, unit="ms", priority="critical"
    ),
    JudgeRequirement(
        name="Megha", role="Product Manager", company="PayPal",
        metric="false_positive_rate", target=2.0, unit="%", priority="critical"
    ),
    JudgeRequirement(
        name="Implicit", role="Technical", company="Hackxios",
        metric="accuracy", target=98.0, unit="%", priority="high"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
#                              METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionMetrics:
    """Metrics for a single transaction analysis."""
    
    # Identification
    transaction_id: str
    timestamp: float
    
    # Timing (ms)
    total_latency: float
    feature_extraction_time: float
    ensemble_time: float
    typology_time: float
    llm_time: float
    compliance_time: float
    
    # Results
    risk_score: int
    risk_level: str
    confidence: float
    
    # Model agreement
    ensemble_score: float
    typology_score: float
    llm_score: float
    compliance_risk: float
    model_agreement: float  # Standard deviation of scores
    
    # Ground truth (for evaluation)
    ground_truth: Optional[str] = None  # "fraud" or "legitimate"
    is_true_positive: Optional[bool] = None
    is_false_positive: Optional[bool] = None
    
    # Metadata
    detected_typologies: int = 0
    compliance_status: str = "unknown"


@dataclass
class AggregateMetrics:
    """Aggregated performance metrics."""
    
    # Counts
    total_transactions: int = 0
    total_fraud_detected: int = 0
    total_legitimate: int = 0
    
    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    
    # Accuracy (when ground truth available)
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    
    # Model performance
    avg_confidence: float = 0.0
    avg_model_agreement: float = 0.0
    
    # Breakdown
    avg_feature_time: float = 0.0
    avg_ensemble_time: float = 0.0
    avg_typology_time: float = 0.0
    avg_llm_time: float = 0.0
    avg_compliance_time: float = 0.0
    
    # Risk distribution
    safe_count: int = 0
    low_count: int = 0
    medium_count: int = 0
    high_count: int = 0
    critical_count: int = 0
    
    # Compliance
    compliant_count: int = 0
    needs_review_count: int = 0
    edd_count: int = 0
    blocked_count: int = 0
    sanctions_hit_count: int = 0
    
    # Judge requirements
    meets_visa_latency: bool = False
    meets_paypal_fpr: bool = False
    meets_accuracy_target: bool = False


class PerformanceMetricsCollector:
    """Collects and analyzes performance metrics in real-time."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize collector with sliding window."""
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.latencies: deque = deque(maxlen=window_size)
        self.start_time = time.time()
        
        # Real-time counters
        self._total_transactions = 0
        self._total_fraud = 0
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        
        # Risk level counters
        self._risk_counts = {
            "safe": 0, "low": 0, "medium": 0, "high": 0, "critical": 0
        }
        
        # Compliance counters
        self._compliance_counts = {
            "COMPLIANT": 0, "NEEDS_REVIEW": 0, "ENHANCED_DUE_DILIGENCE": 0,
            "BLOCKED": 0, "SANCTIONS_HIT": 0
        }
    
    def record_transaction(
        self,
        transaction_id: str,
        total_latency: float,
        feature_time: float,
        ensemble_time: float,
        typology_time: float,
        llm_time: float,
        compliance_time: float,
        risk_score: int,
        risk_level: str,
        confidence: float,
        ensemble_score: float,
        typology_score: float,
        llm_score: float,
        compliance_risk: float,
        detected_typologies: int = 0,
        compliance_status: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> TransactionMetrics:
        """Record metrics for a single transaction."""
        
        # Calculate model agreement (lower std = higher agreement)
        scores = [ensemble_score, typology_score, llm_score, compliance_risk]
        model_agreement = 1 - (np.std(scores) / 50)
        model_agreement = max(0, min(1, model_agreement))
        
        # Determine TP/FP/TN/FN
        is_tp = is_fp = None
        predicted_fraud = risk_score >= 50
        
        if ground_truth:
            actual_fraud = ground_truth.lower() == "fraud"
            if predicted_fraud and actual_fraud:
                is_tp = True
                self._tp += 1
            elif predicted_fraud and not actual_fraud:
                is_fp = True
                self._fp += 1
            elif not predicted_fraud and not actual_fraud:
                self._tn += 1
            else:
                self._fn += 1
        
        metrics = TransactionMetrics(
            transaction_id=transaction_id,
            timestamp=time.time(),
            total_latency=total_latency,
            feature_extraction_time=feature_time,
            ensemble_time=ensemble_time,
            typology_time=typology_time,
            llm_time=llm_time,
            compliance_time=compliance_time,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            ensemble_score=ensemble_score,
            typology_score=typology_score,
            llm_score=llm_score,
            compliance_risk=compliance_risk,
            model_agreement=model_agreement,
            ground_truth=ground_truth,
            is_true_positive=is_tp,
            is_false_positive=is_fp,
            detected_typologies=detected_typologies,
            compliance_status=compliance_status,
        )
        
        # Update counters
        self._total_transactions += 1
        self.latencies.append(total_latency)
        self.metrics_history.append(metrics)
        
        # Risk level
        if risk_level in self._risk_counts:
            self._risk_counts[risk_level] += 1
        
        # Compliance
        if compliance_status in self._compliance_counts:
            self._compliance_counts[compliance_status] += 1
        
        # Fraud count
        if risk_score >= 50:
            self._total_fraud += 1
        
        return metrics
    
    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Calculate aggregate metrics."""
        if not self.metrics_history:
            return AggregateMetrics()
        
        latencies = list(self.latencies)
        metrics_list = list(self.metrics_history)
        
        # Latency percentiles
        if latencies:
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
            avg_lat = np.mean(latencies)
            max_lat = max(latencies)
            min_lat = min(latencies)
        else:
            p50 = p95 = p99 = avg_lat = max_lat = 0
            min_lat = float('inf')
        
        # Accuracy metrics
        total_labeled = self._tp + self._fp + self._tn + self._fn
        if total_labeled > 0:
            accuracy = (self._tp + self._tn) / total_labeled * 100
            precision = self._tp / (self._tp + self._fp) * 100 if (self._tp + self._fp) > 0 else 0
            recall = self._tp / (self._tp + self._fn) * 100 if (self._tp + self._fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = self._fp / (self._fp + self._tn) * 100 if (self._fp + self._tn) > 0 else 0
        else:
            accuracy = precision = recall = f1 = fpr = 0
        
        # Average confidence and agreement
        avg_conf = np.mean([m.confidence for m in metrics_list])
        avg_agree = np.mean([m.model_agreement for m in metrics_list])
        
        # Breakdown times
        avg_feature = np.mean([m.feature_extraction_time for m in metrics_list])
        avg_ensemble = np.mean([m.ensemble_time for m in metrics_list])
        avg_typology = np.mean([m.typology_time for m in metrics_list])
        avg_llm = np.mean([m.llm_time for m in metrics_list])
        avg_compliance = np.mean([m.compliance_time for m in metrics_list])
        
        # Judge requirements
        meets_visa = p95 < 300
        meets_paypal = fpr < 2.0
        meets_accuracy = accuracy > 98.0
        
        return AggregateMetrics(
            total_transactions=self._total_transactions,
            total_fraud_detected=self._total_fraud,
            total_legitimate=self._total_transactions - self._total_fraud,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            max_latency_ms=max_lat,
            min_latency_ms=min_lat,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            avg_confidence=avg_conf,
            avg_model_agreement=avg_agree,
            avg_feature_time=avg_feature,
            avg_ensemble_time=avg_ensemble,
            avg_typology_time=avg_typology,
            avg_llm_time=avg_llm,
            avg_compliance_time=avg_compliance,
            safe_count=self._risk_counts["safe"],
            low_count=self._risk_counts["low"],
            medium_count=self._risk_counts["medium"],
            high_count=self._risk_counts["high"],
            critical_count=self._risk_counts["critical"],
            compliant_count=self._compliance_counts["COMPLIANT"],
            needs_review_count=self._compliance_counts["NEEDS_REVIEW"],
            edd_count=self._compliance_counts["ENHANCED_DUE_DILIGENCE"],
            blocked_count=self._compliance_counts["BLOCKED"],
            sanctions_hit_count=self._compliance_counts["SANCTIONS_HIT"],
            meets_visa_latency=meets_visa,
            meets_paypal_fpr=meets_paypal,
            meets_accuracy_target=meets_accuracy,
        )
    
    def get_judge_report(self) -> str:
        """Generate report for hackathon judges."""
        metrics = self.get_aggregate_metrics()
        
        report = []
        report.append("=" * 70)
        report.append("📊 PAYFLOW PERFORMANCE METRICS - JUDGE REPORT")
        report.append("=" * 70)
        
        report.append(f"\n📈 OVERVIEW:")
        report.append(f"   Total Transactions Analyzed: {metrics.total_transactions:,}")
        report.append(f"   Fraud Detected: {metrics.total_fraud_detected:,}")
        report.append(f"   Legitimate: {metrics.total_legitimate:,}")
        
        report.append(f"\n⚡ LATENCY PERFORMANCE:")
        report.append(f"   Average: {metrics.avg_latency_ms:.1f}ms")
        report.append(f"   P50: {metrics.p50_latency_ms:.1f}ms")
        report.append(f"   P95: {metrics.p95_latency_ms:.1f}ms {'✅' if metrics.p95_latency_ms < 300 else '❌'} (Visa: <300ms)")
        report.append(f"   P99: {metrics.p99_latency_ms:.1f}ms")
        
        report.append(f"\n   Breakdown:")
        report.append(f"   ├── Feature Extraction: {metrics.avg_feature_time:.1f}ms")
        report.append(f"   ├── Neural Ensemble: {metrics.avg_ensemble_time:.1f}ms")
        report.append(f"   ├── Typology Detection: {metrics.avg_typology_time:.1f}ms")
        report.append(f"   ├── LLM Analysis: {metrics.avg_llm_time:.1f}ms")
        report.append(f"   └── Compliance Check: {metrics.avg_compliance_time:.1f}ms")
        
        report.append(f"\n🎯 DETECTION ACCURACY:")
        if metrics.true_positives + metrics.false_positives + metrics.true_negatives + metrics.false_negatives > 0:
            report.append(f"   Accuracy: {metrics.accuracy:.2f}% {'✅' if metrics.accuracy > 98 else '❌'} (Target: >98%)")
            report.append(f"   Precision: {metrics.precision:.2f}%")
            report.append(f"   Recall: {metrics.recall:.2f}%")
            report.append(f"   F1 Score: {metrics.f1_score:.2f}%")
            report.append(f"   False Positive Rate: {metrics.false_positive_rate:.2f}% {'✅' if metrics.false_positive_rate < 2 else '❌'} (PayPal: <2%)")
        else:
            report.append(f"   (Ground truth data not available)")
        
        report.append(f"\n🤖 MODEL PERFORMANCE:")
        report.append(f"   Average Confidence: {metrics.avg_confidence:.1%}")
        report.append(f"   Model Agreement: {metrics.avg_model_agreement:.1%}")
        
        report.append(f"\n📊 RISK DISTRIBUTION:")
        total = max(1, sum(self._risk_counts.values()))
        for level, count in self._risk_counts.items():
            emoji = {"safe": "✅", "low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}.get(level, "•")
            pct = count / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            report.append(f"   {emoji} {level.upper():10s} |{bar}| {count:4d} ({pct:.1f}%)")
        
        report.append(f"\n📋 COMPLIANCE STATUS:")
        total = max(1, sum(self._compliance_counts.values()))
        for status, count in self._compliance_counts.items():
            pct = count / total * 100
            report.append(f"   {status:25s}: {count:4d} ({pct:.1f}%)")
        
        report.append(f"\n" + "=" * 70)
        report.append(f"👨‍⚖️ JUDGE REQUIREMENTS STATUS:")
        report.append(f"=" * 70)
        
        for req in JUDGE_REQUIREMENTS:
            if req.metric == "latency":
                value = metrics.p95_latency_ms
                status = "✅ PASS" if value < req.target else "❌ FAIL"
            elif req.metric == "false_positive_rate":
                value = metrics.false_positive_rate
                status = "✅ PASS" if value < req.target else "❌ FAIL" if metrics.true_positives > 0 else "⏳ PENDING"
            elif req.metric == "accuracy":
                value = metrics.accuracy
                status = "✅ PASS" if value > req.target else "❌ FAIL" if metrics.true_positives > 0 else "⏳ PENDING"
            else:
                value = 0
                status = "⏳ PENDING"
            
            report.append(f"\n   {req.name} ({req.company}):")
            report.append(f"   Metric: {req.metric}")
            report.append(f"   Target: {'<' if req.metric in ['latency', 'false_positive_rate'] else '>'}{req.target}{req.unit}")
            report.append(f"   Actual: {value:.2f}{req.unit}")
            report.append(f"   Status: {status}")
        
        report.append(f"\n" + "=" * 70)
        
        return "\n".join(report)
    
    def to_json(self) -> str:
        """Export metrics as JSON."""
        metrics = self.get_aggregate_metrics()
        return json.dumps(asdict(metrics), indent=2)
    
    def reset(self):
        """Reset all metrics."""
        self.metrics_history.clear()
        self.latencies.clear()
        self._total_transactions = 0
        self._total_fraud = 0
        self._tp = self._fp = self._tn = self._fn = 0
        self._risk_counts = {k: 0 for k in self._risk_counts}
        self._compliance_counts = {k: 0 for k in self._compliance_counts}


# ═══════════════════════════════════════════════════════════════════════════════
#                              GLOBAL COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

_global_collector: Optional[PerformanceMetricsCollector] = None

def get_metrics_collector() -> PerformanceMetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = PerformanceMetricsCollector()
    return _global_collector


# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS COLLECTOR TEST")
    print("=" * 70)
    
    collector = PerformanceMetricsCollector()
    
    # Simulate transactions
    import random
    
    print("\nSimulating 100 transactions...")
    
    for i in range(100):
        # Simulate metrics
        latency = random.uniform(50, 250)
        risk = random.randint(0, 100)
        level = ["safe", "low", "medium", "high", "critical"][min(4, risk // 20)]
        
        # Ground truth (80% accuracy simulation)
        is_fraud = risk > 50
        if random.random() > 0.8:
            is_fraud = not is_fraud
        ground_truth = "fraud" if is_fraud else "legitimate"
        
        collector.record_transaction(
            transaction_id=f"tx_{i:04d}",
            total_latency=latency,
            feature_time=latency * 0.05,
            ensemble_time=latency * 0.20,
            typology_time=latency * 0.15,
            llm_time=latency * 0.55,
            compliance_time=latency * 0.05,
            risk_score=risk,
            risk_level=level,
            confidence=random.uniform(0.7, 1.0),
            ensemble_score=risk + random.uniform(-10, 10),
            typology_score=risk + random.uniform(-10, 10),
            llm_score=risk + random.uniform(-10, 10),
            compliance_risk=risk + random.uniform(-10, 10),
            detected_typologies=random.randint(0, 3),
            compliance_status=random.choice(["COMPLIANT", "NEEDS_REVIEW", "ENHANCED_DUE_DILIGENCE"]),
            ground_truth=ground_truth,
        )
    
    print("\n" + collector.get_judge_report())
