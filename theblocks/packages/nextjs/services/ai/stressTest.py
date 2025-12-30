"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PAYFLOW FRAUD ENGINE - STRESS TEST                                     ║
║                                                                                           ║
║   🔥 Production Stress Test for Lightning Fraud Engine                                   ║
║                                                                                           ║
║   Tests:                                                                                  ║
║   1. Throughput Test: Max transactions per second                                        ║
║   2. Latency Distribution: P50, P95, P99 latencies                                       ║
║   3. Accuracy Test: Fraud detection accuracy                                             ║
║   4. Mixed Load: Real-world transaction mix                                              ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import random
import string
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from lightningFraudEngine import LightningFraudEngine, MIXERS


# ═══════════════════════════════════════════════════════════════════════════════
#                           TEST DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def random_address() -> str:
    """Generate random Ethereum address."""
    return "0x" + "".join(random.choices("0123456789abcdef", k=40))


def generate_safe_tx() -> Tuple[dict, bool]:
    """Generate a clearly safe transaction."""
    return {
        "tx_id": f"safe_{random.randint(1000, 9999)}",
        "sender": random_address(),
        "recipient": random_address(),
        "amount": random.uniform(10, 500),
        "sender_tx_count": random.randint(50, 500),
    }, False  # Expected: not blocked


def generate_medium_tx() -> Tuple[dict, bool]:
    """Generate a medium-risk transaction."""
    return {
        "tx_id": f"med_{random.randint(1000, 9999)}",
        "sender": random_address(),
        "recipient": random_address(),
        "amount": random.uniform(5000, 25000),
        "sender_tx_count": random.randint(20, 100),
    }, False  # Expected: not blocked (but possibly flagged)


def generate_dangerous_tx() -> Tuple[dict, bool]:
    """Generate a clearly dangerous transaction."""
    return {
        "tx_id": f"danger_{random.randint(1000, 9999)}",
        "sender": random_address(),
        "recipient": random.choice(list(MIXERS)),  # Known mixer
        "amount": random.uniform(10000, 100000),
    }, True  # Expected: blocked


def generate_structuring_tx() -> Tuple[dict, bool]:
    """Generate structuring (just under $10k)."""
    return {
        "tx_id": f"struct_{random.randint(1000, 9999)}",
        "sender": random_address(),
        "recipient": random_address(),
        "amount": random.uniform(9000, 9999),
        "sender_tx_count": random.randint(1, 10),
    }, None  # Expected: flagged (not blocked)


def generate_whale_tx() -> Tuple[dict, bool]:
    """Generate whale transaction."""
    return {
        "tx_id": f"whale_{random.randint(1000, 9999)}",
        "sender": random_address(),
        "recipient": random_address(),
        "amount": random.uniform(100000, 500000),
        "sender_tx_count": random.randint(100, 500),
    }, False  # Expected: not blocked (but flagged for review)


def generate_mixed_batch(n: int) -> List[Tuple[dict, bool]]:
    """Generate a realistic mix of transactions."""
    batch = []
    
    # Distribution: 60% safe, 20% medium, 10% whale, 5% structuring, 5% dangerous
    for _ in range(int(n * 0.60)):
        batch.append(generate_safe_tx())
    
    for _ in range(int(n * 0.20)):
        batch.append(generate_medium_tx())
    
    for _ in range(int(n * 0.10)):
        batch.append(generate_whale_tx())
    
    for _ in range(int(n * 0.05)):
        batch.append(generate_structuring_tx())
    
    for _ in range(int(n * 0.05)):
        batch.append(generate_dangerous_tx())
    
    random.shuffle(batch)
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
#                           TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResults:
    """Test results container."""
    total_transactions: int
    total_time_ms: float
    throughput_tps: float
    latencies: List[float]
    correct_predictions: int
    total_predictions: int
    blocked_count: int
    flagged_count: int
    approved_count: int
    
    @property
    def accuracy(self) -> float:
        return self.correct_predictions / max(1, self.total_predictions)
    
    @property
    def avg_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0
    
    @property
    def p50_latency(self) -> float:
        return np.percentile(self.latencies, 50) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0
    
    @property
    def p99_latency(self) -> float:
        return np.percentile(self.latencies, 99) if self.latencies else 0


async def run_stress_test(engine: LightningFraudEngine, n: int = 50) -> TestResults:
    """Run stress test with n transactions."""
    
    batch = generate_mixed_batch(n)
    
    latencies = []
    correct = 0
    total_checks = 0
    blocked = 0
    flagged = 0
    approved = 0
    
    start = time.perf_counter()
    
    for tx, expected_blocked in batch:
        result = await engine.analyze(tx)
        latencies.append(result.latency_ms)
        
        if result.blocked:
            blocked += 1
        if result.flagged:
            flagged += 1
        if result.approved:
            approved += 1
        
        # Check accuracy for transactions with known expected outcome
        if expected_blocked is not None:
            total_checks += 1
            if result.blocked == expected_blocked:
                correct += 1
    
    total_time = (time.perf_counter() - start) * 1000
    
    return TestResults(
        total_transactions=n,
        total_time_ms=total_time,
        throughput_tps=n / (total_time / 1000),
        latencies=latencies,
        correct_predictions=correct,
        total_predictions=total_checks,
        blocked_count=blocked,
        flagged_count=flagged,
        approved_count=approved,
    )


async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   🔥 PAYFLOW FRAUD ENGINE - PRODUCTION STRESS TEST                                       ║
║                                                                                           ║
║   Testing: Throughput, Latency Distribution, Accuracy                                    ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    engine = LightningFraudEngine()
    await engine.init()
    
    # Test 1: Quick warmup
    print("\n🔥 Warmup (10 transactions)...")
    warmup = await run_stress_test(engine, 10)
    print(f"   Warmup complete. Avg latency: {warmup.avg_latency:.1f}ms")
    
    # Test 2: Standard load (50 transactions)
    print("\n📊 Standard Load Test (50 transactions)...")
    standard = await run_stress_test(engine, 50)
    
    print(f"""
   Results:
   • Total Time: {standard.total_time_ms:.0f}ms
   • Throughput: {standard.throughput_tps:.1f} tx/sec
   • Latency:
     - Average: {standard.avg_latency:.1f}ms
     - P50:     {standard.p50_latency:.1f}ms
     - P95:     {standard.p95_latency:.1f}ms
     - P99:     {standard.p99_latency:.1f}ms
   • Decisions:
     - Approved: {standard.approved_count}
     - Flagged:  {standard.flagged_count}
     - Blocked:  {standard.blocked_count}
   • Accuracy: {standard.accuracy * 100:.1f}% ({standard.correct_predictions}/{standard.total_predictions})
""")
    
    # Test 3: Heavy load (100 transactions)
    print("🔥 Heavy Load Test (100 transactions)...")
    heavy = await run_stress_test(engine, 100)
    
    print(f"""
   Results:
   • Total Time: {heavy.total_time_ms:.0f}ms
   • Throughput: {heavy.throughput_tps:.1f} tx/sec
   • Latency:
     - Average: {heavy.avg_latency:.1f}ms
     - P50:     {heavy.p50_latency:.1f}ms
     - P95:     {heavy.p95_latency:.1f}ms
     - P99:     {heavy.p99_latency:.1f}ms
   • Accuracy: {heavy.accuracy * 100:.1f}%
""")
    
    # Summary
    print("=" * 80)
    print("\n🎯 PRODUCTION READINESS CHECK:\n")
    
    checks = [
        ("Average Latency < 100ms", standard.avg_latency < 100, f"{standard.avg_latency:.1f}ms"),
        ("P95 Latency < 300ms", standard.p95_latency < 300, f"{standard.p95_latency:.1f}ms"),
        ("P99 Latency < 500ms", standard.p99_latency < 500, f"{standard.p99_latency:.1f}ms"),
        ("Throughput > 5 tx/sec", standard.throughput_tps > 5, f"{standard.throughput_tps:.1f} tx/sec"),
        ("Accuracy > 90%", standard.accuracy > 0.90, f"{standard.accuracy * 100:.1f}%"),
        ("Blocked Dangerous Txs", standard.blocked_count > 0, f"{standard.blocked_count} blocked"),
    ]
    
    all_passed = True
    for name, passed, value in checks:
        status = "✅" if passed else "❌"
        if not passed:
            all_passed = False
        print(f"   {status} {name}: {value}")
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   🏆 ALL TESTS PASSED - PRODUCTION READY!                                                ║
║                                                                                           ║
║   The fraud detection engine meets all performance and accuracy targets.                 ║
║   Ready for deployment at Hackxios 2K25!                                                 ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   ⚠️  SOME TESTS FAILED - REVIEW REQUIRED                                                ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    asyncio.run(main())
