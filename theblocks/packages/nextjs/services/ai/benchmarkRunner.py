"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                   ULTIMATE FRAUD ENGINE - BENCHMARK RUNNER                                ║
║                                                                                           ║
║   🔥 FULL THROTTLE GPU STRESS TEST                                                       ║
║   📊 LATENCY & ACCURACY BENCHMARKS                                                       ║
║   🎯 PERFORMANCE TUNING ANALYSIS                                                         ║
║   ⚡ RTX 4070 8GB VRAM DEDICATED                                                         ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import json
import random
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np

# Check and install colorama if needed
try:
    from colorama import Fore, Style, init
    init()
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama", "-q"])
    from colorama import Fore, Style, init
    init()


# ═══════════════════════════════════════════════════════════════════════════════
#                           BENCHMARK CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    
    # Test modes
    QUICK_TEST = 10        # Quick validation (10 transactions)
    STANDARD_TEST = 50     # Standard test (50 transactions)
    FULL_TEST = 200        # Full benchmark (200 transactions)
    STRESS_TEST = 500      # Stress test (500 transactions)
    
    # Performance targets
    TARGET_LATENCY_MS = 300     # <300ms target
    ACCEPTABLE_LATENCY_MS = 500  # <500ms acceptable
    MAX_LATENCY_MS = 1000        # >1000ms is failure
    
    # GPU settings to test
    GPU_CONFIGS = [
        {"name": "Thinking Mode (Accuracy)", "num_ctx": 4096, "num_predict": 1280, "temperature": 0.6},
        {"name": "Fast Mode (Speed)", "num_ctx": 2048, "num_predict": 128, "temperature": 0.3},
        {"name": "Balanced Mode", "num_ctx": 3072, "num_predict": 512, "temperature": 0.45},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#                           TRANSACTION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class QuickTransactionGenerator:
    """Generates test transactions quickly."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.counter = 0
    
    def _addr(self) -> str:
        return "0x" + hashlib.md5(str(random.random()).encode()).hexdigest()[:40]
    
    def legitimate(self) -> dict:
        """Generate legitimate transaction."""
        self.counter += 1
        return {
            "tx_id": f"tx_{self.counter:05d}",
            "sender": self._addr(),
            "recipient": self._addr(),
            "amount": random.uniform(100, 5000),
            "token": "USDC",
            "expected_safe": True,
            "typology": None,
        }
    
    def fraudulent(self, typology: str) -> dict:
        """Generate fraudulent transaction."""
        self.counter += 1
        
        if typology == "structuring":
            amount = random.choice([9900, 9950, 9990]) + random.uniform(-30, 30)
        elif typology == "mixer":
            recipient = "0x8589427373d6d84e98730d7795d8f6f8731fda16"
            amount = random.uniform(1000, 50000)
        elif typology == "flash_loan":
            amount = random.uniform(1000000, 10000000)
        elif typology == "rug_pull":
            amount = random.uniform(100000, 5000000)
        else:
            amount = random.uniform(10000, 100000)
        
        return {
            "tx_id": f"tx_{self.counter:05d}",
            "sender": self._addr(),
            "recipient": recipient if typology == "mixer" else self._addr(),
            "amount": amount,
            "token": "USDC",
            "expected_safe": False,
            "typology": typology,
        }
    
    def mixed_batch(self, count: int, fraud_ratio: float = 0.3) -> List[dict]:
        """Generate mixed batch of transactions."""
        txs = []
        fraud_count = int(count * fraud_ratio)
        legit_count = count - fraud_count
        
        # Generate legitimate
        for _ in range(legit_count):
            txs.append(self.legitimate())
        
        # Generate fraudulent
        typologies = ["structuring", "mixer", "flash_loan", "rug_pull", "velocity"]
        for _ in range(fraud_count):
            txs.append(self.fraudulent(random.choice(typologies)))
        
        random.shuffle(txs)
        return txs


# ═══════════════════════════════════════════════════════════════════════════════
#                           BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Run comprehensive benchmarks on the fraud engine."""
    
    def __init__(self):
        self.generator = QuickTransactionGenerator()
        self.results: Dict[str, List] = {
            "latencies": [],
            "scores": [],
            "decisions": [],
            "typologies": [],
        }
        self.config = BenchmarkConfig()
    
    def _print_header(self, text: str):
        """Print header."""
        print(f"\n{Fore.CYAN}{'═' * 70}")
        print(f"{text:^70}")
        print(f"{'═' * 70}{Style.RESET_ALL}")
    
    def _print_metric(self, label: str, value: str, color: str = Fore.WHITE):
        """Print metric."""
        print(f"  {color}• {label:.<45} {value}{Style.RESET_ALL}")
    
    async def run_benchmark(self, test_mode: str = "standard"):
        """Run benchmark suite."""
        
        print(f"""
{Fore.YELLOW}
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗          ║
║   ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝          ║
║   ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝           ║
║   ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗           ║
║   ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗          ║
║   ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝          ║
║                                                                                           ║
║                         ULTIMATE FRAUD ENGINE BENCHMARK                                   ║
║                         PayFlow Protocol - Hackxios 2K25                                  ║
║                                                                                           ║
║   🚀 RTX 4070 | 8GB VRAM Dedicated | Qwen3:8B + Thinking Mode                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        
        # Select test size
        test_sizes = {
            "quick": self.config.QUICK_TEST,
            "standard": self.config.STANDARD_TEST,
            "full": self.config.FULL_TEST,
            "stress": self.config.STRESS_TEST,
        }
        tx_count = test_sizes.get(test_mode, self.config.STANDARD_TEST)
        
        print(f"\n  {Fore.GREEN}📋 Test Mode: {test_mode.upper()} ({tx_count} transactions){Style.RESET_ALL}")
        
        # Initialize engine
        self._print_header("🔧 INITIALIZING ENGINE")
        
        try:
            from ultimateFraudEngine import get_engine, GPUConfig
            engine = await get_engine()
            print(f"  {Fore.GREEN}✓ Engine v{engine.VERSION} loaded{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}✓ GPU: {GPUConfig.GPU_LAYERS} layers, {GPUConfig.NUM_CTX} context{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}✓ Model: {GPUConfig.MODEL}{Style.RESET_ALL}")
        except Exception as e:
            print(f"  {Fore.RED}✗ Failed to initialize engine: {e}{Style.RESET_ALL}")
            return
        
        from ultimateFraudEngine import Transaction
        
        # Generate test transactions
        self._print_header("📦 GENERATING TEST TRANSACTIONS")
        transactions = self.generator.mixed_batch(tx_count, fraud_ratio=0.3)
        print(f"  {Fore.GREEN}✓ Generated {len(transactions)} transactions{Style.RESET_ALL}")
        fraud_count = sum(1 for t in transactions if not t["expected_safe"])
        print(f"  {Fore.GREEN}✓ {fraud_count} fraudulent, {tx_count - fraud_count} legitimate{Style.RESET_ALL}")
        
        # Run benchmark
        self._print_header("⚡ RUNNING BENCHMARK")
        
        start_time = time.time()
        latencies = []
        correct_decisions = 0
        
        for i, tx_data in enumerate(transactions):
            tx = Transaction(
                tx_id=tx_data["tx_id"],
                sender=tx_data["sender"],
                recipient=tx_data["recipient"],
                amount=tx_data["amount"],
                token=tx_data["token"],
            )
            
            result = await engine.analyze(tx)
            latencies.append(result.analysis_time_ms)
            
            # Check correctness
            expected_flagged = not tx_data["expected_safe"]
            actual_flagged = result.flagged or result.blocked
            if expected_flagged == actual_flagged:
                correct_decisions += 1
            
            # Progress update
            if (i + 1) % 10 == 0 or i == len(transactions) - 1:
                avg_lat = np.mean(latencies[-10:])
                pct = (i + 1) / len(transactions) * 100
                status = "🟢" if avg_lat < 300 else "🟡" if avg_lat < 500 else "🔴"
                print(f"  {status} Progress: {i+1}/{len(transactions)} ({pct:.0f}%) | Last 10 avg: {avg_lat:.0f}ms")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        self._print_header("📊 BENCHMARK RESULTS")
        
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        min_lat = min(latencies)
        max_lat = max(latencies)
        throughput = len(transactions) / total_time
        accuracy = correct_decisions / len(transactions) * 100
        
        # Performance metrics
        print(f"\n  {Fore.YELLOW}═══ LATENCY METRICS ═══{Style.RESET_ALL}")
        
        lat_color = Fore.GREEN if avg_latency < 300 else Fore.YELLOW if avg_latency < 500 else Fore.RED
        self._print_metric("Average Latency", f"{avg_latency:.0f}ms (target: <300ms)", lat_color)
        self._print_metric("P50 (Median)", f"{p50:.0f}ms", lat_color)
        self._print_metric("P95", f"{p95:.0f}ms", lat_color)
        self._print_metric("P99", f"{p99:.0f}ms", lat_color)
        self._print_metric("Min", f"{min_lat:.0f}ms", Fore.WHITE)
        self._print_metric("Max", f"{max_lat:.0f}ms", Fore.WHITE)
        
        print(f"\n  {Fore.YELLOW}═══ THROUGHPUT METRICS ═══{Style.RESET_ALL}")
        self._print_metric("Total Time", f"{total_time:.2f}s", Fore.WHITE)
        self._print_metric("Transactions", f"{len(transactions)}", Fore.WHITE)
        self._print_metric("Throughput", f"{throughput:.1f} tx/sec", Fore.GREEN if throughput > 2 else Fore.YELLOW)
        
        print(f"\n  {Fore.YELLOW}═══ ACCURACY METRICS ═══{Style.RESET_ALL}")
        acc_color = Fore.GREEN if accuracy >= 90 else Fore.YELLOW if accuracy >= 80 else Fore.RED
        self._print_metric("Decision Accuracy", f"{accuracy:.1f}%", acc_color)
        self._print_metric("Correct Decisions", f"{correct_decisions}/{len(transactions)}", acc_color)
        
        # Distribution
        print(f"\n  {Fore.YELLOW}═══ LATENCY DISTRIBUTION ═══{Style.RESET_ALL}")
        buckets = {
            "<100ms": sum(1 for l in latencies if l < 100),
            "100-200ms": sum(1 for l in latencies if 100 <= l < 200),
            "200-300ms": sum(1 for l in latencies if 200 <= l < 300),
            "300-500ms": sum(1 for l in latencies if 300 <= l < 500),
            "500ms+": sum(1 for l in latencies if l >= 500),
        }
        
        for bucket, count in buckets.items():
            pct = count / len(latencies) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            color = Fore.GREEN if bucket.startswith("<") or bucket.startswith("1") else Fore.YELLOW if "300" in bucket else Fore.RED
            print(f"  {color}{bucket:.<12} [{bar}] {pct:.1f}% ({count}){Style.RESET_ALL}")
        
        # Final verdict
        print(f"\n  {Fore.YELLOW}═══ BENCHMARK VERDICT ═══{Style.RESET_ALL}")
        
        checks = [
            ("Avg Latency < 300ms", avg_latency < 300, avg_latency < 500),
            ("P95 Latency < 500ms", p95 < 500, p95 < 1000),
            ("Accuracy >= 90%", accuracy >= 90, accuracy >= 80),
            ("Throughput >= 2 tx/s", throughput >= 2, throughput >= 1),
        ]
        
        passes = 0
        partial = 0
        
        for check_name, full_pass, partial_pass in checks:
            if full_pass:
                status = f"{Fore.GREEN}✅ PASS{Style.RESET_ALL}"
                passes += 1
            elif partial_pass:
                status = f"{Fore.YELLOW}⚠️ PARTIAL{Style.RESET_ALL}"
                partial += 1
            else:
                status = f"{Fore.RED}❌ FAIL{Style.RESET_ALL}"
            print(f"  {check_name:.<40} {status}")
        
        # Overall result
        if passes == len(checks):
            print(f"""
{Fore.GREEN}
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   🎉 ALL BENCHMARKS PASSED - PRODUCTION READY! 🚀               ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        elif passes + partial >= 3:
            print(f"""
{Fore.YELLOW}
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   ⚠️  BENCHMARK MOSTLY PASSED - ACCEPTABLE FOR DEMO             ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        else:
            print(f"""
{Fore.RED}
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   ❌ BENCHMARK FAILED - NEEDS OPTIMIZATION                       ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        
        # Engine stats
        self._print_header("📈 ENGINE STATISTICS")
        stats = engine.get_stats()
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {Fore.CYAN}{key}:{Style.RESET_ALL}")
                for k, v in value.items():
                    self._print_metric(f"  {k}", str(v))
            else:
                self._print_metric(key, str(value))
        
        await engine.close()
        
        # Save results
        results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_mode": test_mode,
            "transaction_count": len(transactions),
            "latency": {
                "avg_ms": round(avg_latency, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "min_ms": round(min_lat, 2),
                "max_ms": round(max_lat, 2),
            },
            "throughput_tx_per_sec": round(throughput, 2),
            "accuracy_percent": round(accuracy, 2),
            "distribution": buckets,
            "passes": passes,
            "partial": partial,
            "total_checks": len(checks),
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  {Fore.GREEN}📁 Results saved to: {results_file}{Style.RESET_ALL}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Fraud Engine Benchmark")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full", "stress"],
        default="standard",
        help="Benchmark mode (default: standard)"
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    await runner.run_benchmark(args.mode)


if __name__ == "__main__":
    asyncio.run(main())
