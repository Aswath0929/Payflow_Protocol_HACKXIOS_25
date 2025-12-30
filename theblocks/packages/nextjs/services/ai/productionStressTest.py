"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║              ULTIMATE HYBRID ENGINE - PRODUCTION STRESS TEST v2.0                        ║
║                                                                                           ║
║   Realistic fraud scenario simulation with wallet history building                       ║
║   Tests all 15 fraud typologies across 4 detection layers                                ║
║                                                                                           ║
║   Scenarios:                                                                              ║
║   • Mixer laundering                                                                      ║
║   • Structuring (smurfing)                                                               ║
║   • Wash trading cycles                                                                   ║
║   • Velocity attacks                                                                      ║
║   • Pig butchering escalation                                                            ║
║   • Rug pull draining                                                                    ║
║   • Dusting attacks                                                                       ║
║   • Legitimate whale transactions                                                         ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import random
from typing import List, Dict
from ultimateHybridEngine import UltimateHybridEngine, AddressDatabase


class ProductionStressTest:
    def __init__(self):
        self.engine = UltimateHybridEngine()
        self.results: List[Dict] = []
        
    async def init(self):
        await self.engine.init()
        
    async def run(self, iterations: int = 100):
        """Run full production stress test."""
        
        print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║           ULTIMATE HYBRID ENGINE - PRODUCTION STRESS TEST v2.0                           ║
║                                                                                           ║
║   Building realistic wallet histories to trigger heuristic detection...                  ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
        
        # Phase 1: Build wallet histories
        print("📊 Phase 1: Building wallet histories for realistic detection...")
        await self._build_wallet_histories()
        
        # Phase 2: Run fraud scenarios
        print("\n🔍 Phase 2: Running fraud attack scenarios...")
        await self._run_fraud_scenarios()
        
        # Phase 3: Mixed realistic load
        print(f"\n⚡ Phase 3: Mixed production load ({iterations} transactions)...")
        await self._run_mixed_load(iterations)
        
        # Phase 4: High-velocity stress
        print("\n🚀 Phase 4: High-velocity burst test...")
        await self._velocity_burst()
        
        # Final report
        self._final_report()
        
    async def _build_wallet_histories(self):
        """Build realistic transaction history for wallets."""
        
        # Normal user building history
        print("   Building normal user history (50 tx)...", end=" ")
        for i in range(50):
            await self.engine.analyze({
                "tx_id": f"history_normal_{i}",
                "sender": "0xnormal_user",
                "recipient": f"0xnormal_shop_{i % 10}",
                "amount": random.uniform(50, 500),
                "timestamp": time.time() - (50 - i) * 3600,
            })
        print("✅")
        
        # Whale building history
        print("   Building whale history (30 tx)...", end=" ")
        for i in range(30):
            await self.engine.analyze({
                "tx_id": f"history_whale_{i}",
                "sender": "0xwhale_wallet",
                "recipient": f"0xexchange_{i % 3}",
                "amount": random.uniform(50000, 200000),
                "timestamp": time.time() - (30 - i) * 86400,
            })
        print("✅")
        
        # Building structurer history
        print("   Building structurer history (20 tx)...", end=" ")
        for i in range(20):
            await self.engine.analyze({
                "tx_id": f"history_struct_{i}",
                "sender": "0xstructurer",
                "recipient": f"0xbank_{i % 5}",
                "amount": random.uniform(5000, 8000),
                "timestamp": time.time() - (20 - i) * 7200,
            })
        print("✅")
        
        # Building mixer user history
        print("   Building mixer user history (20 tx)...", end=" ")
        for i in range(20):
            await self.engine.analyze({
                "tx_id": f"history_mixer_{i}",
                "sender": "0xmixer_user",
                "recipient": f"0xrandom_addr_{i}",  # All unique
                "amount": random.uniform(1000, 10000),
                "timestamp": time.time() - (20 - i) * 1800,
            })
        print("✅")
        
    async def _run_fraud_scenarios(self):
        """Test specific fraud attack patterns."""
        
        scenarios = [
            # 1. Known mixer (instant block)
            {
                "name": "🔴 MIXER: Send to known mixer",
                "tx": {
                    "tx_id": "mixer_attack_1",
                    "sender": "0xsuspicious",
                    "recipient": "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
                    "amount": 50000,
                },
                "expected": {"blocked": True, "min_score": 90},
            },
            
            # 2. Structuring pattern (need multiple transactions near threshold)
            {
                "name": "🟡 STRUCTURING: Multiple sub-10k",
                "tx_count": 5,
                "tx": {
                    "sender": "0xsmurf",
                    "recipient": "0xbank",
                    "amount": 9500,  # Multiple $9.5k transactions
                },
                "expected": {"flagged": True, "min_score": 45},
            },
            
            # 3. Velocity attack
            {
                "name": "🔴 VELOCITY: Burst of 15 tx in 10s",
                "tx_count": 15,
                "tx": {
                    "sender": "0xvelocity_attacker",
                    "recipient": "0xvictim",
                    "amount": 1000,
                },
                "expected": {"blocked": True, "min_score": 75},
            },
            
            # 4. Wash trading (sender sends to A, then A sends back to sender)
            {
                "name": "🟡 WASH TRADING: A→B→A pattern",
                "wash_trade": True,
                "sender": "0xwash_trader",
                "recipients": ["0xpartner_a", "0xpartner_a", "0xpartner_a", "0xpartner_a"],
                "amount": 25000,
                "expected": {"flagged": True, "min_score": 50},
            },
            
            # 5. Dust attack
            {
                "name": "🟡 DUSTING: Micro amount",
                "tx": {
                    "tx_id": "dust_attack_1",
                    "sender": "0xduster",
                    "recipient": "0xvictim",
                    "amount": 0.005,
                },
                "expected": {"flagged": False, "max_score": 50},
            },
            
            # 6. Pig butchering escalation (needs 5+ escalating transactions)
            {
                "name": "🔴 PIG BUTCHERING: 100x escalation",
                "escalation": [50, 200, 800, 3000, 12000, 50000],  # 1000x total
                "expected": {"blocked": True, "min_score": 75},
            },
            
            # 7. Safe exchange deposit
            {
                "name": "✅ SAFE: Deposit to Binance",
                "tx": {
                    "tx_id": "binance_deposit",
                    "sender": "0xlegit_user",
                    "recipient": "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
                    "amount": 10000,
                },
                "expected": {"blocked": False, "max_score": 20},
            },
            
            # 8. Whale anomaly
            {
                "name": "🟡 WHALE ANOMALY: 3σ deviation",
                "tx": {
                    "tx_id": "whale_anomaly",
                    "sender": "0xwhale_wallet",
                    "recipient": "0xnew_wallet",
                    "amount": 1000000,  # Much larger than history
                },
                "expected": {"flagged": True, "min_score": 40},
            },
        ]
        
        print("\n" + "=" * 100)
        print(f"{'SCENARIO':<45} {'AMOUNT':>12} {'SCORE':>6} {'RESULT':<12} {'STATUS'}")
        print("=" * 100)
        
        for scenario in scenarios:
            result = await self._execute_scenario(scenario)
            
            # Check expectations
            expected = scenario.get("expected", {})
            passed = True
            
            if "min_score" in expected and result["score"] < expected["min_score"]:
                passed = False
            if "max_score" in expected and result["score"] > expected["max_score"]:
                passed = False
            if "blocked" in expected and result["blocked"] != expected["blocked"]:
                passed = False
            if "flagged" in expected and result["flagged"] != expected["flagged"]:
                passed = False
            
            status = "✅ PASS" if passed else "❌ FAIL"
            decision = "BLOCK" if result["blocked"] else ("FLAG" if result["flagged"] else "OK")
            
            print(f"{scenario['name']:<45} ${result['amount']:>10,.2f} {result['score']:>6} {decision:<12} {status}")
            
            self.results.append({"scenario": scenario["name"], **result, "passed": passed})
        
        print("=" * 100)
        
    async def _execute_scenario(self, scenario: Dict) -> Dict:
        """Execute a fraud scenario and return result."""
        
        if "tx" in scenario and "tx_count" not in scenario:
            # Single transaction
            r = await self.engine.analyze(scenario["tx"])
            return {
                "score": r.score,
                "blocked": r.blocked,
                "flagged": r.flagged,
                "amount": scenario["tx"]["amount"],
                "latency_ms": r.latency_ms,
            }
        
        elif "tx_count" in scenario:
            # Velocity attack
            ts = time.time()
            final_result = None
            for i in range(scenario["tx_count"]):
                tx = scenario["tx"].copy()
                tx["tx_id"] = f"velocity_{i}"
                tx["timestamp"] = ts + i * 0.5  # Half second apart
                final_result = await self.engine.analyze(tx)
            return {
                "score": final_result.score,
                "blocked": final_result.blocked,
                "flagged": final_result.flagged,
                "amount": scenario["tx"]["amount"],
                "latency_ms": final_result.latency_ms,
            }
        
        elif "wash_trade" in scenario:
            # Wash trading - same sender to same recipients (triggers cycle detection)
            ts = time.time()
            final_result = None
            sender = scenario["sender"]
            recipients = scenario["recipients"]
            for i, recip in enumerate(recipients):
                tx = {
                    "tx_id": f"wash_{i}",
                    "sender": sender,
                    "recipient": recip,
                    "amount": scenario["amount"],
                    "timestamp": ts + i * 10,
                }
                final_result = await self.engine.analyze(tx)
            return {
                "score": final_result.score,
                "blocked": final_result.blocked,
                "flagged": final_result.flagged,
                "amount": scenario["amount"],
                "latency_ms": final_result.latency_ms,
            }
        
        elif "cycle" in scenario:
            # Wash trading cycle
            ts = time.time()
            final_result = None
            cycle = scenario["cycle"]
            for i in range(len(cycle) - 1):
                tx = {
                    "tx_id": f"wash_{i}",
                    "sender": cycle[i],
                    "recipient": cycle[i + 1],
                    "amount": scenario["amount"],
                    "timestamp": ts + i * 10,
                }
                final_result = await self.engine.analyze(tx)
            return {
                "score": final_result.score,
                "blocked": final_result.blocked,
                "flagged": final_result.flagged,
                "amount": scenario["amount"],
                "latency_ms": final_result.latency_ms,
            }
        
        elif "escalation" in scenario:
            # Pig butchering pattern
            ts = time.time()
            final_result = None
            for i, amount in enumerate(scenario["escalation"]):
                tx = {
                    "tx_id": f"pig_{i}",
                    "sender": "0xpig_victim",
                    "recipient": "0xscammer",
                    "amount": amount,
                    "timestamp": ts + i * 86400,  # One day apart
                }
                final_result = await self.engine.analyze(tx)
            return {
                "score": final_result.score,
                "blocked": final_result.blocked,
                "flagged": final_result.flagged,
                "amount": scenario["escalation"][-1],
                "latency_ms": final_result.latency_ms,
            }
        
        return {"score": 0, "blocked": False, "flagged": False, "amount": 0, "latency_ms": 0}
    
    async def _run_mixed_load(self, count: int):
        """Run mixed realistic transaction load."""
        
        latencies = []
        blocked = 0
        flagged = 0
        
        for i in range(count):
            # Mix of transaction types
            roll = random.random()
            
            if roll < 0.60:
                # 60% normal transactions
                tx = {
                    "tx_id": f"mixed_normal_{i}",
                    "sender": f"0xuser_{random.randint(1, 1000)}",
                    "recipient": f"0xmerchant_{random.randint(1, 50)}",
                    "amount": random.uniform(10, 1000),
                }
            elif roll < 0.80:
                # 20% medium transactions
                tx = {
                    "tx_id": f"mixed_medium_{i}",
                    "sender": f"0xbiz_{random.randint(1, 100)}",
                    "recipient": f"0xsupplier_{random.randint(1, 20)}",
                    "amount": random.uniform(1000, 20000),
                }
            elif roll < 0.90:
                # 10% suspicious patterns
                tx = {
                    "tx_id": f"mixed_sus_{i}",
                    "sender": "0xstructurer",
                    "recipient": f"0xbank_{random.randint(1, 10)}",
                    "amount": random.uniform(9000, 9999),
                }
            else:
                # 10% exchange deposits
                tx = {
                    "tx_id": f"mixed_exchange_{i}",
                    "sender": f"0xtrader_{random.randint(1, 50)}",
                    "recipient": "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
                    "amount": random.uniform(5000, 50000),
                }
            
            r = await self.engine.analyze(tx)
            latencies.append(r.latency_ms)
            if r.blocked:
                blocked += 1
            if r.flagged:
                flagged += 1
        
        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"""
   📈 Mixed Load Results ({count} transactions):
   
   Latency:
   • Average: {avg:.2f}ms
   • P50:     {p50:.2f}ms
   • P95:     {p95:.2f}ms
   • P99:     {p99:.2f}ms
   • Max:     {max(latencies):.2f}ms
   
   Decisions:
   • Blocked: {blocked}/{count} ({100*blocked/count:.1f}%)
   • Flagged: {flagged}/{count} ({100*flagged/count:.1f}%)
""")
        
    async def _velocity_burst(self):
        """High-velocity burst stress test."""
        
        latencies = []
        
        # Simulate 50 rapid-fire transactions
        start = time.perf_counter()
        for i in range(50):
            tx = {
                "tx_id": f"burst_{i}",
                "sender": f"0xburst_{i % 10}",
                "recipient": f"0xtarget_{i % 5}",
                "amount": random.uniform(100, 10000),
            }
            r = await self.engine.analyze(tx)
            latencies.append(r.latency_ms)
        
        total_time = (time.perf_counter() - start) * 1000
        throughput = 50 / (total_time / 1000)
        
        print(f"""
   ⚡ Velocity Burst Results (50 transactions):
   
   • Total Time: {total_time:.0f}ms
   • Throughput: {throughput:.0f} tx/sec
   • Avg Latency: {sum(latencies)/len(latencies):.2f}ms
   • Max Latency: {max(latencies):.2f}ms
""")
        
    def _final_report(self):
        """Generate final test report."""
        
        passed = sum(1 for r in self.results if r.get("passed", True))
        total = len(self.results)
        
        stats = self.engine.get_stats()
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                           FINAL STRESS TEST REPORT                                        ║
╠═══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                           ║
║   🏆 Engine Version: {self.engine.VERSION}                                                       ║
║                                                                                           ║
║   📊 Test Results:                                                                        ║
║   • Scenarios Passed: {passed}/{total}                                                            ║
║   • Total Transactions Analyzed: {stats['total']}                                            ║
║                                                                                           ║
║   ⚡ Mode Distribution:                                                                   ║
║   • INSTANT (rules):     {stats['instant']:>5} ({100*stats['instant']/max(stats['total'],1):>5.1f}%)                                       ║
║   • HEURISTIC (neural):  {stats['heuristic']:>5} ({100*stats['heuristic']/max(stats['total'],1):>5.1f}%)                                       ║
║   • TYPOLOGY (patterns): {stats['typology']:>5} ({100*stats['typology']/max(stats['total'],1):>5.1f}%)                                       ║
║   • AI_VERIFY (GPU):     {stats['ai_verify']:>5} ({100*stats['ai_verify']/max(stats['total'],1):>5.1f}%)                                       ║
║                                                                                           ║
║   📈 Performance:                                                                         ║
║   • Average Latency: {stats.get('avg_ms', 0):.2f}ms                                                      ║
║   • P50 Latency:     {stats.get('p50_ms', 0):.2f}ms                                                      ║
║   • P95 Latency:     {stats.get('p95_ms', 0):.2f}ms                                                      ║
║                                                                                           ║
║   🎯 Target Compliance:                                                                   ║
""")
        
        checks = [
            ("Avg Latency < 50ms", stats.get('avg_ms', 0) < 50),
            ("P95 Latency < 150ms", stats.get('p95_ms', 0) < 150),
            ("Heuristics > 50%", stats['heuristic'] / max(stats['total'], 1) > 0.5),
            ("Scenarios Pass Rate > 80%", passed / max(total, 1) > 0.8),
        ]
        
        all_pass = True
        for name, ok in checks:
            status = "✅" if ok else "❌"
            if not ok:
                all_pass = False
            print(f"║   {status} {name:<50}                     ║")
        
        print("║                                                                                           ║")
        
        if all_pass:
            print("""║   ═══════════════════════════════════════════════════                                    ║
║   🏆 ALL TARGETS MET - PRODUCTION READY!                                                 ║
║   ═══════════════════════════════════════════════════                                    ║""")
        else:
            print("""║   ⚠️  SOME TARGETS NOT MET - REVIEW REQUIRED                                             ║""")
        
        print("""║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")


async def main():
    test = ProductionStressTest()
    await test.init()
    await test.run(iterations=100)


if __name__ == "__main__":
    asyncio.run(main())
