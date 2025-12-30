"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                   ULTIMATE FRAUD ENGINE - COMPREHENSIVE TEST SUITE                        ║
║                                                                                           ║
║   🧪 15 FRAUD TYPOLOGY TESTS WITH ACCURACY VALIDATION                                    ║
║   ⚡ PERFORMANCE BENCHMARKING FOR <300ms TARGET                                          ║
║   🎯 PRECISION & RECALL METRICS                                                          ║
║   🔥 FULL GPU STRESS TESTING                                                             ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import json
import random
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
from colorama import Fore, Style, init
import numpy as np

# Initialize colorama for Windows
init()

# Import our engine
from ultimateFraudEngine import (
    UltimateFraudEngine, 
    Transaction, 
    FraudAnalysis, 
    FraudType, 
    RiskLevel,
    GPUConfig
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestConfig:
    """Test configuration parameters."""
    # Performance targets
    TARGET_LATENCY_MS = 300  # <300ms target
    MAX_LATENCY_MS = 500     # Max acceptable
    
    # Accuracy targets (by typology)
    MIN_DETECTION_RATE = 0.88  # Minimum 88% detection
    MAX_FALSE_POSITIVE = 0.05   # Maximum 5% false positives
    
    # Test volume
    TRANSACTIONS_PER_TYPOLOGY = 10
    LEGITIMATE_TRANSACTIONS = 50
    STRESS_TEST_VOLUME = 100
    
    # Randomization
    RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#                         TEST TRANSACTION GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionGenerator:
    """Generates test transactions for each fraud typology."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.tx_counter = 0
    
    def _gen_address(self) -> str:
        return "0x" + hashlib.md5(str(random.random()).encode()).hexdigest()[:40]
    
    def _gen_tx_id(self) -> str:
        self.tx_counter += 1
        return f"tx_{self.tx_counter:05d}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                     LEGITIMATE TRANSACTIONS (NEGATIVES)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def legitimate_small(self) -> Transaction:
        """Normal small transaction."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(10, 500),
            sender_history_count=random.randint(10, 100),
            sender_avg_amount=random.uniform(100, 300),
            sender_age_days=random.uniform(30, 365),
            recipient_history_count=random.randint(5, 50),
            recipient_age_days=random.uniform(30, 365),
        )
    
    def legitimate_medium(self) -> Transaction:
        """Normal medium transaction."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(500, 5000),
            sender_history_count=random.randint(50, 500),
            sender_avg_amount=random.uniform(1000, 3000),
            sender_age_days=random.uniform(180, 730),
            recipient_history_count=random.randint(20, 200),
            recipient_age_days=random.uniform(90, 365),
        )
    
    def legitimate_large(self) -> Transaction:
        """Normal large transaction from established wallet."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(10000, 100000),
            sender_history_count=random.randint(200, 1000),
            sender_avg_amount=random.uniform(20000, 50000),
            sender_age_days=random.uniform(365, 1000),
            recipient_history_count=random.randint(100, 500),
            recipient_age_days=random.uniform(180, 730),
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                     FRAUD TYPOLOGY GENERATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def rug_pull(self) -> Tuple[Transaction, FraudType]:
        """Rug pull: Large outflow from new contract."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(100000, 5000000),
            sender_history_count=random.randint(1, 3),
            sender_avg_amount=0,
            sender_age_days=random.uniform(1, 30),
            recipient_history_count=random.randint(0, 2),
            recipient_age_days=random.uniform(0, 7),
            is_contract_recipient=False,
        ), FraudType.RUG_PULL
    
    def pig_butchering(self) -> Tuple[Transaction, FraudType]:
        """Pig butchering: Gradual trust-building, large final transfer."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(50000, 500000),
            sender_history_count=random.randint(15, 30),  # Multiple prior transactions
            sender_avg_amount=random.uniform(1000, 5000),  # Much smaller previously
            sender_age_days=random.uniform(30, 90),
            recipient_history_count=random.randint(100, 500),  # Established recipient
            recipient_age_days=random.uniform(180, 365),
        ), FraudType.PIG_BUTCHERING
    
    def mixer_tumbling(self) -> Tuple[Transaction, FraudType]:
        """Mixer/Tumbling: Transfer to known mixer."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient="0x8589427373d6d84e98730d7795d8f6f8731fda16",  # Tornado Cash
            amount=random.uniform(1000, 50000),
            sender_history_count=random.randint(5, 20),
            sender_avg_amount=random.uniform(5000, 20000),
            sender_age_days=random.uniform(30, 180),
            recipient_history_count=99999,  # Known mixer has many txs
            recipient_age_days=1000,
        ), FraudType.MIXER_TUMBLING
    
    def chain_obfuscation(self) -> Tuple[Transaction, FraudType]:
        """Chain obfuscation: Bridge usage pattern."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),  # Bridge contract
            amount=random.uniform(10000, 100000),
            sender_history_count=random.randint(2, 5),
            sender_avg_amount=0,
            sender_age_days=random.uniform(1, 14),
            recipient_history_count=1,
            recipient_age_days=1,
            is_contract_recipient=True,
        ), FraudType.CHAIN_OBFUSCATION
    
    def fake_token(self) -> Tuple[Transaction, FraudType]:
        """Fake token: Honeypot contract interaction."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(500, 10000),
            token="FAKE_SHIBA_3000X",  # Suspicious token name
            sender_history_count=random.randint(0, 2),
            sender_avg_amount=0,
            sender_age_days=random.uniform(0, 3),
            recipient_history_count=0,
            recipient_age_days=0,
            is_contract_recipient=True,
        ), FraudType.FAKE_TOKEN
    
    def flash_loan(self) -> Tuple[Transaction, FraudType]:
        """Flash loan: Large borrow in same block."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(1000000, 50000000),  # Very large
            sender_history_count=random.randint(0, 1),  # No prior history
            sender_avg_amount=0,
            sender_age_days=0,  # Same-block
            recipient_history_count=0,
            recipient_age_days=0,
            time_since_last_tx=0.001,  # Almost instant
        ), FraudType.FLASH_LOAN
    
    def wash_trading(self) -> Tuple[Transaction, FraudType]:
        """Wash trading: Self-dealing pattern."""
        addr = self._gen_address()
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=addr,
            recipient=addr,  # Same address
            amount=random.uniform(1000, 50000),
            sender_history_count=random.randint(50, 200),
            sender_avg_amount=random.uniform(2000, 10000),
            sender_age_days=random.uniform(7, 30),
            recipient_history_count=random.randint(50, 200),
            recipient_age_days=random.uniform(7, 30),
        ), FraudType.WASH_TRADING
    
    def structuring(self) -> Tuple[Transaction, FraudType]:
        """Structuring: Amount near reporting threshold."""
        threshold = random.choice([9900, 9950, 9990, 49900, 49950])
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=threshold + random.uniform(-50, 50),
            sender_history_count=random.randint(10, 50),
            sender_avg_amount=random.uniform(2000, 5000),
            sender_age_days=random.uniform(60, 180),
            recipient_history_count=random.randint(5, 20),
            recipient_age_days=random.uniform(30, 90),
        ), FraudType.STRUCTURING
    
    def velocity_attack(self) -> Tuple[Transaction, FraudType]:
        """Velocity attack: Burst of transactions."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(500, 5000),
            sender_history_count=random.randint(50, 100),  # Many txs
            sender_avg_amount=random.uniform(500, 2000),
            sender_age_days=random.uniform(1, 7),  # Recent account
            recipient_history_count=random.randint(0, 5),
            recipient_age_days=random.uniform(0, 1),
            time_since_last_tx=0.05,  # 3 minutes
        ), FraudType.VELOCITY_ATTACK
    
    def peel_chain(self) -> Tuple[Transaction, FraudType]:
        """Peel chain: Gradual peeling to new addresses."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(100, 1000),  # Small peel
            sender_history_count=random.randint(20, 50),
            sender_avg_amount=random.uniform(200, 500),
            sender_age_days=random.uniform(1, 7),
            recipient_history_count=0,  # New address each time
            recipient_age_days=0,
        ), FraudType.PEEL_CHAIN
    
    def dusting(self) -> Tuple[Transaction, FraudType]:
        """Dusting: Tiny amount to many addresses."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(0.001, 0.1),  # Dust
            sender_history_count=random.randint(1000, 5000),  # Many txs
            sender_avg_amount=0.01,
            sender_age_days=random.uniform(30, 90),
            recipient_history_count=random.randint(10, 100),
            recipient_age_days=random.uniform(30, 365),
        ), FraudType.DUSTING
    
    def address_poisoning(self) -> Tuple[Transaction, FraudType]:
        """Address poisoning: Similar-looking address."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient="0x28c6c06298d514db089934071355e5743bf21d61",  # Similar to Binance
            amount=random.uniform(0, 0.01),  # Zero/dust
            sender_history_count=random.randint(100, 500),
            sender_avg_amount=0,
            sender_age_days=random.uniform(1, 7),
            recipient_history_count=0,
            recipient_age_days=0,
        ), FraudType.ADDRESS_POISONING
    
    def approval_exploit(self) -> Tuple[Transaction, FraudType]:
        """Approval exploit: Unlimited token approval."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=2**256 - 1,  # Max uint256 (unlimited approval)
            sender_history_count=random.randint(5, 20),
            sender_avg_amount=random.uniform(100, 1000),
            sender_age_days=random.uniform(30, 180),
            recipient_history_count=0,  # New contract
            recipient_age_days=0,
            is_contract_recipient=True,
        ), FraudType.APPROVAL_EXPLOIT
    
    def sim_swap(self) -> Tuple[Transaction, FraudType]:
        """SIM swap: Account recovery after compromise."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(10000, 100000),  # Large drain
            sender_history_count=random.randint(100, 500),  # Established account
            sender_avg_amount=random.uniform(500, 2000),
            sender_age_days=random.uniform(365, 1000),  # Old account
            recipient_history_count=0,  # New destination
            recipient_age_days=0,
            time_since_last_tx=random.uniform(720, 8760),  # 30 days to 1 year gap
        ), FraudType.SIM_SWAP
    
    def romance_scam(self) -> Tuple[Transaction, FraudType]:
        """Romance scam: Trust-based transfers."""
        return Transaction(
            tx_id=self._gen_tx_id(),
            sender=self._gen_address(),
            recipient=self._gen_address(),
            amount=random.uniform(5000, 50000),
            sender_history_count=random.randint(5, 15),  # Building trust
            sender_avg_amount=random.uniform(1000, 3000),  # Gradually increasing
            sender_age_days=random.uniform(90, 365),  # Victim account age
            recipient_history_count=random.randint(50, 200),  # Scammer has history
            recipient_age_days=random.uniform(60, 180),
        ), FraudType.ROMANCE_SCAM


# ═══════════════════════════════════════════════════════════════════════════════
#                           TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class ComprehensiveTestRunner:
    """Runs comprehensive tests on the fraud engine."""
    
    def __init__(self, config: TestConfig = TestConfig()):
        self.config = config
        self.generator = TransactionGenerator(config.RANDOM_SEED)
        self.engine: UltimateFraudEngine = None
        
        # Results tracking
        self.results: Dict[str, Dict] = {}
        self.latencies: List[float] = []
        self.true_positives: int = 0
        self.false_positives: int = 0
        self.true_negatives: int = 0
        self.false_negatives: int = 0
    
    def _print_header(self, title: str):
        """Print section header."""
        print(f"\n{Fore.CYAN}{'═' * 70}")
        print(f"{title:^70}")
        print(f"{'═' * 70}{Style.RESET_ALL}")
    
    def _print_result(self, label: str, value: str, color: str = Fore.WHITE):
        """Print formatted result."""
        print(f"  {color}• {label:.<45} {value}{Style.RESET_ALL}")
    
    async def run_all_tests(self):
        """Run complete test suite."""
        print(f"""
{Fore.YELLOW}
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║     ██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗                       ║
║     ██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝                       ║
║     ██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗                         ║
║     ██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝                         ║
║     ╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗                       ║
║      ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝                       ║
║                                                                                           ║
║     ███████╗██████╗  █████╗ ██╗   ██╗██████╗     ███████╗███╗   ██╗ ██████╗██╗███╗   ██╗║
║     ██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗    ██╔════╝████╗  ██║██╔════╝██║████╗  ██║║
║     █████╗  ██████╔╝███████║██║   ██║██║  ██║    █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║
║     ██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║    ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║
║     ██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║
║     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝     ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝
║                                                                                           ║
║                     COMPREHENSIVE TEST SUITE v3.0                                         ║
║                     PayFlow Protocol - Hackxios 2K25                                      ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        
        # Initialize engine
        self._print_header("🚀 INITIALIZING ENGINE")
        from ultimateFraudEngine import get_engine
        self.engine = await get_engine()
        print(f"  {Fore.GREEN}✓ Engine v{self.engine.VERSION} initialized{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}✓ GPU: {GPUConfig.GPU_LAYERS} layers, {GPUConfig.NUM_CTX} context{Style.RESET_ALL}")
        
        # Run test phases
        await self._test_legitimate_transactions()
        await self._test_all_typologies()
        await self._run_stress_test()
        self._print_final_report()
        
        await self.engine.close()
    
    async def _test_legitimate_transactions(self):
        """Test legitimate transactions (should not be flagged)."""
        self._print_header("🟢 PHASE 1: LEGITIMATE TRANSACTIONS (False Positive Check)")
        
        generators = [
            ("Small", self.generator.legitimate_small),
            ("Medium", self.generator.legitimate_medium),
            ("Large", self.generator.legitimate_large),
        ]
        
        for name, gen_func in generators:
            correct = 0
            total = self.config.LEGITIMATE_TRANSACTIONS // 3
            
            for _ in range(total):
                tx = gen_func()
                result = await self.engine.analyze(tx)
                self.latencies.append(result.analysis_time_ms)
                
                # Legitimate transactions should not be blocked
                if not result.blocked:
                    correct += 1
                    self.true_negatives += 1
                else:
                    self.false_positives += 1
            
            accuracy = correct / total * 100
            color = Fore.GREEN if accuracy >= 95 else Fore.YELLOW if accuracy >= 90 else Fore.RED
            self._print_result(f"{name} Transactions", f"{accuracy:.1f}% correct ({correct}/{total})", color)
    
    async def _test_all_typologies(self):
        """Test all 15 fraud typologies."""
        self._print_header("🔴 PHASE 2: FRAUD TYPOLOGY DETECTION (15 Types)")
        
        typology_generators = [
            ("Rug Pull", self.generator.rug_pull, FraudType.RUG_PULL, 96),
            ("Pig Butchering", self.generator.pig_butchering, FraudType.PIG_BUTCHERING, 94),
            ("Mixer/Tumbling", self.generator.mixer_tumbling, FraudType.MIXER_TUMBLING, 98),
            ("Chain Obfuscation", self.generator.chain_obfuscation, FraudType.CHAIN_OBFUSCATION, 93),
            ("Fake Token", self.generator.fake_token, FraudType.FAKE_TOKEN, 97),
            ("Flash Loan", self.generator.flash_loan, FraudType.FLASH_LOAN, 91),
            ("Wash Trading", self.generator.wash_trading, FraudType.WASH_TRADING, 95),
            ("Structuring", self.generator.structuring, FraudType.STRUCTURING, 99),
            ("Velocity Attack", self.generator.velocity_attack, FraudType.VELOCITY_ATTACK, 94),
            ("Peel Chain", self.generator.peel_chain, FraudType.PEEL_CHAIN, 92),
            ("Dusting", self.generator.dusting, FraudType.DUSTING, 96),
            ("Address Poisoning", self.generator.address_poisoning, FraudType.ADDRESS_POISONING, 97),
            ("Approval Exploit", self.generator.approval_exploit, FraudType.APPROVAL_EXPLOIT, 93),
            ("SIM Swap", self.generator.sim_swap, FraudType.SIM_SWAP, 89),
            ("Romance Scam", self.generator.romance_scam, FraudType.ROMANCE_SCAM, 88),
        ]
        
        for name, gen_func, fraud_type, target_rate in typology_generators:
            detected = 0
            flagged = 0
            total = self.config.TRANSACTIONS_PER_TYPOLOGY
            avg_score = 0
            
            for _ in range(total):
                tx, _ = gen_func()
                result = await self.engine.analyze(tx, force_thinking=True)
                self.latencies.append(result.analysis_time_ms)
                avg_score += result.overall_score
                
                # Check if fraud was detected (blocked or flagged)
                if result.blocked:
                    detected += 1
                    self.true_positives += 1
                elif result.flagged:
                    flagged += 1
                    self.true_positives += 1
                else:
                    self.false_negatives += 1
            
            detection_rate = (detected + flagged) / total * 100
            avg_score = avg_score / total
            color = Fore.GREEN if detection_rate >= target_rate else Fore.YELLOW if detection_rate >= 80 else Fore.RED
            
            self._print_result(
                f"{name} ({fraud_type.code})", 
                f"{detection_rate:.0f}% (B:{detected}/F:{flagged}) avg_score:{avg_score:.0f}",
                color
            )
            
            self.results[fraud_type.code] = {
                "name": name,
                "detected": detected,
                "flagged": flagged,
                "total": total,
                "detection_rate": detection_rate,
                "target_rate": target_rate,
                "avg_score": avg_score,
            }
    
    async def _run_stress_test(self):
        """Run stress test for performance."""
        self._print_header("⚡ PHASE 3: PERFORMANCE STRESS TEST")
        
        start_time = time.time()
        batch_latencies = []
        
        for i in range(self.config.STRESS_TEST_VOLUME):
            # Mix of legitimate and fraud
            if random.random() < 0.7:
                tx = self.generator.legitimate_medium()
            else:
                tx, _ = random.choice([
                    self.generator.structuring,
                    self.generator.velocity_attack,
                    self.generator.rug_pull,
                ])()
            
            result = await self.engine.analyze(tx)
            batch_latencies.append(result.analysis_time_ms)
            
            if (i + 1) % 20 == 0:
                avg = np.mean(batch_latencies[-20:])
                print(f"  Processed {i+1}/{self.config.STRESS_TEST_VOLUME} | Batch avg: {avg:.0f}ms")
        
        elapsed = time.time() - start_time
        
        self._print_result("Total Transactions", f"{self.config.STRESS_TEST_VOLUME}")
        self._print_result("Total Time", f"{elapsed:.2f}s")
        self._print_result("Throughput", f"{self.config.STRESS_TEST_VOLUME/elapsed:.1f} tx/sec")
        self._print_result("Average Latency", f"{np.mean(batch_latencies):.0f}ms")
        self._print_result("P95 Latency", f"{np.percentile(batch_latencies, 95):.0f}ms")
        self._print_result("P99 Latency", f"{np.percentile(batch_latencies, 99):.0f}ms")
        
        self.latencies.extend(batch_latencies)
    
    def _print_final_report(self):
        """Print comprehensive final report."""
        self._print_header("📊 FINAL COMPREHENSIVE REPORT")
        
        # Calculate metrics
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        accuracy = (self.true_positives + self.true_negatives) / max(1, total) * 100
        
        precision = self.true_positives / max(1, self.true_positives + self.false_positives)
        recall = self.true_positives / max(1, self.true_positives + self.false_negatives)
        f1_score = 2 * (precision * recall) / max(0.001, precision + recall)
        
        false_positive_rate = self.false_positives / max(1, self.false_positives + self.true_negatives) * 100
        false_negative_rate = self.false_negatives / max(1, self.false_negatives + self.true_positives) * 100
        
        avg_latency = np.mean(self.latencies)
        p95_latency = np.percentile(self.latencies, 95)
        p99_latency = np.percentile(self.latencies, 99)
        
        # Print metrics
        print(f"\n  {Fore.YELLOW}═══ ACCURACY METRICS ═══{Style.RESET_ALL}")
        self._print_result("Overall Accuracy", f"{accuracy:.2f}%", Fore.GREEN if accuracy >= 90 else Fore.YELLOW)
        self._print_result("Precision", f"{precision:.4f}", Fore.GREEN if precision >= 0.9 else Fore.YELLOW)
        self._print_result("Recall", f"{recall:.4f}", Fore.GREEN if recall >= 0.9 else Fore.YELLOW)
        self._print_result("F1 Score", f"{f1_score:.4f}", Fore.GREEN if f1_score >= 0.9 else Fore.YELLOW)
        self._print_result("False Positive Rate", f"{false_positive_rate:.2f}%", Fore.GREEN if false_positive_rate <= 5 else Fore.RED)
        self._print_result("False Negative Rate", f"{false_negative_rate:.2f}%", Fore.GREEN if false_negative_rate <= 5 else Fore.RED)
        
        print(f"\n  {Fore.YELLOW}═══ CONFUSION MATRIX ═══{Style.RESET_ALL}")
        self._print_result("True Positives (Fraud Caught)", str(self.true_positives), Fore.GREEN)
        self._print_result("True Negatives (Clean Approved)", str(self.true_negatives), Fore.GREEN)
        self._print_result("False Positives (Clean Blocked)", str(self.false_positives), Fore.RED)
        self._print_result("False Negatives (Fraud Missed)", str(self.false_negatives), Fore.RED)
        
        print(f"\n  {Fore.YELLOW}═══ PERFORMANCE METRICS ═══{Style.RESET_ALL}")
        latency_color = Fore.GREEN if avg_latency <= self.config.TARGET_LATENCY_MS else Fore.YELLOW if avg_latency <= self.config.MAX_LATENCY_MS else Fore.RED
        self._print_result("Average Latency", f"{avg_latency:.0f}ms (target: <{self.config.TARGET_LATENCY_MS}ms)", latency_color)
        self._print_result("P95 Latency", f"{p95_latency:.0f}ms", latency_color)
        self._print_result("P99 Latency", f"{p99_latency:.0f}ms", latency_color)
        self._print_result("Total Transactions Tested", str(len(self.latencies)))
        
        # Typology-specific results
        print(f"\n  {Fore.YELLOW}═══ TYPOLOGY DETECTION RATES ═══{Style.RESET_ALL}")
        for code, data in self.results.items():
            achieved = data["detection_rate"]
            target = data["target_rate"]
            status = "✅" if achieved >= target else "⚠️" if achieved >= target - 10 else "❌"
            color = Fore.GREEN if achieved >= target else Fore.YELLOW if achieved >= target - 10 else Fore.RED
            self._print_result(
                f"{status} {data['name']}", 
                f"{achieved:.0f}% (target: {target}%)",
                color
            )
        
        # Final verdict
        print(f"\n  {Fore.YELLOW}═══ FINAL VERDICT ═══{Style.RESET_ALL}")
        
        passes = 0
        checks = [
            ("Accuracy >= 90%", accuracy >= 90),
            ("F1 Score >= 0.85", f1_score >= 0.85),
            ("False Positive Rate <= 5%", false_positive_rate <= 5),
            ("Avg Latency <= 500ms", avg_latency <= 500),
        ]
        
        for check, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            color = Fore.GREEN if passed else Fore.RED
            self._print_result(check, status, color)
            if passed:
                passes += 1
        
        if passes == len(checks):
            print(f"""
{Fore.GREEN}
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   🎉 ALL TESTS PASSED - ENGINE READY FOR DEPLOYMENT 🚀           ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")
        else:
            print(f"""
{Fore.YELLOW}
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   ⚠️  {passes}/{len(checks)} CHECKS PASSED - REVIEW BEFORE DEPLOYMENT     ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{Fore.CYAN}Installing dependencies...{Style.RESET_ALL}")
    
    async def main():
        runner = ComprehensiveTestRunner()
        await runner.run_all_tests()
    
    asyncio.run(main())
