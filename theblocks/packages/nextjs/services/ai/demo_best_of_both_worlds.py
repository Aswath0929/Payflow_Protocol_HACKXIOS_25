#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                  PAYFLOW AI FRAUD DETECTION - BEST OF BOTH WORLDS DEMO               ║
║                                                                                       ║
║   This demo showcases the 3-layer ensemble architecture:                             ║
║   • Layer 1: Local Neural Network (50% weight) - 100% OFFLINE                        ║
║   • Layer 2: Traditional ML (30% weight) - Isolation Forest, DBSCAN                  ║
║   • Layer 3: GPT-4 Enhancement (20% weight) - OPTIONAL, API-based                    ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
from typing import Dict, Any

# Import our AI systems
from localNeuralNetwork import LocalNeuralNetworkEngine, NeuralNetworkPrediction
from secureAIOracle import SecureAIOracle, Config

def print_header():
    print("\n" + "═" * 80)
    print("║" + " " * 20 + "PAYFLOW AI FRAUD DETECTION DEMO" + " " * 26 + "║")
    print("║" + " " * 15 + "Best of Both Worlds: Neural Network + GPT-4" + " " * 17 + "║")
    print("═" * 80 + "\n")

def print_section(title: str):
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}\n")

def print_score_bar(label: str, score: int, max_score: int = 100):
    """Print a visual score bar."""
    bar_length = 40
    filled = int((score / max_score) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    if score <= 20:
        color = "\033[92m"  # Green
    elif score <= 40:
        color = "\033[94m"  # Blue
    elif score <= 60:
        color = "\033[93m"  # Yellow
    elif score <= 80:
        color = "\033[91m"  # Light Red
    else:
        color = "\033[31m"  # Red
    
    reset = "\033[0m"
    print(f"  {label:20s} {color}[{bar}] {score:3d}{reset}")

async def demo_neural_network_standalone():
    """Demo the neural network in standalone mode."""
    print_section("LAYER 1: LOCAL NEURAL NETWORK (100% Offline)")
    
    print("  Initializing Multi-Layer Perceptron + Autoencoder ensemble...")
    start = time.time()
    nn = LocalNeuralNetworkEngine()
    init_time = (time.time() - start) * 1000
    print(f"  ✓ Neural Network initialized in {init_time:.1f}ms")
    print(f"  ✓ Architecture: 13→64→32→16→4 (MLP) + 13→8→4→8→13 (Autoencoder)")
    print(f"  ✓ Self-trained on 600 synthetic samples")
    
    # Test cases
    test_cases = [
        {
            "name": "Normal Transaction",
            "sender": "0xAlice1234",
            "recipient": "0xBob5678",
            "amount": 500.0
        },
        {
            "name": "Suspicious Amount (Near Threshold)",
            "sender": "0xSuspect1234",
            "recipient": "0xTarget5678",
            "amount": 9999.0  # Just under $10k reporting threshold
        },
        {
            "name": "High-Value Transaction",
            "sender": "0xWhale1234",
            "recipient": "0xExchange5678",
            "amount": 500000.0
        }
    ]
    
    print("\n  Testing transactions:\n")
    
    for tc in test_cases:
        result = nn.predict(tc["sender"], tc["recipient"], tc["amount"])
        
        print(f"  📊 {tc['name']} (${tc['amount']:,.0f})")
        print_score_bar("Neural Net Score", result.risk_score)
        print(f"      Risk Level: {result.risk_level} | Confidence: {result.confidence:.0%}")
        print(f"      Anomaly: {'⚠️ YES' if result.is_anomaly else '✓ NO'}")
        if result.flags:
            print(f"      Flags: {', '.join(result.flags[:2])}")
        print()

async def demo_full_oracle():
    """Demo the full SecureAIOracle with all layers."""
    print_section("FULL 3-LAYER ENSEMBLE (Neural Network + ML + GPT-4)")
    
    print("  Initializing SecureAIOracle...")
    oracle = SecureAIOracle()
    
    print(f"  ✓ Model Version: {oracle.MODEL_VERSION}")
    print(f"  ✓ Neural Network: ACTIVE")
    print(f"  ✓ Traditional ML: ACTIVE (Isolation Forest, DBSCAN)")
    print(f"  ✓ GPT-4 Enhancement: {'ACTIVE' if oracle.gpt4.enabled else 'DISABLED (no API key)'}")
    print(f"  ✓ Oracle Signer: {oracle.signer.address[:20]}...")
    
    print("\n  Weight Distribution:")
    print(f"    • Neural Network: 50% (Primary - always runs offline)")
    print(f"    • Traditional ML:  30% (Secondary - statistical analysis)")
    print(f"    • GPT-4:          20% (Enhancement - when available)")
    
    print("\n  Analyzing test transaction...\n")
    
    # Analyze a transaction
    start = time.time()
    result = await oracle.analyze_transaction(
        transaction_id="demo_tx_001",
        sender="0xAlice1234567890abcdef",
        recipient="0xBob1234567890abcdef",
        amount=7500.0,
        timestamp=int(time.time())
    )
    analysis_time = (time.time() - start) * 1000
    
    print(f"  📊 Transaction Analysis Complete ({analysis_time:.1f}ms)\n")
    
    print("  NEURAL NETWORK LAYER (Primary):")
    print_score_bar("NN Risk Score", result.neural_net_score)
    print(f"      Level: {result.neural_net_risk_level} | Confidence: {result.neural_net_confidence:.0%}")
    
    print("\n  TRADITIONAL ML LAYER:")
    print_score_bar("Velocity", result.velocity_score)
    print_score_bar("Amount", result.amount_score)
    print_score_bar("Pattern", result.pattern_score)
    print_score_bar("Graph", result.graph_score)
    print_score_bar("Timing", result.timing_score)
    
    print("\n  GPT-4 LAYER:")
    if oracle.gpt4.enabled:
        print_score_bar("AI Score", result.ai_score)
    else:
        print("      ⚠️ GPT-4 disabled - add OPENAI_API_KEY to enable")
    
    print("\n  FINAL COMBINED RESULT:")
    print_score_bar("OVERALL SCORE", result.overall_score)
    print(f"      Risk Level: {result.risk_level}")
    print(f"      Approved: {'✓ YES' if result.approved else '✗ NO'}")
    print(f"      Flagged: {'⚠️ YES' if result.flagged else 'NO'}")
    print(f"      Blocked: {'🚫 YES' if result.blocked else 'NO'}")
    
    print("\n  🔐 CRYPTOGRAPHIC SIGNATURE:")
    print(f"      Signed by: {result.signer_address}")
    print(f"      Signature: {result.signature[:40]}...")
    print(f"      (Verifiable on-chain via SecureFraudOracle contract)")

async def demo_rapid_transactions():
    """Demo detection of rapid/suspicious transaction patterns."""
    print_section("PATTERN DETECTION: Rapid Transaction Simulation")
    
    nn = LocalNeuralNetworkEngine()
    
    print("  Simulating 30 rapid transactions from same sender...")
    print("  (This tests velocity anomaly detection)\n")
    
    sender = "0xRapidSender123"
    
    for i in range(30):
        result = nn.predict(sender, f"0xRecipient{i}", 100.0 + i * 10)
    
    # Final transaction should show elevated risk
    final = nn.predict(sender, "0xFinalRecipient", 500.0)
    
    print(f"  After 30 rapid transactions:")
    print_score_bar("Risk Score", final.risk_score)
    print(f"      Level: {final.risk_level}")
    print(f"      Anomaly Detected: {'⚠️ YES' if final.is_anomaly else '✓ NO'}")
    if final.flags:
        print(f"      Flags:")
        for flag in final.flags:
            print(f"        • {flag}")

def print_summary():
    print_section("ARCHITECTURE SUMMARY")
    
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    BEST OF BOTH WORLDS                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  🧠 NEURAL NETWORK (50%)                                      │  │
  │  │  • MLP + Autoencoder ensemble                                 │  │
  │  │  • 100% OFFLINE - No internet needed                         │  │
  │  │  • Self-training on transaction patterns                      │  │
  │  │  • <10ms inference time                                       │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                              ▼                                      │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  📊 TRADITIONAL ML (30%)                                      │  │
  │  │  • Isolation Forest (statistical outliers)                    │  │
  │  │  • DBSCAN (behavioral clustering)                            │  │
  │  │  • Velocity/Amount/Pattern/Graph/Timing analysis              │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                              ▼                                      │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  🤖 GPT-4 ENHANCEMENT (20%) - OPTIONAL                        │  │
  │  │  • Context-aware fraud reasoning                              │  │
  │  │  • Natural language explanations                              │  │
  │  │  • Called for: anomalies, high-value, flagged transactions   │  │
  │  │  • Enable with OPENAI_API_KEY                                 │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                              ▼                                      │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  🔐 ECDSA SIGNATURE → BLOCKCHAIN VERIFICATION                 │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
    """)

async def main():
    print_header()
    
    await demo_neural_network_standalone()
    await demo_full_oracle()
    await demo_rapid_transactions()
    print_summary()
    
    print("\n" + "═" * 80)
    print("║" + " " * 25 + "DEMO COMPLETE" + " " * 40 + "║")
    print("║" + " " * 10 + "System ready for Hackxios 2K25 demonstration!" + " " * 18 + "║")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
