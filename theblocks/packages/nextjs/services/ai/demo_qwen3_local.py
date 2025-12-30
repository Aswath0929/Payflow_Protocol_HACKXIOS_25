#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW QWEN3 LOCAL LLM DEMO                                      ║
║                                                                                       ║
║   100% LOCAL AI Fraud Detection - No Cloud API Required!                             ║
║                                                                                       ║
║   Running on:                                                                         ║
║   • Qwen3:8b (Latest 2025 Alibaba model)                                             ║
║   • RTX 4070 GPU (8GB VRAM)                                                          ║
║   • 100% Offline - Your data never leaves your machine                               ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import sys

# Add the current directory to path for imports
sys.path.insert(0, '.')

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║   ██████╗ ██╗   ██╗███████╗███╗   ██╗██████╗     ██╗     ██╗     ███╗   ███╗          ║
║  ██╔═══██╗██║   ██║██╔════╝████╗  ██║╚════██╗    ██║     ██║     ████╗ ████║          ║
║  ██║   ██║██║   ██║█████╗  ██╔██╗ ██║ █████╔╝    ██║     ██║     ██╔████╔██║          ║
║  ██║▄▄ ██║██║   ██║██╔══╝  ██║╚██╗██║ ╚═══██╗    ██║     ██║     ██║╚██╔╝██║          ║
║  ╚██████╔╝╚██████╔╝███████╗██║ ╚████║██████╔╝    ███████╗███████╗██║ ╚═╝ ██║          ║
║   ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝     ╚══════╝╚══════╝╚═╝     ╚═╝          ║
║                                                                                       ║
║   🧠 LOCAL FRAUD DETECTION - 100% OFFLINE ON RTX 4070                                 ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
""")

    # Import our modules
    print("Loading AI modules...")
    from localLLMAnalyzer import Qwen3LocalAnalyzer, check_ollama_status
    from localNeuralNetwork import LocalNeuralNetworkEngine
    
    # Check Ollama status
    print("\n1. Checking Qwen3 Local LLM status...")
    status = await check_ollama_status()
    
    if not status["running"]:
        print("   ❌ Ollama is not running! Start with: ollama serve")
        return
    
    print(f"   ✅ Ollama server: Running")
    print(f"   ✅ Available models: {status['models']}")
    print(f"   ✅ Qwen3 available: {status['qwen3_available']}")
    
    if not status["qwen3_available"]:
        print("\n   ❌ Qwen3 not found. Pull with: ollama pull qwen3:8b")
        return
    
    # Initialize analyzers
    print("\n2. Initializing AI Engines...")
    
    # Neural Network
    print("   Loading Neural Network (Pure NumPy)...")
    neural_net = LocalNeuralNetworkEngine()
    print("   ✅ Neural Network ready!")
    
    # Qwen3 LLM
    print("   Loading Qwen3 Local LLM (RTX 4070 GPU)...")
    qwen3 = Qwen3LocalAnalyzer(model_name="qwen3:8b")
    available = await qwen3.check_availability()
    
    if not available:
        print("   ❌ Qwen3 model not available")
        return
    
    print("   Warming up Qwen3 (loading to GPU)...")
    start = time.time()
    await qwen3.warm_up()
    load_time = time.time() - start
    print(f"   ✅ Qwen3 loaded to GPU in {load_time:.1f}s")
    
    # Test transactions
    print("\n" + "═" * 75)
    print("                        FRAUD DETECTION TESTS")
    print("═" * 75)
    
    test_cases = [
        {
            "name": "Normal Transaction",
            "tx_id": "tx_normal_001",
            "sender": "0xAlice123456789012345678901234567890123456",
            "recipient": "0xBob123456789012345678901234567890123456",
            "amount": 250.0,
            "sender_profile": {"transaction_count": 100, "total_volume": 50000, "avg_amount": 500},
            "recipient_profile": {"transaction_count": 50, "total_volume": 25000},
        },
        {
            "name": "Structuring Attempt ($9,999 - just under $10K)",
            "tx_id": "tx_structuring_001",
            "sender": "0xSuspect12345678901234567890123456789012",
            "recipient": "0xShell123456789012345678901234567890123",
            "amount": 9999.0,
            "sender_profile": {"transaction_count": 5, "total_volume": 49995, "avg_amount": 9999},
            "recipient_profile": {"transaction_count": 0, "total_volume": 0},
        },
        {
            "name": "High-Risk Offshore Transaction",
            "tx_id": "tx_offshore_001",
            "sender": "0xNewUser12345678901234567890123456789012",
            "recipient": "0xOffshore1234567890123456789012345678901",
            "amount": 49999.0,
            "sender_profile": {"transaction_count": 1, "total_volume": 50000, "avg_amount": 50000},
            "recipient_profile": {"transaction_count": 500, "total_volume": 10000000},
        },
        {
            "name": "Rapid Velocity Attack",
            "tx_id": "tx_velocity_001",
            "sender": "0xBot1234567890123456789012345678901234567",
            "recipient": "0xMixer12345678901234567890123456789012345",
            "amount": 100.0,
            "sender_profile": {"transaction_count": 200, "total_volume": 20000, "avg_amount": 100},
            "recipient_profile": {"transaction_count": 10000, "total_volume": 1000000},
        },
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 75}")
        print(f"TEST {i}: {tc['name']}")
        print(f"{'─' * 75}")
        print(f"  Amount: ${tc['amount']:,.2f}")
        print(f"  Sender TX Count: {tc['sender_profile']['transaction_count']}")
        print(f"  Recipient TX Count: {tc['recipient_profile']['transaction_count']}")
        
        # Phase 1: Neural Network (Fast)
        print("\n  📊 Phase 1: Neural Network (Pure NumPy)...")
        nn_start = time.time()
        nn_result = neural_net.predict(
            sender=tc["sender"],
            recipient=tc["recipient"],
            amount=tc["amount"],
            timestamp=int(time.time())
        )
        nn_time = (time.time() - nn_start) * 1000
        
        print(f"     Risk Score: {nn_result.risk_score}/100")
        print(f"     Risk Level: {nn_result.risk_level}")
        print(f"     Is Anomaly: {'⚠️ YES' if nn_result.is_anomaly else '✅ NO'}")
        print(f"     Inference: {nn_time:.1f}ms")
        
        # Phase 2: Qwen3 LLM (Deep Analysis)
        print("\n  🧠 Phase 2: Qwen3 Local LLM (RTX 4070 GPU)...")
        
        local_scores = {
            "velocity": 30 if tc['sender_profile']['transaction_count'] > 100 else 10,
            "amount": 50 if tc['amount'] > 9000 else 10,
            "pattern": 40 if tc['recipient_profile']['transaction_count'] == 0 else 10,
            "graph": 20,
            "timing": 15,
            "isolation_forest": 0.3
        }
        
        llm_start = time.time()
        llm_result = await qwen3.analyze(
            transaction_id=tc["tx_id"],
            sender=tc["sender"],
            recipient=tc["recipient"],
            amount=tc["amount"],
            sender_profile=tc["sender_profile"],
            recipient_profile=tc["recipient_profile"],
            local_scores=local_scores,
            neural_net_result={
                "risk_score": nn_result.risk_score,
                "risk_level": nn_result.risk_level,
                "is_anomaly": nn_result.is_anomaly,
                "flags": nn_result.flags
            }
        )
        llm_time = (time.time() - llm_start) * 1000
        
        print(f"     Risk Score: {llm_result.risk_score}/100")
        print(f"     Risk Level: {llm_result.risk_level}")
        print(f"     Approved: {'✅ YES' if llm_result.approved else '❌ BLOCKED'}")
        print(f"     Confidence: {llm_result.confidence:.0%}")
        print(f"     Inference: {llm_time:.0f}ms")
        print(f"     Model: {llm_result.model_used}")
        print(f"     Local: {'✅ 100% Offline' if llm_result.is_local else '☁️ Cloud'}")
        
        if llm_result.explanation:
            print(f"\n     💬 Explanation:")
            # Wrap text nicely
            exp = llm_result.explanation
            for line in [exp[i:i+60] for i in range(0, len(exp), 60)]:
                print(f"        {line}")
        
        if llm_result.alerts:
            print(f"\n     ⚠️ Alerts:")
            for alert in llm_result.alerts[:3]:
                print(f"        • {alert[:70]}")
        
        # Combined result
        print(f"\n  📊 COMBINED RESULT:")
        combined_score = int(nn_result.risk_score * 0.4 + llm_result.risk_score * 0.6)
        print(f"     Neural Net (40%): {nn_result.risk_score}")
        print(f"     Qwen3 LLM (60%):  {llm_result.risk_score}")
        print(f"     FINAL SCORE:      {combined_score}/100")
        print(f"     Total Time:       {nn_time + llm_time:.0f}ms")
    
    # Summary
    print("\n" + "═" * 75)
    print("                          DEMO COMPLETE!")
    print("═" * 75)
    print("""
    🎉 PayFlow Qwen3 Local LLM is fully operational!
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Layer 1: Neural Network (NumPy)     - 40% weight, <5ms        │
    │  Layer 2: Qwen3 Local LLM (GPU)      - 60% weight, <500ms      │
    │  Layer 3: Cryptographic Signing      - ECDSA for on-chain      │
    └─────────────────────────────────────────────────────────────────┘
    
    Benefits:
    ✅ 100% LOCAL - No data leaves your machine
    ✅ No API keys or cloud costs
    ✅ Real-time inference on RTX 4070
    ✅ Advanced MoE reasoning with Qwen3
    ✅ Cryptographic proof of all decisions
    
    Ready for Hackxios 2K25! 🚀
""")
    
    await qwen3.close()

if __name__ == "__main__":
    asyncio.run(main())
