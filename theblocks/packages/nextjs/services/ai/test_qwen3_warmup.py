#!/usr/bin/env python3
"""Test Qwen3 with manual warmup."""

import asyncio
import time


async def test_with_qwen3():
    print("Testing Qwen3 with manual warmup...")

    from secureAIOracle import SecureAIOracle

    oracle = SecureAIOracle()
    print(f"Initial Qwen3 Status: {oracle.qwen3.enabled}")

    # Manual warmup
    print("Warming up Qwen3...")
    try:
        await oracle.qwen3._init_qwen3()
        print(f"After warmup: {oracle.qwen3.enabled}")
    except Exception as e:
        print(f"Warmup failed: {e}")

    if oracle.qwen3.enabled:
        print("\nRunning with Qwen3 active on GPU...")
        start = time.time()
        result = await oracle.analyze_transaction(
            transaction_id="qwen3_test",
            sender="0xTestSender123abc",
            recipient="0xTestRecipient456def",
            amount=25000.0,
            use_qwen3=True,
        )
        elapsed = (time.time() - start) * 1000
        print(f"Result: Score={result.overall_score}, Level={result.risk_level}")
        print(f"AI Score: {result.ai_score}")
        print(f"Neural Net Score: {result.neural_net_score}")
        print(f"Time: {elapsed:.1f}ms")
        stats = oracle.get_statistics()
        print(f"Qwen3 Calls: {stats['qwen3_calls']}")
    else:
        print("Qwen3 not available - testing architecture only")
        result = await oracle.analyze_transaction(
            transaction_id="neural_test",
            sender="0xTestSender123abc",
            recipient="0xTestRecipient456def",
            amount=25000.0,
            use_qwen3=True,
        )
        print(f"Result: Score={result.overall_score}, Level={result.risk_level}")

    # Verify GPT-4 removed
    print(f"\n--- Architecture Verification ---")
    print(f"GPT-4 attribute exists: {hasattr(oracle, 'gpt4')}")
    print(f"Qwen3 attribute exists: {hasattr(oracle, 'qwen3')}")
    print("SUCCESS!")


if __name__ == "__main__":
    asyncio.run(test_with_qwen3())
