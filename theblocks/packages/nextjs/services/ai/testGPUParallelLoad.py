"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW PARALLEL GPU LOAD TEST                                    ║
║                                                                                       ║
║   Tests Qwen3:8B under dual workload stress:                                         ║
║   1. AI Chatbot (conversational, streaming)                                          ║
║   2. Fraud Detection (real-time, high priority)                                      ║
║                                                                                       ║
║   RTX 4070 (8GB VRAM) Performance Validation                                         ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import httpx
import time
import json
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any
import random
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:8b"

# GPU-optimized options (matching our config)
GPU_OPTIONS = {
    "num_gpu": 99,
    "num_ctx": 4096,
    "num_batch": 512,
    "num_thread": 8,
    "use_mmap": True,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# Test prompts
CHATBOT_PROMPTS = [
    "What is PayFlow Protocol and how does it work?",
    "Explain the benefits of stablecoin payments",
    "How does blockchain ensure transaction security?",
    "What is the difference between USDC and USDT?",
    "Can you explain smart contract escrow?",
]

FRAUD_PROMPTS = [
    """Analyze this transaction for fraud:
    - Amount: $50,000 USDC
    - Sender: 0xABC123...
    - Recipient: 0xDEF456...
    - Pattern: Large transfer to new wallet
    Respond with JSON: {"risk_score": 0-100, "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL"}""",
    
    """Detect fraud risk:
    - Amount: $1,500 USDC
    - Sender history: 50 transactions
    - Recipient: Known exchange
    - Time: Business hours
    Respond with JSON: {"risk_score": 0-100, "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL"}""",
]

# ═══════════════════════════════════════════════════════════════════════════════
#                              RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    request_type: str
    success: bool
    latency_ms: float
    tokens_generated: int = 0
    error: str = None


class TestStats:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def add(self, result: TestResult):
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results"}
        
        chat_results = [r for r in self.results if r.request_type == "chatbot"]
        fraud_results = [r for r in self.results if r.request_type == "fraud"]
        
        def stats_for(results: List[TestResult]) -> Dict:
            if not results:
                return {"count": 0}
            
            latencies = [r.latency_ms for r in results if r.success]
            success_count = len([r for r in results if r.success])
            
            return {
                "count": len(results),
                "success_count": success_count,
                "success_rate": f"{(success_count / len(results)) * 100:.1f}%",
                "min_latency_ms": round(min(latencies), 0) if latencies else 0,
                "max_latency_ms": round(max(latencies), 0) if latencies else 0,
                "avg_latency_ms": round(statistics.mean(latencies), 0) if latencies else 0,
                "median_latency_ms": round(statistics.median(latencies), 0) if latencies else 0,
            }
        
        return {
            "total_duration_s": round(time.time() - self.start_time, 1),
            "total_requests": len(self.results),
            "chatbot": stats_for(chat_results),
            "fraud_detection": stats_for(fraud_results),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def warm_up_model(client: httpx.AsyncClient) -> bool:
    """Load model to GPU VRAM."""
    print("\n🔥 Warming up Qwen3:8B on GPU...")
    
    try:
        start = time.time()
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": "Hello",
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    **GPU_OPTIONS,
                    "num_predict": 1
                }
            },
            timeout=60.0
        )
        
        if response.status_code == 200:
            elapsed = (time.time() - start) * 1000
            print(f"  ✅ Model loaded in {elapsed:.0f}ms")
            return True
        else:
            print(f"  ❌ Warm-up failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Warm-up error: {e}")
        return False


async def test_chatbot_request(
    client: httpx.AsyncClient,
    prompt: str,
    stats: TestStats
) -> TestResult:
    """Send a chatbot request."""
    start = time.time()
    
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    **GPU_OPTIONS,
                    "temperature": 0.7,  # More creative for chat
                    "num_predict": 128,  # REDUCED for faster test (was 256)
                    "num_ctx": 2048,     # Smaller context for parallel
                }
            },
            timeout=30.0
        )
        
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            tokens = data.get("eval_count", 0)
            result = TestResult(
                request_type="chatbot",
                success=True,
                latency_ms=latency,
                tokens_generated=tokens
            )
        else:
            result = TestResult(
                request_type="chatbot",
                success=False,
                latency_ms=latency,
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        result = TestResult(
            request_type="chatbot",
            success=False,
            latency_ms=(time.time() - start) * 1000,
            error=str(e)
        )
    
    stats.add(result)
    return result


async def test_fraud_request(
    client: httpx.AsyncClient,
    prompt: str,
    stats: TestStats
) -> TestResult:
    """Send a fraud detection request (high priority)."""
    start = time.time()
    
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "format": "json",
                "options": {
                    **GPU_OPTIONS,
                    "temperature": 0.3,  # More deterministic
                    "num_predict": 128,  # Short response
                }
            },
            timeout=10.0
        )
        
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            tokens = data.get("eval_count", 0)
            result = TestResult(
                request_type="fraud",
                success=True,
                latency_ms=latency,
                tokens_generated=tokens
            )
        else:
            result = TestResult(
                request_type="fraud",
                success=False,
                latency_ms=latency,
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        result = TestResult(
            request_type="fraud",
            success=False,
            latency_ms=(time.time() - start) * 1000,
            error=str(e)
        )
    
    stats.add(result)
    return result


async def run_sequential_test(client: httpx.AsyncClient, stats: TestStats):
    """Run sequential test (no overlap)."""
    print("\n📊 Test 1: Sequential Requests (baseline)")
    print("=" * 50)
    
    # 3 chatbot requests
    for i, prompt in enumerate(CHATBOT_PROMPTS[:3]):
        result = await test_chatbot_request(client, prompt, stats)
        status = "✅" if result.success else "❌"
        print(f"  Chat {i+1}: {result.latency_ms:.0f}ms {status}")
    
    # 2 fraud requests
    for i, prompt in enumerate(FRAUD_PROMPTS):
        result = await test_fraud_request(client, prompt, stats)
        status = "✅" if result.success else "❌"
        print(f"  Fraud {i+1}: {result.latency_ms:.0f}ms {status}")


async def run_parallel_test(client: httpx.AsyncClient, stats: TestStats):
    """Run parallel test (simulating dual workload)."""
    print("\n📊 Test 2: Parallel Requests (stress test)")
    print("=" * 50)
    
    # Create mixed tasks
    tasks = []
    
    # 3 chatbot + 3 fraud in parallel
    for i, prompt in enumerate(CHATBOT_PROMPTS[:3]):
        tasks.append(("chat", i, test_chatbot_request(client, prompt, stats)))
    
    for i, prompt in enumerate(FRAUD_PROMPTS):
        tasks.append(("fraud", i, test_fraud_request(client, prompt, stats)))
        # Duplicate fraud to stress test
        tasks.append(("fraud", i+10, test_fraud_request(client, prompt, stats)))
    
    # Run all in parallel
    print(f"  Running {len(tasks)} requests in parallel...")
    start = time.time()
    
    results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
    
    parallel_time = (time.time() - start) * 1000
    print(f"  Total parallel time: {parallel_time:.0f}ms")
    
    # Report results
    for (req_type, idx, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"    {req_type.upper()} {idx}: ❌ Exception: {result}")
        else:
            status = "✅" if result.success else "❌"
            print(f"    {req_type.upper()} {idx}: {result.latency_ms:.0f}ms {status}")


async def run_burst_test(client: httpx.AsyncClient, stats: TestStats):
    """Run burst test (simulating spike load)."""
    print("\n📊 Test 3: Burst Load (10 requests at once)")
    print("=" * 50)
    
    # 10 requests at once (5 chat, 5 fraud)
    tasks = []
    
    for i in range(5):
        prompt = random.choice(CHATBOT_PROMPTS)
        tasks.append(test_chatbot_request(client, prompt, stats))
        
        prompt = random.choice(FRAUD_PROMPTS)
        tasks.append(test_fraud_request(client, prompt, stats))
    
    print(f"  Sending burst of {len(tasks)} requests...")
    start = time.time()
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    burst_time = (time.time() - start) * 1000
    
    success_count = sum(1 for r in results if not isinstance(r, Exception) and r.success)
    print(f"  Completed in {burst_time:.0f}ms")
    print(f"  Success: {success_count}/{len(results)}")


async def test_neural_ensemble():
    """Test the neural ensemble with GPU acceleration."""
    print("\n📊 Test 4: Neural Ensemble (GPU/NumPy)")
    print("=" * 50)
    
    try:
        # Import from local module
        from expertNeuralEnsemble import ExpertNeuralEnsemble
        
        print("  Initializing neural ensemble...")
        ensemble = ExpertNeuralEnsemble(seed=42)
        
        # Train with synthetic data
        print("  Training on synthetic data...")
        import numpy as np
        np.random.seed(42)
        
        X_train = np.random.randn(1000, 34).astype(np.float32)
        y_train = np.random.randint(0, 100, 1000).astype(np.float32)
        
        start = time.time()
        ensemble.train(X_train, y_train)
        train_time = (time.time() - start) * 1000
        print(f"  ✅ Training completed in {train_time:.0f}ms")
        
        # Inference test
        print("  Running inference test...")
        latencies = []
        
        for i in range(100):
            x = np.random.randn(34).astype(np.float32)
            start = time.time()
            result = ensemble.predict(x)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = statistics.mean(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"  ✅ Inference: avg={avg_latency:.2f}ms, P99={p99_latency:.2f}ms")
        print(f"  ✅ GPU enabled: {ensemble.use_gpu}")
        
    except ImportError as e:
        print(f"  ⚠️ Could not import ensemble: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("         PAYFLOW GPU PARALLEL LOAD TEST")
    print("         RTX 4070 • Qwen3:8B • Dual Workload")
    print("=" * 70)
    
    # Check Ollama
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code != 200:
                print("❌ Ollama not running. Start with: ollama serve")
                return
            
            models = [m["name"] for m in response.json().get("models", [])]
            if not any(MODEL in m for m in models):
                print(f"❌ Model {MODEL} not found. Pull with: ollama pull {MODEL}")
                return
                
            print(f"✅ Ollama running, model {MODEL} available")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            return
    
    # Run tests
    stats = TestStats()
    
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_connections=20)
    ) as client:
        
        # Warm up
        if not await warm_up_model(client):
            print("⚠️ Warm-up failed, continuing anyway...")
        
        # Sequential baseline
        await run_sequential_test(client, stats)
        
        # Parallel stress test
        await run_parallel_test(client, stats)
        
        # Burst test
        await run_burst_test(client, stats)
    
    # Neural ensemble test
    await test_neural_ensemble()
    
    # Print summary
    print("\n" + "=" * 70)
    print("                    TEST SUMMARY")
    print("=" * 70)
    
    summary = stats.get_summary()
    print(f"\n📈 Total Duration: {summary['total_duration_s']}s")
    print(f"📊 Total Requests: {summary['total_requests']}")
    
    print("\n🤖 CHATBOT Performance:")
    chat = summary.get('chatbot', {})
    if chat.get('count', 0) > 0:
        print(f"   Requests: {chat['count']}")
        print(f"   Success Rate: {chat['success_rate']}")
        print(f"   Latency: {chat['avg_latency_ms']}ms avg, {chat['median_latency_ms']}ms median")
        print(f"   Range: {chat['min_latency_ms']}ms - {chat['max_latency_ms']}ms")
    
    print("\n🔍 FRAUD DETECTION Performance:")
    fraud = summary.get('fraud_detection', {})
    if fraud.get('count', 0) > 0:
        print(f"   Requests: {fraud['count']}")
        print(f"   Success Rate: {fraud['success_rate']}")
        print(f"   Latency: {fraud['avg_latency_ms']}ms avg, {fraud['median_latency_ms']}ms median")
        print(f"   Range: {fraud['min_latency_ms']}ms - {fraud['max_latency_ms']}ms")
    
    # Performance verdict
    print("\n" + "=" * 70)
    
    chat_ok = chat.get('avg_latency_ms', 99999) < 5000  # 5s for chat
    fraud_ok = fraud.get('avg_latency_ms', 99999) < 1000  # 1s for fraud
    
    if chat_ok and fraud_ok:
        print("✅ GPU OPTIMIZATION: PASSED")
        print("   Both workloads meet performance targets!")
    elif fraud_ok:
        print("⚠️ GPU OPTIMIZATION: PARTIAL")
        print("   Fraud detection OK, chatbot needs tuning")
    else:
        print("❌ GPU OPTIMIZATION: NEEDS IMPROVEMENT")
        print("   Consider reducing concurrent requests or context size")
    
    print("\n🎮 RTX 4070 GPU Test Complete!")


if __name__ == "__main__":
    asyncio.run(main())
