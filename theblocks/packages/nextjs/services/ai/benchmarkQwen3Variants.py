"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  QWEN3:8B VARIANT BENCHMARK - FULL THINKING vs QUANTIZED COMPARISON                     ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║  Testing:                                                                                ║
║  1. qwen3:8b (Full) + Thinking Mode Enabled                                              ║
║  2. qwen3:8b-q4_K_M (4-bit Quantized)                                                    ║
║                                                                                          ║
║  Hardware: RTX 4070 Laptop GPU (8GB VRAM)                                                ║
║  Hackxios 2K25 - PayFlow Protocol                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import httpx
import asyncio
import json
import time
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434"

# Models to test
MODELS = {
    "qwen3:8b": {
        "name": "Qwen3:8B Full (Thinking)",
        "think": True,  # Enable thinking mode
        "description": "Full precision with thinking mode"
    },
    "qwen3:8b-q4_K_M": {
        "name": "Qwen3:8B Q4_K_M (Quantized)",
        "think": True,  # Test with thinking first
        "description": "4-bit quantized version"
    },
    "qwen3:8b-no-think": {
        "model": "qwen3:8b",
        "name": "Qwen3:8B Full (No Thinking)",
        "think": False,  # Disable thinking mode
        "description": "Full precision without thinking"
    },
    "qwen3:8b-q4_K_M-no-think": {
        "model": "qwen3:8b-q4_K_M",
        "name": "Qwen3:8B Q4_K_M (No Thinking)",
        "think": False,  # Disable thinking mode
        "description": "4-bit quantized without thinking"
    }
}

# Fraud detection system prompt (concise)
SYSTEM_PROMPT = """You are an expert financial crime AI for PayFlow Protocol.
Analyze transactions for: money laundering, velocity anomalies, amount patterns, counterparty risk.
Output ONLY valid JSON: {"risk_score": 0-100, "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL", "approved": bool, "alerts": [], "explanation": "brief", "confidence": 0.0-1.0}"""

# GPU settings
GPU_OPTIONS = {
    "num_gpu": 99,
    "num_ctx": 4096,
    "num_batch": 512,
    "num_thread": 8,
    "use_mmap": True,
    "temperature": 0.2,
    "top_p": 0.85,
    "top_k": 30,
    "num_predict": 256,
    "repeat_penalty": 1.15,
}

# ═══════════════════════════════════════════════════════════════════════════════
#                              BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_transaction() -> Dict[str, Any]:
    """Generate a random test transaction."""
    return {
        "amount": random.uniform(100, 100000),
        "token": random.choice(["USDC", "PYUSD", "DAI"]),
        "sender_tx": random.randint(0, 500),
        "sender_vol": random.uniform(0, 5_000_000),
        "sender_avg": random.uniform(100, 25000),
        "sender_age": random.randint(0, 365),
        "recipient_tx": random.randint(0, 200),
        "recipient_age": random.randint(0, 365),
        "ml_velocity": random.randint(0, 100),
        "ml_amount": random.randint(0, 100),
        "ml_pattern": random.randint(0, 100),
        "ml_graph": random.randint(0, 100),
    }


def build_prompt(tx: Dict) -> str:
    """Build fraud analysis prompt."""
    return f"""TX: ${tx['amount']:,.0f} {tx['token']}
Sender: {tx['sender_tx']} txs, ${tx['sender_vol']:,.0f} vol, {tx['sender_age']}d old, {(tx['amount'] / tx['sender_avg'] * 100) if tx['sender_avg'] > 0 else 100:.0f}% of avg
Recipient: {tx['recipient_tx']} txs, {tx['recipient_age']}d old{"⚠️NEW" if tx['recipient_tx'] == 0 else ""}
ML: vel={tx['ml_velocity']} amt={tx['ml_amount']} pat={tx['ml_pattern']} gph={tx['ml_graph']}
JSON verdict:"""


async def run_single_inference(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    think: bool
) -> Dict[str, Any]:
    """Run a single inference and return timing + result."""
    start = time.time()
    
    try:
        # Build request with think parameter
        request_body = {
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": GPU_OPTIONS.copy(),
            "think": think  # Ollama's think parameter
        }
        
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json=request_body,
            timeout=60.0
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            raw_response = data.get("response", "")
            
            # Remove thinking tags
            import re
            clean = re.sub(r'<think>[\s\S]*?</think>', '', raw_response).strip()
            
            # Parse score
            try:
                parsed = json.loads(clean)
                score = int(parsed.get("risk_score", 50))  # Ensure int
                level = str(parsed.get("risk_level", "MEDIUM"))  # Ensure str
            except:
                score = 50
                level = "PARSE_ERROR"
            
            return {
                "success": True,
                "latency_ms": elapsed_ms,
                "score": score,
                "level": level,
                "tokens_prompt": data.get("prompt_eval_count", 0),
                "tokens_generated": data.get("eval_count", 0),
                "thinking_tokens": len(re.findall(r'<think>[\s\S]*?</think>', raw_response))
            }
        else:
            return {
                "success": False,
                "latency_ms": elapsed_ms,
                "error": f"HTTP {response.status_code}"
            }
            
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {
            "success": False,
            "latency_ms": elapsed_ms,
            "error": str(e)
        }


async def warmup_model(client: httpx.AsyncClient, model: str, think: bool) -> Dict:
    """Warm up a model and get VRAM usage."""
    print(f"  Warming up {model}...")
    start = time.time()
    
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": "Hello",
                "stream": False,
                "keep_alive": "30m",
                "think": think,
                "options": {"num_gpu": 99, "num_predict": 5}
            },
            timeout=120.0
        )
        
        load_time = (time.time() - start) * 1000
        
        # Get VRAM usage
        ps_response = await client.get(f"{OLLAMA_URL}/api/ps")
        vram_mb = 0
        if ps_response.status_code == 200:
            for m in ps_response.json().get("models", []):
                if model in m.get("name", ""):
                    vram_mb = m.get("size_vram", 0) / (1024*1024)
        
        return {"success": True, "load_time_ms": load_time, "vram_mb": vram_mb}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


async def benchmark_model(
    model_key: str,
    config: Dict,
    num_transactions: int = 15
) -> Dict[str, Any]:
    """Benchmark a single model variant."""
    
    actual_model = config.get("model", model_key)
    think = config.get("think", True)
    name = config.get("name", model_key)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {name}")
    print(f"Model: {actual_model} | Think: {think}")
    print(f"{'='*60}")
    
    async with httpx.AsyncClient() as client:
        # Warmup
        warmup = await warmup_model(client, actual_model, think)
        if not warmup.get("success"):
            return {"model": name, "error": warmup.get("error"), "results": []}
        
        print(f"  ✅ Loaded in {warmup['load_time_ms']:.0f}ms | VRAM: {warmup['vram_mb']:.0f}MB")
        
        # Generate test transactions
        transactions = [generate_test_transaction() for _ in range(num_transactions)]
        
        # Run benchmark
        results = []
        for i, tx in enumerate(transactions):
            prompt = build_prompt(tx)
            result = await run_single_inference(client, actual_model, prompt, think)
            results.append(result)
            
            if result["success"]:
                status = "✅" if result["latency_ms"] < 500 else "⚠️"
                print(f"  {status} TX {i+1:2d}/{num_transactions} | "
                      f"Score: {result['score']:3d} | "
                      f"Level: {result['level']:8s} | "
                      f"Time: {result['latency_ms']:7.1f}ms | "
                      f"Tokens: {result['tokens_generated']}")
            else:
                print(f"  ❌ TX {i+1:2d}/{num_transactions} | Error: {result.get('error', 'Unknown')}")
        
        # Calculate statistics
        successful = [r for r in results if r.get("success")]
        if successful:
            latencies = [r["latency_ms"] for r in successful]
            stats = {
                "model": name,
                "actual_model": actual_model,
                "think_mode": think,
                "total": num_transactions,
                "successful": len(successful),
                "failed": num_transactions - len(successful),
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 5 else max(latencies),
                "stddev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "under_300ms": len([l for l in latencies if l < 300]),
                "under_500ms": len([l for l in latencies if l < 500]),
                "under_1000ms": len([l for l in latencies if l < 1000]),
                "vram_mb": warmup["vram_mb"],
                "load_time_ms": warmup["load_time_ms"],
                "avg_tokens": statistics.mean([r["tokens_generated"] for r in successful])
            }
        else:
            stats = {"model": name, "error": "All inferences failed"}
        
        return stats


async def run_full_benchmark():
    """Run complete benchmark suite."""
    print("\n" + "═"*70)
    print("  QWEN3:8B VARIANT BENCHMARK - FULL COMPARISON")
    print("  RTX 4070 Laptop GPU (8GB VRAM)")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═"*70)
    
    # Test configurations (ordered for fair comparison)
    test_configs = [
        ("qwen3:8b", MODELS["qwen3:8b"]),
        ("qwen3:8b-no-think", MODELS["qwen3:8b-no-think"]),
        ("qwen3:8b-q4_K_M", MODELS["qwen3:8b-q4_K_M"]),
        ("qwen3:8b-q4_K_M-no-think", MODELS["qwen3:8b-q4_K_M-no-think"]),
    ]
    
    all_results = []
    
    for model_key, config in test_configs:
        result = await benchmark_model(model_key, config, num_transactions=15)
        all_results.append(result)
        
        # Unload model between tests for fair comparison
        async with httpx.AsyncClient() as client:
            await client.post(f"{OLLAMA_URL}/api/generate", 
                            json={"model": config.get("model", model_key), "keep_alive": "0"})
        await asyncio.sleep(2)
    
    # Print comparison table
    print("\n" + "═"*90)
    print("  BENCHMARK RESULTS COMPARISON")
    print("═"*90)
    print(f"{'Model':<35} {'Avg(ms)':<10} {'Med(ms)':<10} {'P95(ms)':<10} {'<500ms':<8} {'VRAM(MB)':<10}")
    print("-"*90)
    
    for r in all_results:
        if "error" in r and r["error"]:
            print(f"{r['model']:<35} ERROR: {r['error']}")
        else:
            pct_under_500 = f"{r['under_500ms']}/{r['total']}"
            print(f"{r['model']:<35} {r['avg_latency_ms']:<10.1f} {r['median_latency_ms']:<10.1f} "
                  f"{r['p95_latency_ms']:<10.1f} {pct_under_500:<8} {r['vram_mb']:<10.0f}")
    
    print("═"*90)
    
    # Find best configuration
    valid_results = [r for r in all_results if "avg_latency_ms" in r]
    if valid_results:
        best = min(valid_results, key=lambda x: x["avg_latency_ms"])
        print(f"\n🏆 BEST CONFIGURATION: {best['model']}")
        print(f"   Average Latency: {best['avg_latency_ms']:.1f}ms")
        print(f"   VRAM Usage: {best['vram_mb']:.0f}MB")
        print(f"   Tokens/Response: {best['avg_tokens']:.0f}")
        
        # Performance tier
        if best['avg_latency_ms'] < 300:
            print("   🎉 EXCELLENT: Under 300ms target!")
        elif best['avg_latency_ms'] < 500:
            print("   ✅ GOOD: Under 500ms")
        elif best['avg_latency_ms'] < 1000:
            print("   ⚠️ ACCEPTABLE: Under 1 second")
        else:
            print("   ❌ NEEDS OPTIMIZATION: Over 1 second")
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = asyncio.run(run_full_benchmark())
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📊 Results saved to benchmark_results.json")
