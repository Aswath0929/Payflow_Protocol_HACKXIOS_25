# PayFlow GPU Optimization Guide
## RTX 4070 Laptop GPU (8GB VRAM)

## 🎮 GPU Performance Results

### Sequential Performance (Baseline)
| Workload | Latency | Target | Status |
|----------|---------|--------|--------|
| Fraud Detection | **573ms** | <1000ms | ✅ PASS |
| Chatbot Response | **6.5s** | <30s | ✅ PASS |
| Neural Ensemble | **2.2ms** | <50ms | ✅ PASS |

### Parallel Performance (Dual Workload)
| Scenario | Result | Notes |
|----------|--------|-------|
| 2 concurrent requests | ✅ Both succeed | Optimal configuration |
| 7+ concurrent requests | ⚠️ Some timeout | Queue builds up |
| 10 burst requests | 60% success | GPU saturated |

## 🔧 Optimal Configuration

### Ollama Environment Variables
```powershell
# Set before starting Ollama
$env:OLLAMA_NUM_PARALLEL = 2          # Max 2 parallel requests
$env:OLLAMA_CONTEXT_LENGTH = 4096     # Context window size
$env:OLLAMA_KEEP_ALIVE = "30m"        # Keep model in VRAM
```

### GPU-Optimized Ollama Options
```python
GPU_OPTIONS = {
    "num_gpu": 99,          # All layers on GPU
    "num_ctx": 4096,        # Context window
    "num_batch": 512,       # Batch size
    "num_thread": 8,        # CPU threads
    "use_mmap": True,       # Memory-mapped loading
    "top_k": 40,            # Token sampling
    "top_p": 0.9,           # Nucleus sampling
    "repeat_penalty": 1.1,  # Prevent repetition
}
```

### Request Priority
| Type | Priority | Temperature | Max Tokens | Timeout |
|------|----------|-------------|------------|---------|
| Fraud Detection | HIGH (10) | 0.3 | 128 | 10s |
| Chatbot | NORMAL (5) | 0.7 | 256 | 30s |

## 📊 VRAM Allocation (8GB Total)
```
┌─────────────────────────────────────┐
│ Qwen3:8B Model Weights     ~5.0 GB │
│ KV Cache (4K context)      ~1.5 GB │
│ Neural Ensemble Matrices   ~0.3 GB │
│ Working Memory             ~1.2 GB │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Start Ollama with Parallel Support
```powershell
$env:OLLAMA_NUM_PARALLEL = 2
ollama serve
```

### 2. Warm Up GPU
```python
# Run once at startup to load model to VRAM
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen3:8b",
    "prompt": "Hello",
    "keep_alive": "30m",
    "options": {"num_gpu": 99, "num_predict": 1}
})
```

### 3. Run Fraud Detection (High Priority)
```python
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen3:8b",
    "prompt": fraud_prompt,
    "format": "json",
    "keep_alive": "30m",
    "options": {
        "num_gpu": 99,
        "temperature": 0.3,
        "num_predict": 128
    }
})
```

## 📈 Performance Tips

1. **Warm up on startup**: Load model once, keep in VRAM for 30 mins
2. **Limit concurrency**: Max 2 parallel requests on RTX 4070
3. **Shorter fraud responses**: Use `num_predict: 128` for fraud detection
4. **Use JSON format**: Forces structured output, faster parsing
5. **Monitor with `/api/ps`**: Check loaded models and VRAM usage

## 🔍 Troubleshooting

### High Latency
- Check if model is loaded: `curl localhost:11434/api/ps`
- Reduce context size if OOM: `num_ctx: 2048`
- Lower batch size: `num_batch: 256`

### Request Timeouts
- Reduce parallel requests
- Increase timeout for chatbot
- Priority queue fraud detection

### OOM Errors
- Ollama auto-enters "low VRAM mode" for 8GB GPUs
- Reduce `num_ctx` to 2048
- Use smaller model variant: `qwen3:4b`

## 📁 Files Modified

| File | Changes |
|------|---------|
| `gpuOptimizer.py` | GPU configuration, request queue |
| `localLLMAnalyzer.py` | GPU-optimized Ollama params |
| `expertNeuralEnsemble.py` | CuPy GPU acceleration (optional) |
| `AIChatbotPro.tsx` | GPU-optimized frontend config |

## ✅ Hackxios 2K25 Ready!

The PayFlow Protocol is optimized for:
- ✅ Real-time fraud detection (<1s)
- ✅ Conversational AI chatbot
- ✅ GPU-accelerated inference
- ✅ Parallel workload handling
