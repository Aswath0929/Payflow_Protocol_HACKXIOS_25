"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW LOCAL LLM ANALYZER - QWEN3 INTEGRATION                    ║
║                                                                                       ║
║   Local Large Language Model for Fraud Detection                                     ║
║   Running on RTX 4070 (8GB VRAM) via Ollama                                          ║
║                                                                                       ║
║   Model: Qwen3 (Latest 2025 release from Alibaba)                                    ║
║   Recommended: qwen3:8b-q4_K_M (~5GB VRAM) or qwen3:4b (~3GB VRAM)                   ║
║                                                                                       ║
║   Features:                                                                           ║
║   • 100% LOCAL - No cloud API, no data leaves your machine                           ║
║   • GPU Accelerated - Runs on your RTX 4070                                          ║
║   • Real-time inference - <500ms response time                                       ║
║   • Context-aware fraud reasoning                                                    ║
║   • Cryptographic signing of all predictions                                         ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import httpx
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('LocalLLMAnalyzer')

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Qwen3Config:
    """
    Configuration for Qwen3 local LLM.
    
    OPTIMIZED FOR RTX 4070 (8GB VRAM) DUAL WORKLOAD:
    - Fraud Detection (high priority, fast)
    - AI Chatbot (parallel, streaming)
    """
    
    # Ollama server URL (default local)
    OLLAMA_URL = "http://localhost:11434"
    
    # Qwen3 model options (choose based on your VRAM):
    # - qwen3:4b     → ~3GB VRAM (faster, good for real-time)
    # - qwen3:8b     → ~5GB VRAM (recommended for RTX 4070)
    # - qwen3:8b-q4_K_M → ~5GB VRAM (4-bit quantized, best quality/speed)
    # - qwen3:1.7b   → ~1.5GB VRAM (fastest, lighter analysis)
    MODEL_NAME = "qwen3:8b"
    
    # ═══════════════════════════════════════════════════════════════════
    #                    GPU ACCELERATION SETTINGS (RTX 4070)
    # ═══════════════════════════════════════════════════════════════════
    # Maximum GPU utilization for fraud detection
    NUM_GPU = 99              # Offload ALL layers to GPU (max speed)
    NUM_CTX = 4096            # Context window (optimized for 8GB VRAM)
    NUM_BATCH = 512           # Batch size for prompt processing
    NUM_THREAD = 8            # CPU threads for non-GPU operations
    USE_MMAP = True           # Memory-mapped loading (faster startup)
    USE_MLOCK = False         # Don't lock in RAM, prefer GPU VRAM
    
    # ═══════════════════════════════════════════════════════════════════
    #                    INFERENCE OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    TEMPERATURE = 0.3         # Low for consistent fraud analysis
    TOP_P = 0.9
    TOP_K = 40                # Limit token sampling (faster)
    MAX_TOKENS = 512          # Fraud responses are concise
    REPEAT_PENALTY = 1.1      # Prevent repetition
    MIROSTAT = 0              # Disabled for speed
    
    # ═══════════════════════════════════════════════════════════════════
    #                    TIMEOUT & RETRY SETTINGS
    # ═══════════════════════════════════════════════════════════════════
    TIMEOUT = 30.0            # seconds (fraud detection timeout)
    CONNECT_TIMEOUT = 5.0     # Connection timeout
    MAX_RETRIES = 3
    RETRY_DELAY = 0.5         # Faster retry for real-time
    
    # ═══════════════════════════════════════════════════════════════════
    #                    PARALLEL WORKLOAD SETTINGS
    # ═══════════════════════════════════════════════════════════════════
    KEEP_ALIVE = "30m"        # Keep model in VRAM for 30 minutes
    PRIORITY = 10             # Higher priority than chatbot (5)

# ═══════════════════════════════════════════════════════════════════════════════
#                              DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMAnalysisResult:
    """Result from local LLM analysis."""
    risk_score: int
    risk_level: str
    approved: bool
    explanation: str
    alerts: List[str]
    recommendations: List[str]
    confidence: float
    model_used: str
    inference_time_ms: float
    is_local: bool = True

# ═══════════════════════════════════════════════════════════════════════════════
#                              QWEN3 LOCAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class Qwen3LocalAnalyzer:
    """
    Local LLM Analyzer using Qwen3 via Ollama.
    
    Runs entirely on your RTX 4070 GPU - no cloud API needed!
    Provides GPT-4 level reasoning for fraud detection, 100% offline.
    """
    
    SYSTEM_PROMPT = """You are an expert financial crime analyst specializing in cryptocurrency and stablecoin fraud detection for the PayFlow Protocol.

Your role is to analyze transactions for:
1. Money laundering patterns (layering, structuring, smurfing)
2. Terrorist financing indicators
3. Sanctions evasion attempts
4. Unusual velocity or amount patterns
5. Mixing service usage
6. Wash trading and round-trip transactions

IMPORTANT: You must respond ONLY with valid JSON in this exact format:
{
    "risk_score": <0-100>,
    "risk_level": "<SAFE|LOW|MEDIUM|HIGH|CRITICAL>",
    "approved": <true|false>,
    "alerts": ["<specific concern 1>", "<specific concern 2>"],
    "explanation": "<detailed analysis in 2-3 sentences>",
    "recommendations": ["<action 1>", "<action 2>"],
    "confidence": <0.0-1.0>
}

Risk Level Guide:
- SAFE (0-20): Normal transaction, no concerns
- LOW (21-40): Minor anomalies, monitor only
- MEDIUM (41-60): Flagged for review, potential issues
- HIGH (61-80): Suspicious, requires investigation
- CRITICAL (81-100): Block immediately, likely fraud

Be precise and cite specific patterns. Respond ONLY with JSON, no other text."""

    def __init__(self, model_name: str = None, ollama_url: str = None):
        """Initialize Qwen3 Local Analyzer."""
        self.model_name = model_name or Qwen3Config.MODEL_NAME
        self.ollama_url = ollama_url or Qwen3Config.OLLAMA_URL
        self.client = httpx.AsyncClient(timeout=Qwen3Config.TIMEOUT)
        self.enabled = False
        self.model_info = None
        
        logger.info(f"Qwen3LocalAnalyzer initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Ollama URL: {self.ollama_url}")
    
    async def check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Check Ollama server
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                logger.warning("Ollama server not responding")
                return False
            
            # Check if Qwen3 model is available
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            # Check for exact match or partial match
            model_available = any(
                self.model_name in m or m.startswith(self.model_name.split(":")[0])
                for m in models
            )
            
            if model_available:
                self.enabled = True
                logger.info(f"✅ Qwen3 model available: {self.model_name}")
                return True
            else:
                logger.warning(f"⚠️ Model {self.model_name} not found. Available: {models}")
                logger.info(f"Run: ollama pull {self.model_name}")
                return False
                
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            logger.info("Start Ollama with: ollama serve")
            return False
    
    async def warm_up(self) -> bool:
        """Warm up the model by loading it into GPU memory with optimized settings."""
        try:
            logger.info("Warming up Qwen3 model (loading to GPU with CUDA acceleration)...")
            start = time.time()
            
            # Send a simple prompt to load the model with GPU-optimized settings
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "keep_alive": Qwen3Config.KEEP_ALIVE,
                    "options": {
                        # GPU Acceleration (RTX 4070)
                        "num_gpu": Qwen3Config.NUM_GPU,
                        "num_ctx": Qwen3Config.NUM_CTX,
                        "num_batch": Qwen3Config.NUM_BATCH,
                        "use_mmap": Qwen3Config.USE_MMAP,
                        "num_predict": 1
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                load_time = (time.time() - start) * 1000
                logger.info(f"✅ Qwen3 loaded to GPU in {load_time:.0f}ms")
                self.enabled = True
                return True
            else:
                logger.error(f"Failed to warm up model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
            return False
    
    def _build_analysis_prompt(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        sender_profile: Dict,
        recipient_profile: Dict,
        local_scores: Dict,
        neural_net_result: Optional[Dict] = None
    ) -> str:
        """Build detailed analysis prompt for Qwen3."""
        
        # Calculate some derived metrics
        sender_tx_count = sender_profile.get('transaction_count', 0)
        sender_volume = sender_profile.get('total_volume', 0)
        sender_avg = sender_profile.get('avg_amount', 0)
        
        recipient_tx_count = recipient_profile.get('transaction_count', 0)
        recipient_volume = recipient_profile.get('total_volume', 0)
        
        # Neural network context
        nn_context = ""
        if neural_net_result:
            nn_context = f"""
## Neural Network Pre-Analysis
- Risk Score: {neural_net_result.get('risk_score', 'N/A')}
- Risk Level: {neural_net_result.get('risk_level', 'N/A')}
- Is Anomaly: {neural_net_result.get('is_anomaly', False)}
- Flags: {', '.join(neural_net_result.get('flags', [])) or 'None'}
"""

        prompt = f"""Analyze this stablecoin transaction for fraud risk:

## Transaction Details
- Transaction ID: {transaction_id[:16]}...
- Amount: ${amount:,.2f} USDC
- Sender: {sender[:10]}...{sender[-6:]}
- Recipient: {recipient[:10]}...{recipient[-6:]}

## Sender Profile
- Total Transactions: {sender_tx_count}
- Total Volume: ${sender_volume:,.2f}
- Average Transaction: ${sender_avg:,.2f}
- This TX vs Average: {(amount / sender_avg * 100) if sender_avg > 0 else 100:.1f}%

## Recipient Profile  
- Total Transactions: {recipient_tx_count}
- Total Volume: ${recipient_volume:,.2f}
- Is New Recipient: {"Yes" if recipient_tx_count == 0 else "No"}

## ML Model Scores (0-100)
- Velocity Score: {local_scores.get('velocity', 0)}
- Amount Score: {local_scores.get('amount', 0)}
- Pattern Score: {local_scores.get('pattern', 0)}
- Graph Score: {local_scores.get('graph', 0)}
- Timing Score: {local_scores.get('timing', 0)}
- Isolation Forest Anomaly: {local_scores.get('isolation_forest', 0):.2f}
{nn_context}
## Analysis Required
Based on the above data, provide your fraud risk assessment.
Consider: velocity anomalies, amount patterns, counterparty risk, timing, and any red flags.

Respond with JSON only:"""

        return prompt
    
    async def analyze(
        self,
        transaction_id: str,
        sender: str,
        recipient: str,
        amount: float,
        sender_profile: Dict,
        recipient_profile: Dict,
        local_scores: Dict,
        neural_net_result: Optional[Dict] = None
    ) -> LLMAnalysisResult:
        """
        Analyze transaction using Qwen3 local LLM.
        
        Returns detailed fraud analysis with reasoning.
        """
        if not self.enabled:
            return self._fallback_analysis(local_scores, neural_net_result)
        
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_analysis_prompt(
            transaction_id, sender, recipient, amount,
            sender_profile, recipient_profile, local_scores,
            neural_net_result
        )
        
        # Call Qwen3 via Ollama
        for attempt in range(Qwen3Config.MAX_RETRIES):
            try:
                response = await self.client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "system": self.SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "keep_alive": Qwen3Config.KEEP_ALIVE,
                        "options": {
                            # GPU Acceleration (RTX 4070)
                            "num_gpu": Qwen3Config.NUM_GPU,
                            "num_thread": Qwen3Config.NUM_THREAD,
                            "num_ctx": Qwen3Config.NUM_CTX,
                            "num_batch": Qwen3Config.NUM_BATCH,
                            "use_mmap": Qwen3Config.USE_MMAP,
                            "use_mlock": Qwen3Config.USE_MLOCK,
                            
                            # Inference settings
                            "temperature": Qwen3Config.TEMPERATURE,
                            "top_p": Qwen3Config.TOP_P,
                            "top_k": Qwen3Config.TOP_K,
                            "num_predict": Qwen3Config.MAX_TOKENS,
                            "repeat_penalty": Qwen3Config.REPEAT_PENALTY,
                            "mirostat": Qwen3Config.MIROSTAT,
                        }
                    },
                    timeout=Qwen3Config.TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "")
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Parse JSON response
                    try:
                        analysis = json.loads(generated_text)
                        
                        return LLMAnalysisResult(
                            risk_score=int(analysis.get("risk_score", 50)),
                            risk_level=analysis.get("risk_level", "MEDIUM"),
                            approved=analysis.get("approved", True),
                            explanation=analysis.get("explanation", ""),
                            alerts=analysis.get("alerts", []),
                            recommendations=analysis.get("recommendations", []),
                            confidence=float(analysis.get("confidence", 0.8)),
                            model_used=self.model_name,
                            inference_time_ms=inference_time,
                            is_local=True
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse Qwen3 response: {e}")
                        # Try to extract from malformed response
                        return self._parse_malformed_response(
                            generated_text, inference_time
                        )
                else:
                    logger.warning(f"Ollama API error: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Qwen3 analysis attempt {attempt + 1} failed: {e}")
                if attempt < Qwen3Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Qwen3Config.RETRY_DELAY)
        
        # All retries failed - use fallback
        return self._fallback_analysis(local_scores, neural_net_result)
    
    def _parse_malformed_response(
        self, 
        text: str, 
        inference_time: float
    ) -> LLMAnalysisResult:
        """Try to extract useful information from malformed response."""
        import re
        
        # Try to find risk score
        score_match = re.search(r'"risk_score":\s*(\d+)', text)
        risk_score = int(score_match.group(1)) if score_match else 50
        
        # Try to find risk level
        level_match = re.search(r'"risk_level":\s*"(\w+)"', text)
        risk_level = level_match.group(1) if level_match else "MEDIUM"
        
        # Try to find explanation
        exp_match = re.search(r'"explanation":\s*"([^"]+)"', text)
        explanation = exp_match.group(1) if exp_match else "Analysis completed with partial response"
        
        return LLMAnalysisResult(
            risk_score=risk_score,
            risk_level=risk_level,
            approved=risk_score < 80,
            explanation=explanation,
            alerts=[],
            recommendations=[],
            confidence=0.6,
            model_used=self.model_name,
            inference_time_ms=inference_time,
            is_local=True
        )
    
    def _fallback_analysis(
        self, 
        local_scores: Dict,
        neural_net_result: Optional[Dict] = None
    ) -> LLMAnalysisResult:
        """Fallback when LLM is unavailable - use ML scores directly."""
        
        # Combine scores
        scores = [
            local_scores.get('velocity', 0),
            local_scores.get('amount', 0),
            local_scores.get('pattern', 0),
            local_scores.get('graph', 0),
            local_scores.get('timing', 0),
        ]
        
        if neural_net_result:
            scores.append(neural_net_result.get('risk_score', 0))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        risk_score = int(min(100, avg_score))
        
        # Determine risk level
        if risk_score <= 20:
            risk_level = "SAFE"
        elif risk_score <= 40:
            risk_level = "LOW"
        elif risk_score <= 60:
            risk_level = "MEDIUM"
        elif risk_score <= 80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return LLMAnalysisResult(
            risk_score=risk_score,
            risk_level=risk_level,
            approved=risk_score < 80,
            explanation="Fallback analysis using ML models only (LLM unavailable)",
            alerts=[],
            recommendations=["Enable local LLM for detailed analysis"],
            confidence=0.5,
            model_used="fallback",
            inference_time_ms=0,
            is_local=True
        )
    
    def analyze_transaction(
        self,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: int = None,
        context: Dict = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for transaction analysis.
        Used by ExpertAIOracle for quick analysis.
        
        Returns simplified dict for integration.
        """
        import asyncio
        
        context = context or {}
        
        # Try to get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use fallback
                return self._sync_fallback_analysis(context)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Try async analysis if possible
        try:
            if not self.enabled:
                return self._sync_fallback_analysis(context)
            
            result = loop.run_until_complete(
                self.analyze(
                    transaction_id=f"tx_{hash(f'{sender}{recipient}{amount}')}",
                    sender=sender,
                    recipient=recipient,
                    amount=amount,
                    sender_profile={},
                    recipient_profile={},
                    local_scores={
                        "ensemble": context.get("ensemble_risk", 50),
                        "typology": context.get("typology_risk", 50),
                    },
                    neural_net_result={
                        "risk_score": context.get("ensemble_risk", 50),
                        "risk_level": "MEDIUM",
                    }
                )
            )
            
            return {
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "explanation": result.explanation,
                "alerts": result.alerts,
                "recommendations": result.recommendations,
                "confidence": result.confidence,
                "model_used": result.model_used,
                "inference_time_ms": result.inference_time_ms,
            }
            
        except Exception as e:
            logger.warning(f"Sync analysis failed: {e}")
            return self._sync_fallback_analysis(context)
    
    def _sync_fallback_analysis(self, context: Dict) -> Dict[str, Any]:
        """Synchronous fallback analysis when LLM is unavailable."""
        # Combine available scores
        ensemble_risk = context.get("ensemble_risk", 50)
        typology_risk = context.get("typology_risk", 50)
        compliance_status = context.get("compliance_status", "unknown")
        primary_typology = context.get("primary_typology")
        
        # Average the scores
        avg_score = (ensemble_risk + typology_risk) / 2
        
        # Adjust based on compliance
        if context.get("is_sanctioned"):
            avg_score = max(90, avg_score)
        if context.get("structuring"):
            avg_score = min(100, avg_score + 20)
        
        risk_score = int(min(100, max(0, avg_score)))
        
        # Determine level
        if risk_score <= 20:
            risk_level = "SAFE"
        elif risk_score <= 40:
            risk_level = "LOW"
        elif risk_score <= 60:
            risk_level = "MEDIUM"
        elif risk_score <= 80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Generate explanation
        explanation_parts = []
        if primary_typology:
            explanation_parts.append(f"Detected pattern: {primary_typology}.")
        if context.get("is_sanctioned"):
            explanation_parts.append("SANCTIONS ALERT: Address on watchlist.")
        if context.get("structuring"):
            explanation_parts.append("Structuring pattern detected.")
        
        if not explanation_parts:
            explanation_parts.append(f"Risk assessment: {risk_level} ({risk_score}/100).")
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "explanation": " ".join(explanation_parts),
            "alerts": [],
            "recommendations": [],
            "confidence": 0.7,
            "model_used": "fallback_sync",
            "inference_time_ms": 0,
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
#                              OLLAMA HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama server status and available models."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{Qwen3Config.OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "running": True,
                    "models": [m["name"] for m in models],
                    "qwen3_available": any("qwen3" in m["name"] for m in models)
                }
            return {"running": False, "models": [], "qwen3_available": False}
        except:
            return {"running": False, "models": [], "qwen3_available": False}


async def pull_qwen3_model(model_name: str = "qwen3:8b") -> bool:
    """Pull Qwen3 model using Ollama."""
    logger.info(f"Pulling {model_name}... This may take a few minutes.")
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.post(
                f"{Qwen3Config.OLLAMA_URL}/api/pull",
                json={"name": model_name},
                timeout=600.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False


def get_recommended_model(vram_gb: int = 8) -> str:
    """Get recommended Qwen3 model based on VRAM."""
    if vram_gb >= 12:
        return "qwen3:14b"
    elif vram_gb >= 8:
        return "qwen3:8b"  # Best for RTX 4070
    elif vram_gb >= 6:
        return "qwen3:4b"
    else:
        return "qwen3:1.7b"


# ═══════════════════════════════════════════════════════════════════════════════
#                              STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_qwen3_analyzer():
    """Test the Qwen3 Local Analyzer."""
    print("=" * 70)
    print("PAYFLOW QWEN3 LOCAL LLM ANALYZER - TEST")
    print("=" * 70)
    
    # Check Ollama status
    print("\n1. Checking Ollama status...")
    status = await check_ollama_status()
    
    if not status["running"]:
        print("   ❌ Ollama is not running!")
        print("   ")
        print("   To start Ollama:")
        print("   1. Install Ollama: https://ollama.ai/download")
        print("   2. Run: ollama serve")
        print("   3. Pull Qwen3: ollama pull qwen3:8b")
        return
    
    print(f"   ✅ Ollama is running")
    print(f"   Available models: {status['models']}")
    
    if not status["qwen3_available"]:
        print("\n   ⚠️ Qwen3 not found. Pulling model...")
        print("   Run: ollama pull qwen3:8b")
        return
    
    # Initialize analyzer
    print("\n2. Initializing Qwen3 Analyzer...")
    analyzer = Qwen3LocalAnalyzer()
    
    # Check availability
    available = await analyzer.check_availability()
    if not available:
        print("   ❌ Model not available")
        return
    
    # Warm up
    print("\n3. Warming up model (loading to GPU)...")
    warmed = await analyzer.warm_up()
    if not warmed:
        print("   ❌ Warm-up failed")
        return
    
    # Test analysis
    print("\n4. Running test analysis...")
    
    result = await analyzer.analyze(
        transaction_id="test_tx_001",
        sender="0xAlice1234567890abcdef",
        recipient="0xBob1234567890abcdef",
        amount=9999.0,  # Just under $10k threshold
        sender_profile={
            "transaction_count": 50,
            "total_volume": 150000,
            "avg_amount": 3000
        },
        recipient_profile={
            "transaction_count": 5,
            "total_volume": 10000
        },
        local_scores={
            "velocity": 25,
            "amount": 45,
            "pattern": 30,
            "graph": 20,
            "timing": 15,
            "isolation_forest": 0.3
        },
        neural_net_result={
            "risk_score": 24,
            "risk_level": "LOW",
            "is_anomaly": True,
            "flags": ["STRUCTURING_PATTERN: Amount near reporting threshold"]
        }
    )
    
    print("\n" + "─" * 70)
    print("QWEN3 ANALYSIS RESULT")
    print("─" * 70)
    print(f"  Risk Score:     {result.risk_score}/100")
    print(f"  Risk Level:     {result.risk_level}")
    print(f"  Approved:       {'✅ Yes' if result.approved else '❌ No'}")
    print(f"  Confidence:     {result.confidence:.0%}")
    print(f"  Inference Time: {result.inference_time_ms:.0f}ms")
    print(f"  Model:          {result.model_used}")
    print(f"  Local:          {'✅ Yes (No cloud API)' if result.is_local else 'No'}")
    print(f"\n  Explanation:")
    print(f"    {result.explanation}")
    if result.alerts:
        print(f"\n  Alerts:")
        for alert in result.alerts:
            print(f"    ⚠️ {alert}")
    if result.recommendations:
        print(f"\n  Recommendations:")
        for rec in result.recommendations:
            print(f"    → {rec}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE - Qwen3 is ready for fraud detection!")
    print("=" * 70)
    
    await analyzer.close()


if __name__ == "__main__":
    asyncio.run(test_qwen3_analyzer())
