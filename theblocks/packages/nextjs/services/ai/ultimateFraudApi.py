"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PAYFLOW ULTIMATE FRAUD API SERVICE                                     ║
║                                                                                           ║
║   🚀 FastAPI REST API for Ultimate Fraud Detection Engine                               ║
║   🧠 Full GPU Acceleration with Qwen3:8B Thinking Mode                                   ║
║   ⚡ Optimized for <300ms Response Time                                                  ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import time
import logging

# Import our ultimate engine
from ultimateFraudEngine import (
    UltimateFraudEngine,
    Transaction,
    FraudAnalysis,
    get_engine,
    GPUConfig,
    RiskLevel,
    FraudType,
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UltimateFraudAPI")


# ═══════════════════════════════════════════════════════════════════════════════
#                           PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionRequest(BaseModel):
    """Request model for transaction analysis."""
    tx_id: Optional[str] = Field(None, description="Transaction ID (auto-generated if not provided)")
    sender: str = Field(..., description="Sender wallet address")
    recipient: str = Field(..., description="Recipient wallet address")
    amount: float = Field(..., gt=0, description="Transaction amount")
    token: str = Field("USDC", description="Token symbol")
    chain: str = Field("ethereum", description="Blockchain network")
    gas_price: Optional[float] = Field(None, description="Gas price in Gwei")
    block_number: Optional[int] = Field(None, description="Block number")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    force_thinking: bool = Field(False, description="Force deep analysis with thinking mode")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender": "0x742d35Cc6634C0532925a3b844Bc9e7595f9fD81",
                "recipient": "0x53d284357ec70cE289D6D64134DfAc8E511c8a3D",
                "amount": 15000.0,
                "token": "USDC",
                "chain": "ethereum",
                "force_thinking": False
            }
        }


class BatchTransactionRequest(BaseModel):
    """Request model for batch transaction analysis."""
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=100)
    force_thinking: bool = Field(False, description="Force thinking mode for all transactions")


class FraudAnalysisResponse(BaseModel):
    """Response model for fraud analysis."""
    tx_id: str
    timestamp: float
    overall_score: int = Field(..., ge=0, le=100)
    risk_level: str
    risk_emoji: str
    component_scores: Dict[str, int]
    detected_typologies: List[Dict[str, Any]]
    primary_typology: Optional[str]
    ai_reasoning: str
    ai_confidence: float
    thinking_used: bool
    decisions: Dict[str, bool]
    explanations: List[str]
    recommendations: List[str]
    performance: Dict[str, float]


class BlacklistWhitelistRequest(BaseModel):
    """Request model for blacklist/whitelist operations."""
    address: str = Field(..., description="Wallet address to add/remove")
    reason: Optional[str] = Field(None, description="Reason for the action")


class WalletProfileResponse(BaseModel):
    """Response model for wallet profile."""
    address: str
    transaction_count: int
    total_volume: float
    avg_amount: float
    risk_history: List[Dict[str, Any]]
    is_blacklisted: bool
    is_whitelisted: bool
    is_mixer: bool
    is_exchange: bool
    age_days: float


class EngineStatsResponse(BaseModel):
    """Response model for engine statistics."""
    version: str
    total_analyses: int
    total_blocked: int
    total_flagged: int
    block_rate: float
    flag_rate: float
    profiles_tracked: int
    blacklist_size: int
    whitelist_size: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_loaded: bool
    gpu_config: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    gpu_layers: int
    uptime_seconds: float


# ═══════════════════════════════════════════════════════════════════════════════
#                           FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Global engine reference
engine: Optional[UltimateFraudEngine] = None
startup_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global engine, startup_time
    
    # Startup
    logger.info("🚀 Starting Ultimate Fraud API Service...")
    startup_time = time.time()
    engine = await get_engine()
    logger.info(f"✅ Engine v{engine.VERSION} loaded with GPU acceleration")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Ultimate Fraud API Service...")
    if engine:
        await engine.close()
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="PayFlow Ultimate Fraud Detection API",
    description="""
# 🛡️ Ultimate Fraud Detection Engine

**Real-time AI-powered fraud detection for Web3 transactions**

## Features
- 🚀 Full GPU acceleration (RTX 4070, 8GB VRAM dedicated)
- 🧠 Qwen3:8B with thinking mode for maximum accuracy
- ⚡ <300ms response time target
- 🎯 15 fraud typologies detected
- 📊 Real-time wallet profiling

## Fraud Typologies Detected
1. Rug Pulls ($8B market impact)
2. Pig Butchering ($7.5B)
3. Mixer/Tumbling ($5.6B)
4. Chain Obfuscation ($4.3B)
5. Fake Tokens ($2.8B)
6. Flash Loan Attacks ($1.9B)
7. Wash Trading ($1.5B)
8. Structuring/Smurfing ($1.2B)
9. Velocity Attacks ($0.9B)
10. Peel Chains ($0.7B)
11. Dusting Attacks ($0.5B)
12. Address Poisoning ($0.4B)
13. Approval Exploits ($0.3B)
14. SIM Swap ($0.2B)
15. Romance Scams ($0.2B)

## Hackxios 2K25 - PayFlow Protocol
    """,
    version="3.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#                           API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root():
    """API root endpoint."""
    return {
        "name": "PayFlow Ultimate Fraud Detection API",
        "version": "3.0.0",
        "description": "Real-time AI-powered fraud detection for Web3 transactions",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    global engine, startup_time
    
    return HealthResponse(
        status="healthy" if engine and engine.model_loaded else "degraded",
        version=engine.VERSION if engine else "unknown",
        model_loaded=engine.model_loaded if engine else False,
        gpu_layers=GPUConfig.GPU_LAYERS,
        uptime_seconds=time.time() - startup_time,
    )


@app.post("/analyze", response_model=FraudAnalysisResponse, tags=["Analysis"])
async def analyze_transaction(request: TransactionRequest):
    """
    🔍 Analyze a single transaction for fraud risk.
    
    Uses adaptive thinking mode:
    - Fast path: Low-risk transactions (< 40 initial score)
    - Thinking path: Higher risk or when force_thinking=true
    
    Returns comprehensive fraud analysis including:
    - Risk score (0-100)
    - Risk level (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
    - Detected fraud typologies
    - AI reasoning (when thinking mode used)
    - Decisions (approved, flagged, blocked)
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Create transaction object
    tx = Transaction(
        tx_id=request.tx_id or f"tx_{int(time.time()*1000)}",
        sender=request.sender,
        recipient=request.recipient,
        amount=request.amount,
        token=request.token,
        chain=request.chain,
        gas_price=request.gas_price,
        block_number=request.block_number,
        timestamp=request.timestamp or time.time(),
    )
    
    # Analyze
    result = await engine.analyze(tx, force_thinking=request.force_thinking)
    
    return FraudAnalysisResponse(**result.to_dict())


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(request: BatchTransactionRequest):
    """
    🔍 Analyze multiple transactions in batch.
    
    Processes up to 100 transactions in a single request.
    Returns array of fraud analysis results.
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    results = []
    
    for tx_req in request.transactions:
        tx = Transaction(
            tx_id=tx_req.tx_id or f"tx_{int(time.time()*1000)}_{len(results)}",
            sender=tx_req.sender,
            recipient=tx_req.recipient,
            amount=tx_req.amount,
            token=tx_req.token,
            chain=tx_req.chain,
            gas_price=tx_req.gas_price,
            block_number=tx_req.block_number,
            timestamp=tx_req.timestamp or time.time(),
        )
        
        result = await engine.analyze(tx, force_thinking=request.force_thinking)
        results.append(result.to_dict())
    
    return {
        "count": len(results),
        "results": results,
        "summary": {
            "approved": sum(1 for r in results if r["decisions"]["approved"] and not r["decisions"]["flagged"]),
            "flagged": sum(1 for r in results if r["decisions"]["flagged"]),
            "blocked": sum(1 for r in results if r["decisions"]["blocked"]),
        }
    }


@app.get("/wallet/{address}", response_model=WalletProfileResponse, tags=["Profiles"])
async def get_wallet_profile(address: str):
    """
    👤 Get wallet risk profile.
    
    Returns behavioral profile for a wallet address including:
    - Transaction history summary
    - Risk score history
    - Blacklist/whitelist status
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    profile = engine.profiler.get_profile(address)
    
    return WalletProfileResponse(
        address=profile["address"],
        transaction_count=profile["transaction_count"],
        total_volume=profile["total_volume"],
        avg_amount=profile["avg_amount"],
        risk_history=profile["risk_history"][-20:],  # Last 20 entries
        is_blacklisted=profile["is_blacklisted"],
        is_whitelisted=profile["is_whitelisted"],
        is_mixer=profile["is_mixer"],
        is_exchange=profile["is_exchange"],
        age_days=(time.time() - profile["created_at"]) / 86400,
    )


@app.post("/blacklist", tags=["Management"])
async def add_to_blacklist(request: BlacklistWhitelistRequest):
    """
    🚫 Add address to blacklist.
    
    Blacklisted addresses will be immediately flagged
    with a high risk score on any transaction.
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    address = request.address.lower()
    engine.profiler.blacklist.add(address)
    
    # Update profile
    profile = engine.profiler.get_profile(address)
    profile["is_blacklisted"] = True
    
    logger.info(f"🚫 Address blacklisted: {address} | Reason: {request.reason}")
    
    return {
        "success": True,
        "address": address,
        "action": "blacklisted",
        "reason": request.reason,
    }


@app.delete("/blacklist/{address}", tags=["Management"])
async def remove_from_blacklist(address: str):
    """
    ✅ Remove address from blacklist.
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    address = address.lower()
    engine.profiler.blacklist.discard(address)
    
    # Update profile
    profile = engine.profiler.get_profile(address)
    profile["is_blacklisted"] = False
    
    logger.info(f"✅ Address removed from blacklist: {address}")
    
    return {
        "success": True,
        "address": address,
        "action": "removed_from_blacklist",
    }


@app.post("/whitelist", tags=["Management"])
async def add_to_whitelist(request: BlacklistWhitelistRequest):
    """
    ⭐ Add address to whitelist.
    
    Whitelisted addresses (e.g., known exchanges)
    will receive reduced scrutiny.
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    address = request.address.lower()
    engine.profiler.whitelist.add(address)
    
    # Update profile
    profile = engine.profiler.get_profile(address)
    profile["is_whitelisted"] = True
    
    logger.info(f"⭐ Address whitelisted: {address} | Reason: {request.reason}")
    
    return {
        "success": True,
        "address": address,
        "action": "whitelisted",
        "reason": request.reason,
    }


@app.delete("/whitelist/{address}", tags=["Management"])
async def remove_from_whitelist(address: str):
    """
    ❌ Remove address from whitelist.
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    address = address.lower()
    engine.profiler.whitelist.discard(address)
    
    # Update profile
    profile = engine.profiler.get_profile(address)
    profile["is_whitelisted"] = False
    
    logger.info(f"❌ Address removed from whitelist: {address}")
    
    return {
        "success": True,
        "address": address,
        "action": "removed_from_whitelist",
    }


@app.get("/stats", response_model=EngineStatsResponse, tags=["Statistics"])
async def get_engine_stats():
    """
    📊 Get engine statistics.
    
    Returns performance metrics including:
    - Analysis counts
    - Block/flag rates
    - Latency statistics
    - GPU configuration
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    stats = engine.get_stats()
    return EngineStatsResponse(**stats)


@app.get("/typologies", tags=["Info"])
async def get_typologies():
    """
    📋 Get list of detected fraud typologies.
    
    Returns all 15 fraud typologies with:
    - Market impact
    - Detection target rate
    """
    return {
        "count": 15,
        "typologies": [
            {
                "code": ft.code,
                "name": ft.name,
                "market_impact_millions": ft.market_impact_m,
                "detection_target_percent": ft.detection_target,
            }
            for ft in FraudType
        ]
    }


@app.get("/config", tags=["Info"])
async def get_gpu_config():
    """
    ⚙️ Get GPU configuration.
    
    Returns current GPU acceleration settings.
    """
    return {
        "model": GPUConfig.MODEL,
        "ollama_url": GPUConfig.OLLAMA_URL,
        "gpu_layers": GPUConfig.GPU_LAYERS,
        "context_size": GPUConfig.NUM_CTX,
        "batch_size": GPUConfig.NUM_BATCH,
        "threads": GPUConfig.NUM_THREAD,
        "keep_alive": GPUConfig.KEEP_ALIVE,
        "thinking_temperature": GPUConfig.TEMPERATURE_THINKING,
        "fast_temperature": GPUConfig.TEMPERATURE_FAST,
        "thinking_budget_tokens": GPUConfig.THINKING_BUDGET_TOKENS,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║     🛡️  PAYFLOW ULTIMATE FRAUD DETECTION API                                            ║
║                                                                                           ║
║     🚀 Full GPU Acceleration (RTX 4070, 8GB VRAM)                                        ║
║     🧠 Qwen3:8B with Thinking Mode                                                        ║
║     ⚡ Target: <300ms Response Time                                                      ║
║                                                                                           ║
║     Starting server on http://localhost:8000                                             ║
║     API Documentation: http://localhost:8000/docs                                        ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "ultimateFraudApi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable for production
        workers=1,     # Single worker for GPU
        log_level="info",
    )
