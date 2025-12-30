"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PAYFLOW FRAUD DETECTION API - PRODUCTION v5.0                          ║
║                                                                                           ║
║   🚀 FastAPI REST Service for Lightning Fraud Detection                                  ║
║                                                                                           ║
║   ENDPOINTS:                                                                              ║
║   POST /analyze     - Analyze single transaction                                         ║
║   POST /batch       - Analyze multiple transactions                                      ║
║   POST /blacklist   - Add address to blacklist                                           ║
║   POST /whitelist   - Add address to whitelist                                           ║
║   GET  /stats       - Get engine statistics                                              ║
║   GET  /health      - Health check                                                       ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import time
import logging

from lightningFraudEngine import LightningFraudEngine, BLACKLIST, WHITELIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudAPI")

# ═══════════════════════════════════════════════════════════════════════════════
#                           FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="PayFlow Fraud Detection API",
    description="Lightning-fast blockchain fraud detection with GPU-accelerated AI",
    version="5.0.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine instance
engine: Optional[LightningFraudEngine] = None


# ═══════════════════════════════════════════════════════════════════════════════
#                           MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionRequest(BaseModel):
    """Request to analyze a transaction."""
    tx_id: str = Field(..., description="Transaction ID")
    sender: str = Field(..., description="Sender address")
    recipient: str = Field(..., description="Recipient address")
    amount: float = Field(..., description="Transaction amount in USD")
    token: str = Field(default="USDC", description="Token symbol")
    chain: str = Field(default="ethereum", description="Blockchain")
    sender_tx_count: Optional[int] = Field(default=None, description="Sender's transaction count")
    is_contract: bool = Field(default=False, description="Is recipient a contract")


class BatchRequest(BaseModel):
    """Request to analyze multiple transactions."""
    transactions: List[TransactionRequest]


class AddressRequest(BaseModel):
    """Request to add address to list."""
    address: str = Field(..., description="Address to add")
    reason: Optional[str] = Field(default=None, description="Reason for listing")


class FraudResponse(BaseModel):
    """Response from fraud analysis."""
    tx_id: str
    score: int
    risk_level: str
    risk_emoji: str
    mode: str
    approved: bool
    blocked: bool
    flagged: bool
    reasons: List[str]
    latency_ms: float


class BatchResponse(BaseModel):
    """Response from batch analysis."""
    results: List[FraudResponse]
    total_transactions: int
    blocked_count: int
    flagged_count: int
    approved_count: int
    total_latency_ms: float
    average_latency_ms: float


class StatsResponse(BaseModel):
    """Engine statistics."""
    total_transactions: int
    instant_count: int
    quick_count: int
    verify_count: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    blacklist_size: int
    whitelist_size: int


# ═══════════════════════════════════════════════════════════════════════════════
#                           STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    """Initialize fraud engine on startup."""
    global engine
    logger.info("🚀 Starting PayFlow Fraud Detection API...")
    
    engine = LightningFraudEngine()
    await engine.init()
    
    logger.info("✅ Fraud engine ready!")


# ═══════════════════════════════════════════════════════════════════════════════
#                           ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "5.0.0-lightning",
        "engine_ready": engine is not None and engine.ai.ready,
    }


@app.post("/analyze", response_model=FraudResponse)
async def analyze_transaction(tx: TransactionRequest):
    """Analyze a single transaction for fraud."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    tx_dict = tx.model_dump()
    result = await engine.analyze(tx_dict)
    
    return FraudResponse(
        tx_id=result.tx_id,
        score=result.score,
        risk_level=result.risk.label,
        risk_emoji=result.risk.emoji,
        mode=result.mode.value,
        approved=result.approved,
        blocked=result.blocked,
        flagged=result.flagged,
        reasons=result.reasons,
        latency_ms=result.latency_ms,
    )


@app.post("/batch", response_model=BatchResponse)
async def analyze_batch(batch: BatchRequest):
    """Analyze multiple transactions."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start = time.perf_counter()
    
    results = []
    blocked = 0
    flagged = 0
    approved = 0
    
    for tx in batch.transactions:
        result = await engine.analyze(tx.model_dump())
        
        results.append(FraudResponse(
            tx_id=result.tx_id,
            score=result.score,
            risk_level=result.risk.label,
            risk_emoji=result.risk.emoji,
            mode=result.mode.value,
            approved=result.approved,
            blocked=result.blocked,
            flagged=result.flagged,
            reasons=result.reasons,
            latency_ms=result.latency_ms,
        ))
        
        if result.blocked:
            blocked += 1
        if result.flagged:
            flagged += 1
        if result.approved:
            approved += 1
    
    total_latency = (time.perf_counter() - start) * 1000
    
    return BatchResponse(
        results=results,
        total_transactions=len(results),
        blocked_count=blocked,
        flagged_count=flagged,
        approved_count=approved,
        total_latency_ms=total_latency,
        average_latency_ms=total_latency / max(1, len(results)),
    )


@app.post("/blacklist")
async def add_to_blacklist(req: AddressRequest):
    """Add address to blacklist."""
    addr = req.address.lower()
    BLACKLIST.add(addr)
    
    logger.info(f"🚫 Blacklisted: {addr[:10]}... - {req.reason or 'No reason'}")
    
    return {
        "success": True,
        "address": addr,
        "action": "blacklisted",
        "total_blacklisted": len(BLACKLIST),
    }


@app.post("/whitelist")
async def add_to_whitelist(req: AddressRequest):
    """Add address to whitelist."""
    addr = req.address.lower()
    WHITELIST.add(addr)
    
    logger.info(f"✅ Whitelisted: {addr[:10]}... - {req.reason or 'No reason'}")
    
    return {
        "success": True,
        "address": addr,
        "action": "whitelisted",
        "total_whitelisted": len(WHITELIST),
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get engine statistics."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    stats = engine.get_stats()
    
    return StatsResponse(
        total_transactions=stats.get("total", 0),
        instant_count=stats.get("instant", 0),
        quick_count=stats.get("quick", 0),
        verify_count=stats.get("verify", 0),
        average_latency_ms=stats.get("avg_ms", 0),
        p50_latency_ms=stats.get("p50_ms", 0),
        p95_latency_ms=stats.get("p95_ms", 0),
        blacklist_size=len(BLACKLIST),
        whitelist_size=len(WHITELIST),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                           MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   🚀 PAYFLOW FRAUD DETECTION API                                                         ║
║                                                                                           ║
║   Starting server on http://localhost:8000                                               ║
║   Docs available at http://localhost:8000/docs                                           ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
