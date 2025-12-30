"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                  ULTIMATE HYBRID ENGINE - REST API v2.0                                  ║
║                                                                                           ║
║   Production FastAPI service for PayFlow fraud detection                                  ║
║                                                                                           ║
║   ENDPOINTS:                                                                              ║
║   POST /analyze         - Single transaction analysis                                     ║
║   POST /batch           - Batch transaction analysis                                      ║
║   POST /blacklist       - Add address to blacklist                                        ║
║   POST /whitelist       - Add address to whitelist                                        ║
║   GET  /stats           - Get engine statistics                                           ║
║   GET  /health          - Health check                                                    ║
║                                                                                           ║
║   Hackxios 2K25 - PayFlow Protocol                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import logging
import time
import json

from ultimateHybridEngine import UltimateHybridEngine, AddressDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HybridAPI')

app = FastAPI(
    title="PayFlow Ultimate Hybrid Fraud Detection API",
    description="Neural heuristics + 15-typology + GPU AI verification",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[UltimateHybridEngine] = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial connection success message with stats structure expected by frontend
        await websocket.send_json({
            "type": "connected",
            "data": {
                "message": "Connected to PayFlow Hybrid AI Oracle",
                "timestamp": time.time(),
                "oracle_address": "0xHybridOracleV2...",
                "stats": {
                    "total_analyses": 0,
                    "total_blocked": 0,
                    "total_flagged": 0,
                    "avg_latency_ms": 0,
                    "qwen3_calls": 0,
                    "neural_net_predictions": 0,
                    "cache_hits": 0,
                    "total_profiles": 0,
                    "blacklist_size": 0,
                    "oracle_address": "0xHybridOracleV2...",
                    "qwen3_enabled": True,
                    "ml_trained": True,
                    "websocket_connections": len(manager.active_connections)
                }
            }
        })
        
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle ping/pong or other client messages
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "data": time.time()})
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


class TransactionRequest(BaseModel):
    tx_id: Optional[str] = None
    sender: str = Field(..., description="Sender wallet address")
    recipient: str = Field(..., description="Recipient wallet address")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    timestamp: Optional[float] = None
    token_symbol: Optional[str] = "USDC"
    chain_id: Optional[int] = 1

    class Config:
        json_schema_extra = {
            "example": {
                "sender": "0xAlice123...",
                "recipient": "0xBob456...",
                "amount": 5000.00,
                "token_symbol": "USDC"
            }
        }


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class AddressAction(BaseModel):
    address: str
    reason: Optional[str] = None


class AnalysisResponse(BaseModel):
    tx_id: str
    score: int
    risk_level: str
    risk_emoji: str
    risk_color: str
    mode: str
    approved: bool
    blocked: bool
    flagged: bool
    reasons: List[str]
    typologies: List[str]
    confidence: float
    latency_ms: float


class BatchResponse(BaseModel):
    total: int
    blocked: int
    flagged: int
    approved: int
    avg_latency_ms: float
    results: List[AnalysisResponse]


class StatsResponse(BaseModel):
    version: str
    total_analyzed: int
    instant_count: int
    heuristic_count: int
    typology_count: int
    ai_verify_count: int
    blocked_count: int
    flagged_count: int
    approved_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    gpu_ready: bool
    version: str


# Startup time for uptime calculation
startup_time = time.time()


@app.on_event("startup")
async def startup():
    global engine
    logger.info("🚀 Starting Ultimate Hybrid Engine API...")
    engine = UltimateHybridEngine()
    await engine.init()
    logger.info("✅ API ready!")


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "PayFlow Ultimate Hybrid Fraud Detection API",
        "version": engine.VERSION if engine else "loading",
        "architecture": "4-Layer Hybrid: Rules → Neural Heuristics → Typology → GPU AI",
        "typologies": 15,
        "analyzers": 5,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if engine else "starting",
        engine_ready=engine is not None,
        gpu_ready=engine.gpu.ready if engine else False,
        version=engine.VERSION if engine else "0.0.0",
    )


@app.post("/analyze", tags=["Analysis"])
async def analyze_transaction(tx: TransactionRequest):
    """
    Analyze a single transaction through the 4-layer hybrid architecture.
    
    Returns risk score, decision, detected typologies, and analysis latency.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        tx_dict = {
            "tx_id": tx.tx_id,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "timestamp": tx.timestamp or time.time(),
        }
        
        result = await engine.analyze(tx_dict)
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error analyzing transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse, tags=["Analysis"])
async def batch_analyze(batch: BatchRequest):
    """
    Analyze multiple transactions in a batch.
    
    More efficient than individual calls for high-volume processing.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    results = []
    blocked = 0
    flagged = 0
    approved = 0
    total_latency = 0
    
    for tx in batch.transactions:
        tx_dict = {
            "tx_id": tx.tx_id,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "timestamp": tx.timestamp or time.time(),
        }
        
        result = await engine.analyze(tx_dict)
        results.append(AnalysisResponse(**result.to_dict()))
        
        if result.blocked:
            blocked += 1
        if result.flagged:
            flagged += 1
        if result.approved:
            approved += 1
        total_latency += result.latency_ms
    
    return BatchResponse(
        total=len(batch.transactions),
        blocked=blocked,
        flagged=flagged,
        approved=approved,
        avg_latency_ms=total_latency / len(batch.transactions) if batch.transactions else 0,
        results=results,
    )


@app.post("/blacklist", tags=["Management"])
async def add_to_blacklist(action: AddressAction):
    """
    Add an address to the blacklist for instant blocking.
    """
    addr = action.address.lower()
    AddressDatabase.BLACKLIST.add(addr)
    
    return {
        "success": True,
        "address": addr,
        "action": "blacklisted",
        "reason": action.reason,
        "total_blacklisted": len(AddressDatabase.BLACKLIST),
    }


@app.post("/whitelist", tags=["Management"])
async def add_to_whitelist(action: AddressAction):
    """
    Add an address to the whitelist for instant approval.
    """
    addr = action.address.lower()
    AddressDatabase.WHITELIST.add(addr)
    
    return {
        "success": True,
        "address": addr,
        "action": "whitelisted",
        "reason": action.reason,
        "total_whitelisted": len(AddressDatabase.WHITELIST),
    }


@app.delete("/blacklist/{address}", tags=["Management"])
async def remove_from_blacklist(address: str):
    """Remove an address from the blacklist."""
    addr = address.lower()
    if addr in AddressDatabase.BLACKLIST:
        AddressDatabase.BLACKLIST.remove(addr)
        return {"success": True, "address": addr, "action": "removed from blacklist"}
    return {"success": False, "address": addr, "error": "Address not in blacklist"}


@app.delete("/whitelist/{address}", tags=["Management"])
async def remove_from_whitelist(address: str):
    """Remove an address from the whitelist."""
    addr = address.lower()
    if addr in AddressDatabase.WHITELIST:
        AddressDatabase.WHITELIST.remove(addr)
        return {"success": True, "address": addr, "action": "removed from whitelist"}
    return {"success": False, "address": addr, "error": "Address not in whitelist"}


@app.get("/stats", response_model=StatsResponse, tags=["Metrics"])
async def get_statistics():
    """
    Get comprehensive engine statistics and performance metrics.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    stats = engine.get_stats()
    total = stats.get("total", 1)
    
    return StatsResponse(
        version=engine.VERSION,
        total_analyzed=total,
        instant_count=stats.get("instant", 0),
        heuristic_count=stats.get("heuristic", 0),
        typology_count=stats.get("typology", 0),
        ai_verify_count=stats.get("ai_verify", 0),
        blocked_count=stats.get("blocked", 0),
        flagged_count=stats.get("flagged", 0),
        approved_count=stats.get("approved", 0),
        avg_latency_ms=stats.get("avg_ms", 0),
        p50_latency_ms=stats.get("p50_ms", 0),
        p95_latency_ms=stats.get("p95_ms", 0),
        uptime_seconds=time.time() - startup_time,
    )


@app.get("/lists", tags=["Management"])
async def get_lists():
    """Get current blacklist and whitelist counts."""
    return {
        "blacklist_count": len(AddressDatabase.BLACKLIST),
        "whitelist_count": len(AddressDatabase.WHITELIST),
        "known_mixers": len(AddressDatabase.MIXERS),
        "known_safe_exchanges": len(AddressDatabase.SAFE_EXCHANGES),
        "flash_loan_providers": len(AddressDatabase.FLASH_LOAN_PROVIDERS),
    }


if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                           ║
║   🚀 STARTING ULTIMATE HYBRID FRAUD DETECTION API                                        ║
║                                                                                           ║
║   Architecture:                                                                           ║
║   • Layer 1: Instant Rules (<1ms)                                                        ║
║   • Layer 2: Neural Heuristics (<5ms)                                                    ║
║   • Layer 3: 15-Typology Detector (<10ms)                                                ║
║   • Layer 4: GPU AI Verification (<100ms)                                                ║
║                                                                                           ║
║   Performance Targets:                                                                    ║
║   • Average Latency: <50ms                                                               ║
║   • P95 Latency: <150ms                                                                  ║
║   • Throughput: 15,000+ tx/sec                                                           ║
║                                                                                           ║
║   Docs: http://localhost:8000/docs                                                       ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
