"""
PayFlow AI Fraud Detection API
FastAPI-based REST API for real-time fraud scoring

ENDPOINTS:
- POST /analyze - Analyze a transaction
- GET /wallet/{address} - Get wallet risk profile
- GET /stats - Get fraud detection statistics
- POST /blacklist - Add address to blacklist
- POST /whitelist - Add address to whitelist
- GET /health - Health check
"""

import os
import time
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fraudDetector import get_detector, TransactionAnalysis, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudAPI")

# Initialize FastAPI
app = FastAPI(
    title="PayFlow AI Fraud Detection API",
    description="Real-time ML-based risk scoring for stablecoin transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TransactionRequest(BaseModel):
    """Request model for transaction analysis."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    sender: str = Field(..., description="Sender wallet address")
    recipient: str = Field(..., description="Recipient wallet address")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "0x123abc",
                "sender": "0xAlice...",
                "recipient": "0xBob...",
                "amount": 1000.0
            }
        }


class TransactionResponse(BaseModel):
    """Response model for transaction analysis."""
    transaction_id: str
    sender: str
    recipient: str
    amount: float
    
    # Risk scores
    overall_score: int = Field(..., ge=0, le=100)
    velocity_score: int = Field(..., ge=0, le=100)
    amount_score: int = Field(..., ge=0, le=100)
    pattern_score: int = Field(..., ge=0, le=100)
    graph_score: int = Field(..., ge=0, le=100)
    timing_score: int = Field(..., ge=0, le=100)
    
    # Classification
    risk_level: str
    approved: bool
    flagged: bool
    blocked: bool
    
    # Explanations
    explanations: List[str]
    
    # Metadata
    analysis_time_ms: float
    model_version: str


class WalletRiskResponse(BaseModel):
    """Response model for wallet risk profile."""
    address: str
    risk_score: int
    risk_level: str
    peak_risk_score: Optional[int] = 0
    is_blacklisted: bool
    is_whitelisted: bool
    transaction_count: int
    total_volume: Optional[float] = 0
    avg_amount: Optional[float] = 0


class StatisticsResponse(BaseModel):
    """Response model for fraud detection statistics."""
    total_analyses: int
    total_blocked: int
    total_flagged: int
    total_profiles: int
    blacklist_size: int
    whitelist_size: int
    model_version: str
    block_rate: float
    flag_rate: float


class BlacklistRequest(BaseModel):
    """Request model for blacklisting an address."""
    address: str
    reason: Optional[str] = "Manual blacklist"


class OracleSubmission(BaseModel):
    """Format for submitting to FraudOracle smart contract."""
    transactionId: str
    sender: str
    recipient: str
    amount: int  # In token decimals
    velocityScore: int
    amountScore: int
    patternScore: int
    graphScore: int
    timingScore: int
    overallScore: int
    approved: bool
    blocked: bool
    modelVersion: str


class BatchAnalysisRequest(BaseModel):
    """Request model for batch transaction analysis."""
    transactions: List[TransactionRequest]


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    results: List[TransactionResponse]
    total_processed: int
    total_blocked: int
    total_flagged: int
    processing_time_ms: float


# API Endpoints

@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "service": "PayFlow AI Fraud Detection",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    detector = get_detector()
    return {
        "status": "healthy",
        "model_version": detector.MODEL_VERSION,
        "uptime": "operational",
        "total_analyses": detector.total_analyses
    }


@app.post("/analyze", response_model=TransactionResponse)
async def analyze_transaction(request: TransactionRequest):
    """
    Analyze a single transaction for fraud risk.
    
    Returns comprehensive risk scores and decision.
    """
    start_time = time.time()
    detector = get_detector()
    
    try:
        analysis = detector.analyze_transaction(
            transaction_id=request.transaction_id,
            sender=request.sender,
            recipient=request.recipient,
            amount=request.amount,
            timestamp=request.timestamp
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return TransactionResponse(
            transaction_id=analysis.transaction_id,
            sender=analysis.sender,
            recipient=analysis.recipient,
            amount=analysis.amount,
            overall_score=analysis.overall_score,
            velocity_score=analysis.velocity_score,
            amount_score=analysis.amount_score,
            pattern_score=analysis.pattern_score,
            graph_score=analysis.graph_score,
            timing_score=analysis.timing_score,
            risk_level=analysis.risk_level.name,
            approved=analysis.approved,
            flagged=analysis.flagged,
            blocked=analysis.blocked,
            explanations=analysis.explanations,
            analysis_time_ms=processing_time,
            model_version=detector.MODEL_VERSION
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple transactions in batch.
    
    More efficient for processing multiple transactions.
    """
    start_time = time.time()
    detector = get_detector()
    
    results = []
    total_blocked = 0
    total_flagged = 0
    
    for tx in request.transactions:
        analysis = detector.analyze_transaction(
            transaction_id=tx.transaction_id,
            sender=tx.sender,
            recipient=tx.recipient,
            amount=tx.amount,
            timestamp=tx.timestamp
        )
        
        if analysis.blocked:
            total_blocked += 1
        if analysis.flagged:
            total_flagged += 1
        
        results.append(TransactionResponse(
            transaction_id=analysis.transaction_id,
            sender=analysis.sender,
            recipient=analysis.recipient,
            amount=analysis.amount,
            overall_score=analysis.overall_score,
            velocity_score=analysis.velocity_score,
            amount_score=analysis.amount_score,
            pattern_score=analysis.pattern_score,
            graph_score=analysis.graph_score,
            timing_score=analysis.timing_score,
            risk_level=analysis.risk_level.name,
            approved=analysis.approved,
            flagged=analysis.flagged,
            blocked=analysis.blocked,
            explanations=analysis.explanations,
            analysis_time_ms=0,
            model_version=detector.MODEL_VERSION
        ))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchAnalysisResponse(
        results=results,
        total_processed=len(results),
        total_blocked=total_blocked,
        total_flagged=total_flagged,
        processing_time_ms=processing_time
    )


@app.get("/wallet/{address}", response_model=WalletRiskResponse)
async def get_wallet_risk(address: str):
    """
    Get risk profile for a specific wallet address.
    """
    detector = get_detector()
    risk = detector.get_wallet_risk(address)
    
    return WalletRiskResponse(**risk)


@app.get("/stats", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get fraud detection statistics.
    """
    detector = get_detector()
    stats = detector.get_statistics()
    
    return StatisticsResponse(**stats)


@app.post("/blacklist")
async def blacklist_address(request: BlacklistRequest):
    """
    Add an address to the blacklist.
    """
    detector = get_detector()
    detector.blacklist_address(request.address, request.reason)
    
    return {
        "success": True,
        "address": request.address,
        "action": "blacklisted",
        "reason": request.reason
    }


@app.post("/whitelist")
async def whitelist_address(request: BlacklistRequest):
    """
    Add an address to the whitelist.
    """
    detector = get_detector()
    detector.whitelist_address(request.address)
    
    return {
        "success": True,
        "address": request.address,
        "action": "whitelisted"
    }


@app.delete("/blacklist/{address}")
async def remove_from_blacklist(address: str):
    """
    Remove an address from the blacklist.
    """
    detector = get_detector()
    detector.blacklist.discard(address.lower())
    
    return {
        "success": True,
        "address": address,
        "action": "removed_from_blacklist"
    }


@app.post("/oracle-format", response_model=OracleSubmission)
async def get_oracle_format(request: TransactionRequest):
    """
    Analyze transaction and return in format for FraudOracle smart contract.
    
    This is what gets submitted on-chain.
    """
    detector = get_detector()
    
    analysis = detector.analyze_transaction(
        transaction_id=request.transaction_id,
        sender=request.sender,
        recipient=request.recipient,
        amount=request.amount,
        timestamp=request.timestamp
    )
    
    oracle_data = detector.to_oracle_format(analysis)
    
    return OracleSubmission(**oracle_data)


@app.post("/known-bad-actors")
async def add_known_bad_actor(request: BlacklistRequest):
    """
    Add a known bad actor (from Chainalysis, etc).
    This affects graph analysis for connected wallets.
    """
    detector = get_detector()
    detector.add_known_bad_actor(request.address)
    
    return {
        "success": True,
        "address": request.address,
        "action": "added_as_known_bad_actor"
    }


# Run with: uvicorn fraudApi:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
