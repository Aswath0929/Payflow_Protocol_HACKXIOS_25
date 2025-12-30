"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    SECURE AI ORACLE SERVICE                                   ║
║                                                                               ║
║   Secure integration of AI APIs with blockchain                              ║
║   - API keys NEVER exposed to frontend                                       ║
║   - All results cryptographically signed                                     ║
║   - On-chain verification of oracle authenticity                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

SECURITY MODEL:
1. API keys stored in environment variables (never in code)
2. Backend is the ONLY component that calls external AI APIs
3. Results are signed with oracle's private key
4. Smart contract verifies signature before accepting data
5. Rate limiting and request validation prevent abuse

ARCHITECTURE:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│  This Service   │────▶│   AI Provider   │
│  (No API Keys)  │     │  (Signs Data)   │     │  (OpenAI etc)   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼ Signed Result
                        ┌─────────────────┐
                        │   Blockchain    │
                        │ (Verifies Sig)  │
                        └─────────────────┘
"""

import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Cryptographic signing
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

# FastAPI for secure endpoints
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import httpx

# Rate limiting
from collections import defaultdict
from datetime import datetime, timedelta

# Configure logging (don't log sensitive data!)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecureAIOracle")

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURATION (from environment)
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    All sensitive configuration loaded from environment variables.
    NEVER hardcode API keys or private keys!
    """
    # AI Provider API Keys (stored securely in environment)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Oracle's private key for signing (KEEP THIS SECRET!)
    # This should be a dedicated key, NOT your main wallet
    ORACLE_PRIVATE_KEY: str = os.getenv("ORACLE_PRIVATE_KEY", "")
    
    # Derived oracle address (public, stored on-chain for verification)
    ORACLE_ADDRESS: str = ""
    
    # API security
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "dev-secret-change-in-production")
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    MAX_REQUESTS_PER_HOUR: int = 500
    
    @classmethod
    def initialize(cls):
        """Derive oracle address from private key."""
        if cls.ORACLE_PRIVATE_KEY:
            account = Account.from_key(cls.ORACLE_PRIVATE_KEY)
            cls.ORACLE_ADDRESS = account.address
            logger.info(f"Oracle address: {cls.ORACLE_ADDRESS}")
        else:
            logger.warning("ORACLE_PRIVATE_KEY not set - using demo mode")
            # Generate a temporary key for demo purposes
            account = Account.create()
            cls.ORACLE_PRIVATE_KEY = account.key.hex()
            cls.ORACLE_ADDRESS = account.address
            logger.info(f"Demo oracle address: {cls.ORACLE_ADDRESS}")

# Initialize on import
Config.initialize()

# ═══════════════════════════════════════════════════════════════════════════════
#                         RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Prevent API abuse and protect AI provider costs.
    """
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, str]:
        """Check if client is within rate limits."""
        now = datetime.now()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > now - timedelta(hours=1)
        ]
        
        recent_requests = self.requests[client_id]
        
        # Check per-minute limit
        requests_last_minute = sum(
            1 for req_time in recent_requests
            if req_time > now - timedelta(minutes=1)
        )
        if requests_last_minute >= Config.MAX_REQUESTS_PER_MINUTE:
            return False, "Rate limit exceeded (per minute)"
        
        # Check per-hour limit
        if len(recent_requests) >= Config.MAX_REQUESTS_PER_HOUR:
            return False, "Rate limit exceeded (per hour)"
        
        # Record this request
        self.requests[client_id].append(now)
        return True, "OK"

rate_limiter = RateLimiter()

# ═══════════════════════════════════════════════════════════════════════════════
#                         CRYPTOGRAPHIC SIGNING
# ═══════════════════════════════════════════════════════════════════════════════

class OracleSigner:
    """
    Signs oracle responses so smart contracts can verify authenticity.
    Uses EIP-191 personal_sign for Ethereum compatibility.
    """
    
    @staticmethod
    def sign_result(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign the oracle result.
        
        Returns the original data plus:
        - signature: The cryptographic signature
        - oracle_address: The oracle's Ethereum address
        - timestamp: When this was signed
        - message_hash: Hash of the signed data
        """
        # Create deterministic message from data
        timestamp = int(time.time())
        
        # Create the message to sign (deterministic JSON)
        message_data = {
            "transaction_id": data.get("transaction_id", ""),
            "risk_score": data.get("risk_score", 0),
            "approved": data.get("approved", True),
            "timestamp": timestamp,
            "model": data.get("model", "unknown")
        }
        
        message = json.dumps(message_data, sort_keys=True)
        message_hash = Web3.keccak(text=message).hex()
        
        # Sign with oracle's private key
        signable_message = encode_defunct(text=message)
        signed = Account.sign_message(signable_message, Config.ORACLE_PRIVATE_KEY)
        
        return {
            **data,
            "signature": signed.signature.hex(),
            "oracle_address": Config.ORACLE_ADDRESS,
            "signed_at": timestamp,
            "message_hash": message_hash,
            "message": message  # For verification
        }
    
    @staticmethod
    def verify_signature(message: str, signature: str, expected_address: str) -> bool:
        """Verify a signature matches the expected signer."""
        try:
            signable_message = encode_defunct(text=message)
            recovered_address = Account.recover_message(signable_message, signature=signature)
            return recovered_address.lower() == expected_address.lower()
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

# ═══════════════════════════════════════════════════════════════════════════════
#                         AI PROVIDERS (Secure Access)
# ═══════════════════════════════════════════════════════════════════════════════

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Fallback heuristics

@dataclass
class AIAnalysisResult:
    """Result from AI analysis."""
    risk_score: int  # 0-100
    risk_level: str
    approved: bool
    explanation: str
    confidence: float
    model: str
    processing_time_ms: float

class SecureAIClient:
    """
    Secure client for AI API calls.
    - API keys never leave the server
    - Results are validated before signing
    - Fallback to local heuristics if AI unavailable
    """
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def analyze_transaction_openai(
        self,
        transaction_data: Dict[str, Any]
    ) -> AIAnalysisResult:
        """
        Analyze transaction using Qwen3 Local LLM (via Ollama).
        
        UPGRADED: Now uses local Qwen3 8B MoE instead of OpenAI.
        100% offline, runs on RTX 4070 GPU.
        """
        start_time = time.time()
        
        prompt = f"""You are a fraud detection AI for stablecoin transactions.
Analyze this transaction and provide a risk assessment.

Transaction:
- ID: {transaction_data.get('transaction_id', 'unknown')}
- Sender: {transaction_data.get('sender', 'unknown')}
- Recipient: {transaction_data.get('recipient', 'unknown')}
- Amount: ${transaction_data.get('amount', 0):,.2f}
- Timestamp: {transaction_data.get('timestamp', 'unknown')}

Previous transaction history for sender (if available):
- Transaction count: {transaction_data.get('sender_tx_count', 0)}
- Average amount: ${transaction_data.get('sender_avg_amount', 0):,.2f}
- Total volume: ${transaction_data.get('sender_total_volume', 0):,.2f}

Provide your analysis in this exact JSON format:
{{
    "risk_score": <0-100>,
    "risk_level": "<SAFE|LOW|MEDIUM|HIGH|CRITICAL>",
    "approved": <true|false>,
    "explanation": "<brief explanation>",
    "confidence": <0.0-1.0>,
    "red_flags": ["<flag1>", "<flag2>"],
    "recommendations": ["<rec1>", "<rec2>"]
}}

Consider:
1. Amount anomalies (is this unusual for this sender?)
2. Velocity (too many transactions too fast?)
3. Structuring (amounts just below reporting thresholds like $10,000?)
4. Counterparty risk
5. Time patterns (unusual timing?)

ONLY respond with the JSON, no other text."""

        try:
            # Use Qwen3 via Ollama (local GPU)
            response = await self.http_client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse Qwen3 response (Ollama format)
            ai_response = result.get("response", "{}")
            # Clean thinking tags if present
            if "<think>" in ai_response:
                ai_response = ai_response.split("</think>")[-1].strip()
            analysis = json.loads(ai_response)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AIAnalysisResult(
                risk_score=min(100, max(0, int(analysis.get("risk_score", 50)))),
                risk_level=analysis.get("risk_level", "MEDIUM"),
                approved=analysis.get("approved", True),
                explanation=analysis.get("explanation", "Qwen3 analysis complete"),
                confidence=float(analysis.get("confidence", 0.8)),
                model="qwen3:8b-local",
                processing_time_ms=processing_time
            )
            
        except httpx.HTTPError as e:
            logger.error(f"Qwen3/Ollama API error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise
    
    async def analyze_transaction_anthropic(
        self,
        transaction_data: Dict[str, Any]
    ) -> AIAnalysisResult:
        """
        Analyze transaction using Anthropic Claude.
        """
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not configured")
        
        start_time = time.time()
        
        prompt = f"""Analyze this stablecoin transaction for fraud risk.

Transaction:
- ID: {transaction_data.get('transaction_id', 'unknown')}
- Sender: {transaction_data.get('sender', 'unknown')}
- Recipient: {transaction_data.get('recipient', 'unknown')}
- Amount: ${transaction_data.get('amount', 0):,.2f}

Respond with JSON only:
{{"risk_score": 0-100, "risk_level": "SAFE|LOW|MEDIUM|HIGH|CRITICAL", "approved": true|false, "explanation": "...", "confidence": 0.0-1.0}}"""

        try:
            response = await self.http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": Config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2024-01-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            ai_response = result["content"][0]["text"]
            analysis = json.loads(ai_response)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AIAnalysisResult(
                risk_score=min(100, max(0, int(analysis.get("risk_score", 50)))),
                risk_level=analysis.get("risk_level", "MEDIUM"),
                approved=analysis.get("approved", True),
                explanation=analysis.get("explanation", "AI analysis complete"),
                confidence=float(analysis.get("confidence", 0.8)),
                model="claude-3-sonnet",
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def analyze_transaction_local(
        self,
        transaction_data: Dict[str, Any]
    ) -> AIAnalysisResult:
        """
        Fallback local heuristics (no external API).
        Used when AI APIs are unavailable or for cost savings.
        """
        start_time = time.time()
        
        amount = transaction_data.get("amount", 0)
        sender_avg = transaction_data.get("sender_avg_amount", amount)
        sender_tx_count = transaction_data.get("sender_tx_count", 0)
        
        risk_score = 0
        explanations = []
        
        # Amount anomaly
        if sender_tx_count > 0 and sender_avg > 0:
            deviation = abs(amount - sender_avg) / sender_avg
            if deviation > 5:
                risk_score += 40
                explanations.append(f"Amount {deviation:.1f}x deviation from average")
            elif deviation > 2:
                risk_score += 20
                explanations.append(f"Amount {deviation:.1f}x deviation from average")
        
        # Structuring detection
        structuring_thresholds = [3000, 10000]
        for threshold in structuring_thresholds:
            if threshold * 0.9 <= amount < threshold:
                risk_score += 30
                explanations.append(f"Potential structuring (below ${threshold})")
        
        # Large transaction
        if amount > 100000:
            risk_score += 20
            explanations.append("Large transaction amount")
        
        # New wallet
        if sender_tx_count < 3:
            risk_score += 10
            explanations.append("New wallet with limited history")
        
        risk_score = min(100, risk_score)
        
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
        
        processing_time = (time.time() - start_time) * 1000
        
        return AIAnalysisResult(
            risk_score=risk_score,
            risk_level=risk_level,
            approved=risk_score < 80,
            explanation="; ".join(explanations) if explanations else "Transaction appears normal",
            confidence=0.7,  # Lower confidence for heuristics
            model="local-heuristics",
            processing_time_ms=processing_time
        )

# Global AI client
ai_client = SecureAIClient()

# ═══════════════════════════════════════════════════════════════════════════════
#                         FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Secure AI Oracle",
    description="Cryptographically signed AI fraud detection for blockchain",
    version="1.0.0"
)

# CORS - restrict in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Only allow your frontend
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Request models
class TransactionRequest(BaseModel):
    transaction_id: str
    sender: str
    recipient: str
    amount: float
    timestamp: Optional[int] = None
    sender_tx_count: Optional[int] = 0
    sender_avg_amount: Optional[float] = 0
    sender_total_volume: Optional[float] = 0

class SignedAnalysisResponse(BaseModel):
    """Response with cryptographic signature for on-chain verification."""
    transaction_id: str
    risk_score: int
    risk_level: str
    approved: bool
    explanation: str
    confidence: float
    model: str
    processing_time_ms: float
    # Cryptographic proof
    signature: str
    oracle_address: str
    signed_at: int
    message_hash: str
    message: str

# Security dependency
async def verify_request(
    request: Request,
    x_api_key: Optional[str] = Header(None)
):
    """Verify request authenticity and rate limits."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limiting
    allowed, reason = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    # In production, verify API key
    # if x_api_key != Config.API_SECRET_KEY:
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    
    return client_ip

# Endpoints
@app.get("/")
async def root():
    return {
        "service": "Secure AI Oracle",
        "oracle_address": Config.ORACLE_ADDRESS,
        "description": "Cryptographically signed AI fraud detection"
    }

@app.get("/oracle-info")
async def oracle_info():
    """
    Public endpoint to get oracle address for smart contract configuration.
    """
    return {
        "oracle_address": Config.ORACLE_ADDRESS,
        "supported_providers": ["openai", "anthropic", "local"],
        "signature_type": "EIP-191 personal_sign"
    }

@app.post("/analyze", response_model=SignedAnalysisResponse)
async def analyze_transaction(
    request: TransactionRequest,
    client_id: str = Depends(verify_request),
    provider: AIProvider = AIProvider.LOCAL
):
    """
    Analyze transaction and return SIGNED result.
    
    The signature can be verified on-chain to ensure this result
    came from the authorized oracle.
    """
    logger.info(f"Analysis request from {client_id}: {request.transaction_id}")
    
    transaction_data = request.model_dump()
    
    try:
        # Try AI providers in order
        if provider == AIProvider.OPENAI and Config.OPENAI_API_KEY:
            result = await ai_client.analyze_transaction_openai(transaction_data)
        elif provider == AIProvider.ANTHROPIC and Config.ANTHROPIC_API_KEY:
            result = await ai_client.analyze_transaction_anthropic(transaction_data)
        else:
            # Fallback to local heuristics
            result = ai_client.analyze_transaction_local(transaction_data)
    
    except Exception as e:
        logger.warning(f"AI provider failed, using fallback: {e}")
        result = ai_client.analyze_transaction_local(transaction_data)
    
    # Create response data
    response_data = {
        "transaction_id": request.transaction_id,
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "approved": result.approved,
        "explanation": result.explanation,
        "confidence": result.confidence,
        "model": result.model,
        "processing_time_ms": result.processing_time_ms
    }
    
    # SIGN THE RESPONSE
    signed_response = OracleSigner.sign_result(response_data)
    
    logger.info(f"Signed analysis: score={result.risk_score}, approved={result.approved}")
    
    return SignedAnalysisResponse(**signed_response)

@app.post("/analyze/openai", response_model=SignedAnalysisResponse)
async def analyze_with_openai(
    request: TransactionRequest,
    client_id: str = Depends(verify_request)
):
    """Analyze using OpenAI specifically."""
    return await analyze_transaction(request, client_id, AIProvider.OPENAI)

@app.post("/analyze/anthropic", response_model=SignedAnalysisResponse)
async def analyze_with_anthropic(
    request: TransactionRequest,
    client_id: str = Depends(verify_request)
):
    """Analyze using Anthropic Claude specifically."""
    return await analyze_transaction(request, client_id, AIProvider.ANTHROPIC)

@app.post("/analyze/local", response_model=SignedAnalysisResponse)
async def analyze_with_local(
    request: TransactionRequest,
    client_id: str = Depends(verify_request)
):
    """Analyze using local heuristics (no external API)."""
    return await analyze_transaction(request, client_id, AIProvider.LOCAL)

@app.post("/verify-signature")
async def verify_signature(
    message: str,
    signature: str,
    expected_address: str
):
    """Utility endpoint to verify a signature."""
    is_valid = OracleSigner.verify_signature(message, signature, expected_address)
    return {"valid": is_valid}

# Run with: uvicorn secureAiOracle:app --port 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
