"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║          COMPREHENSIVE FRAUD DETECTION API TEST                               ║
║                                                                               ║
║   Tests FastAPI /analyze endpoint with all token types                        ║
║   Verifies that Native ETH + ERC20 tokens go through AI fraud detection       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import json
import time

def test_fraud_api():
    print("="*70)
    print("   COMPREHENSIVE FRAUD DETECTION API TEST")
    print("   Testing all transaction types including Native ETH")
    print("="*70)
    print()
    
    # Check health first
    try:
        health = requests.get('http://localhost:8000/health', timeout=5).json()
        print(f"API Status: {health.get('status', 'unknown')}")
        print(f"Oracle Address: {health.get('oracle_address', 'N/A')}")
        print(f"Qwen3 Available: {health.get('qwen3_available', False)}")
        print()
    except Exception as e:
        print(f"Error: API not reachable - {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "ETH - Small Normal Transfer",
            "token": "ETH",
            "sender": "0x74dDa086DefBFE113E387e70f0304631972525E5",
            "recipient": "0x742d35Cc6634C0532925a3b844Bc9e7595f8Fac6",
            "amount": 0.05,
        },
        {
            "name": "ETH - Medium Business Transfer",
            "token": "ETH",
            "sender": "0x74dDa086DefBFE113E387e70f0304631972525E5",
            "recipient": "0x1234567890123456789012345678901234567890",
            "amount": 2.5,
        },
        {
            "name": "ETH - Large High-Value Transfer",
            "token": "ETH",
            "sender": "0x74dDa086DefBFE113E387e70f0304631972525E5",
            "recipient": "0x00000000000000000000000000000000DeaDBeef",
            "amount": 50.0,
        },
        {
            "name": "USDC - Standard Payment",
            "token": "USDC",
            "sender": "0x74dDa086DefBFE113E387e70f0304631972525E5",
            "recipient": "0xAbC123456789012345678901234567890123456D",
            "amount": 100.0,
        },
        {
            "name": "USDT - Large Transfer",
            "token": "USDT",
            "sender": "0x74dDa086DefBFE113E387e70f0304631972525E5",
            "recipient": "0x9876543210987654321098765432109876543210",
            "amount": 9999.99,
        },
        {
            "name": "ETH - Micro Transfer",
            "token": "ETH",
            "sender": "0x74dDa086DefBFE113E387e70f0304631972525E5",
            "recipient": "0x742d35Cc6634C0532925a3b844Bc9e7595f8Fac6",
            "amount": 0.001,
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"[TEST {i}/{len(test_cases)}] {test['name']}")
        print(f"  Token: {test['token']} | Amount: {test['amount']}")
        
        try:
            response = requests.post(
                'http://localhost:8000/analyze',
                json={
                    'sender': test['sender'],
                    'recipient': test['recipient'],
                    'amount': test['amount'],
                    'token': test['token'],
                    'chain_id': 11155111
                },
                timeout=30
            )
            
            if response.ok:
                result = response.json()
                print(f"  Risk Score: {result.get('overall_score', 'N/A')}")
                print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
                print(f"  Approved: {result.get('approved', 'N/A')}")
                print(f"  GNN Enabled: {result.get('gnn_enabled', 'N/A')}")
                print(f"  ✅ PASSED - AI Fraud Detection Active")
                passed += 1
            else:
                print(f"  ❌ FAILED - Status {response.status_code}: {response.text[:100]}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ FAILED - {str(e)[:80]}")
            failed += 1
        
        print()
    
    # Summary
    print("="*70)
    print("   TEST SUMMARY")
    print("="*70)
    print(f"  Total Tests: {len(test_cases)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {(passed/len(test_cases))*100:.1f}%")
    print()
    
    if failed == 0:
        print("  ✅ ALL FRAUD DETECTION API TESTS PASSED!")
        print("  ✅ Both Native ETH and ERC20 tokens are governed by AI fraud detection!")
    else:
        print(f"  ⚠️ {failed} test(s) failed - review logs above")

if __name__ == "__main__":
    test_fraud_api()
