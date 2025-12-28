"""
Comprehensive Expert System Test
"""
import time
import numpy as np

print('='*70)
print('COMPREHENSIVE EXPERT SYSTEM TEST')
print('='*70)

# Initialize expert oracle
print('\n1. Initializing Expert AI Oracle...')
start = time.time()
from expertAIOracle import ExpertAIOracle, ExpertConfig
oracle = ExpertAIOracle(config=ExpertConfig())
init_time = time.time() - start
print(f'   Initialization time: {init_time:.2f}s')

# Test cases with expected outcomes
test_cases = [
    {'name': 'Normal $500 transfer', 'sender': '0xAlice123456789012345678901234567890123456', 'recipient': '0xBob123456789012345678901234567890123456', 'amount': 500.0, 'expected_level': 'safe_or_low'},
    {'name': 'Structuring $9,999', 'sender': '0xSmurf123456789012345678901234567890123456', 'recipient': '0xTarget23456789012345678901234567890123456', 'amount': 9999.0, 'expected_level': 'high'},
    {'name': 'Tornado Cash mixer', 'sender': '0xUser1234567890123456789012345678901234567', 'recipient': '0x722122df12d4e14e13ac3b6895a86e84145b6967', 'amount': 10000.0, 'expected_level': 'critical'},
    {'name': 'Small $25 payment', 'sender': '0xRegular23456789012345678901234567890123456', 'recipient': '0xMerchant4567890123456789012345678901234567', 'amount': 25.0, 'expected_level': 'safe_or_low'},
    {'name': 'Large $50k business', 'sender': '0xBusiness234567890123456789012345678901234', 'recipient': '0x28c6c06298d514db089934071355e5743bf21d60', 'amount': 50000.0, 'expected_level': 'medium_or_high'},
]

print('\n2. Running test cases...\n')
results = []
latencies = []

for i, tc in enumerate(test_cases):
    verdict = oracle.analyze(
        sender=tc['sender'],
        recipient=tc['recipient'],
        amount=tc['amount'],
    )
    
    latencies.append(verdict.total_time_ms)
    
    # Check if result matches expectation
    level = verdict.risk_level
    expected = tc['expected_level']
    
    if expected == 'safe_or_low':
        passed = level in ['safe', 'low']
    elif expected == 'medium_or_high':
        passed = level in ['medium', 'high']
    else:
        passed = level == expected
    
    status = '✅' if passed else '⚠️'
    emoji = verdict.risk_emoji
    
    print(f'{status} Test {i+1}: {tc["name"]}')
    print(f'   Score: {verdict.risk_score}/100 | Level: {level.upper()} {emoji}')
    print(f'   Latency: {verdict.total_time_ms:.1f}ms | Confidence: {verdict.confidence:.1%}')
    if verdict.primary_typology:
        print(f'   Typology: {verdict.primary_typology}')
    print()
    
    results.append({'passed': passed, 'latency': verdict.total_time_ms, 'score': verdict.risk_score})

# Summary
print('='*70)
print('PERFORMANCE SUMMARY')
print('='*70)
print(f'Tests Passed: {sum(1 for r in results if r["passed"])}/{len(results)}')
print(f'Avg Latency: {np.mean(latencies):.1f}ms')
print(f'P95 Latency: {np.percentile(latencies, 95):.1f}ms')
print(f'Max Latency: {max(latencies):.1f}ms')
print(f'All under 300ms: {"✅ YES" if max(latencies) < 300 else "❌ NO"}')
print('='*70)
