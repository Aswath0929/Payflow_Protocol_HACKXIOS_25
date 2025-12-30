"""
Test the FastAPI /analyze endpoint with ETH transactions
"""
import requests
import json

# Test ETH transaction
response = requests.post(
    'http://localhost:8000/analyze',
    json={
        'sender': '0x1234567890abcdef1234567890abcdef12345678',
        'recipient': '0xabcdef1234567890abcdef1234567890abcdef12',
        'amount': 0.5,
        'token': 'ETH',
        'chain_id': 11155111
    }
)

print('='*50)
print('ETH FRAUD DETECTION API TEST')
print('='*50)
print('Status Code:', response.status_code)
if response.ok:
    result = response.json()
    print(f"Risk Score: {result.get('overall_score', 'N/A')}")
    print(f"Risk Level: {result.get('risk_level', 'N/A')}")
    print(f"Approved: {result.get('approved', 'N/A')}")
    print(f"GNN Enabled: {result.get('gnn_enabled', 'N/A')}")
    print()
    print('SUCCESS: ETH transactions work with fraud detection API!')
else:
    print('Error:', response.text[:500])
