#!/usr/bin/env python3
"""
Test Ollama API endpoints to ensure they work correctly
Run this to debug Ollama connectivity issues
"""

import requests
import json

OLLAMA_HOST = "http://127.0.0.1:11434"

print("\n" + "="*70)
print("OLLAMA API CONNECTIVITY TEST")
print("="*70 + "\n")

# Test 1: Check if Ollama is running
print("[TEST 1] Is Ollama running?")
try:
    response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f"✅ Ollama is RUNNING")
        print(f"   Available models: {len(models)}")
        for model in models:
            name = model.get('name', 'unknown')
            size = model.get('size', 0)
            size_gb = size / (1024**3)
            print(f"     • {name} ({size_gb:.2f} GB)")
    else:
        print(f"❌ Ollama returned status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("❌ OLLAMA NOT RUNNING - Cannot connect to http://127.0.0.1:11434")
    print("   Fix: Run 'ollama serve' in another terminal")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Test 2: Test embedding API
print("\n[TEST 2] Testing Embedding API (/api/embed)")
try:
    payload = {
        "model": "nomic-embed-text",
        "input": ["This is a test sentence", "Another test sentence"]
    }
    print(f"   Sending: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{OLLAMA_HOST}/api/embed", json=payload, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        if "embeddings" in data:
            embeddings = data["embeddings"]
            print(f"✅ Embedding API WORKS")
            print(f"   Embeddings returned: {len(embeddings)}")
            if embeddings:
                print(f"   Embedding dimensions: {len(embeddings[0])}")
        else:
            print(f"❌ Response missing 'embeddings' field")
            print(f"   Response: {data}")
    else:
        print(f"❌ API returned status {response.status_code}")
        print(f"   Response: {response.text}")
except requests.exceptions.Timeout:
    print(f"❌ Timeout - embedding model too slow or not responding")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Test generation API
print("\n[TEST 3] Testing Generation API (/api/generate)")
try:
    payload = {
        "model": "llama3",
        "prompt": "Say hello in one sentence",
        "stream": False,
        "num_predict": 50,
        "temperature": 0.1
    }
    print(f"   Sending: {json.dumps({k: v for k, v in payload.items() if k != 'prompt'}, indent=2)}")
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    
    if response.status_code == 200:
        data = response.json()
        if "response" in data:
            print(f"✅ Generation API WORKS")
            print(f"   Response: {data['response'][:100]}...")
        else:
            print(f"❌ Response missing 'response' field")
            print(f"   Response: {data}")
    else:
        print(f"❌ API returned status {response.status_code}")
        print(f"   Response: {response.text}")
except requests.exceptions.Timeout:
    print(f"❌ Timeout - LLM model too slow or not responding")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
