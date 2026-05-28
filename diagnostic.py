#!/usr/bin/env python3
"""
Quick diagnostic script to verify all components are working
Run this in the backend directory with activated venv
"""

import sys
import requests
from pathlib import Path

print("\n" + "="*60)
print("AI Research Assistant - Diagnostic Check")
print("="*60 + "\n")

# Check 1: Backend is running
print("[1/5] Checking if backend is running...")
try:
    response = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=3)
    print(f"✅ Backend is running: {response.status_code}")
    health_data = response.json()
    print(f"   Status: {health_data.get('status')}")
    if health_data.get('details'):
        print(f"   Ollama running: {health_data['details'].get('ollama_running')}")
except Exception as e:
    print(f"❌ Backend is NOT running: {e}")
    sys.exit(1)

# Check 2: Ollama is accessible
print("\n[2/5] Checking Ollama connection...")
try:
    response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
    print(f"✅ Ollama is running")
    models = response.json().get('models', [])
    print(f"   Available models: {len(models)}")
    for model in models:
        print(f"     - {model.get('name')}")
except Exception as e:
    print(f"❌ Ollama is NOT running: {e}")
    print("   Make sure to run: ollama serve")

# Check 3: Test embedding endpoint
print("\n[3/5] Testing embedding API...")
try:
    response = requests.post(
        "http://127.0.0.1:11434/api/embed",
        json={"model": "nomic-embed-text", "input": ["test"]},
        timeout=30
    )
    if response.status_code == 200:
        print("✅ Embedding API working")
    else:
        print(f"❌ Embedding API returned {response.status_code}")
except Exception as e:
    print(f"❌ Embedding API error: {e}")

# Check 4: Test LLM endpoint
print("\n[4/5] Testing LLM generation API...")
try:
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "llama3", "prompt": "say hello", "stream": False},
        timeout=30
    )
    if response.status_code == 200:
        print("✅ LLM generation API working")
    else:
        print(f"❌ LLM API returned {response.status_code}")
except Exception as e:
    print(f"❌ LLM API error: {e}")

# Check 5: Test papers endpoint
print("\n[5/5] Testing papers endpoint...")
try:
    response = requests.get("http://127.0.0.1:8000/api/v1/papers", timeout=3)
    if response.status_code == 200:
        papers = response.json()
        print(f"✅ Papers endpoint working")
        print(f"   Papers in library: {len(papers) if isinstance(papers, list) else 'unknown'}")
    else:
        print(f"❌ Papers endpoint returned {response.status_code}")
except Exception as e:
    print(f"❌ Papers endpoint error: {e}")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60 + "\n")
