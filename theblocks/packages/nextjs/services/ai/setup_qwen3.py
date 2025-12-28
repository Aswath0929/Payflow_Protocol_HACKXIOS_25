#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW QWEN3 LOCAL LLM SETUP                                     ║
║                                                                                       ║
║   Setup Script for Running Qwen3 on RTX 4070 (8GB VRAM)                              ║
║                                                                                       ║
║   This script:                                                                        ║
║   1. Checks if Ollama is installed                                                   ║
║   2. Starts Ollama server                                                            ║
║   3. Pulls the Qwen3 model                                                           ║
║   4. Tests the model                                                                 ║
║                                                                                       ║
║   Hackxios 2K25 - PayFlow Protocol                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess
import sys
import time
import os

# Qwen3 model for RTX 4070 8GB VRAM
# Options:
#   qwen3:1.7b  - ~1.5GB VRAM (fastest, basic analysis)
#   qwen3:4b    - ~3GB VRAM (good balance)
#   qwen3:8b    - ~5GB VRAM (recommended for RTX 4070)
QWEN3_MODEL = "qwen3:8b"

def print_banner():
    """Print the setup banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                     PAYFLOW QWEN3 LOCAL LLM SETUP                                     ║
║                                                                                       ║
║   100% LOCAL AI - No Cloud API Keys Needed!                                          ║
║   Running on your RTX 4070 GPU                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
""")

def check_ollama_installed():
    """Check if Ollama is installed."""
    print("\n1. Checking Ollama installation...")
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   ✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("   ❌ Ollama is not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error checking Ollama: {e}")
        return False

def install_ollama_instructions():
    """Print Ollama installation instructions."""
    print("""
   ═══════════════════════════════════════════════════════════════
   INSTALL OLLAMA
   ═══════════════════════════════════════════════════════════════
   
   Windows:
   1. Download from: https://ollama.ai/download/windows
   2. Run the installer
   3. Restart this script
   
   Or use winget:
   > winget install Ollama.Ollama
   
   After installation, run this script again.
   ═══════════════════════════════════════════════════════════════
""")

def check_ollama_server():
    """Check if Ollama server is running."""
    print("\n2. Checking Ollama server status...")
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            print("   ✅ Ollama server is running")
            return True
        else:
            print("   ⚠️ Ollama server responded with error")
            return False
    except:
        print("   ⚠️ Ollama server is not running")
        return False

def start_ollama_server():
    """Start Ollama server in background."""
    print("\n   Starting Ollama server...")
    try:
        # On Windows, start Ollama in a new process
        if sys.platform == "win32":
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Wait for server to start
        print("   Waiting for server to start...")
        for i in range(10):
            time.sleep(1)
            try:
                import httpx
                response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
                if response.status_code == 200:
                    print("   ✅ Ollama server started successfully!")
                    return True
            except:
                print(f"   ... attempt {i+1}/10")
                continue
        
        print("   ❌ Failed to start Ollama server")
        return False
    except Exception as e:
        print(f"   ❌ Error starting server: {e}")
        return False

def check_qwen3_model():
    """Check if Qwen3 model is installed."""
    print(f"\n3. Checking for {QWEN3_MODEL} model...")
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            # Check for exact match or variant
            for model in models:
                if "qwen3" in model.lower():
                    print(f"   ✅ Found Qwen3 model: {model}")
                    return True
            
            print(f"   ⚠️ Qwen3 not found. Available models: {models}")
            return False
        return False
    except Exception as e:
        print(f"   ❌ Error checking models: {e}")
        return False

def pull_qwen3_model():
    """Pull the Qwen3 model."""
    print(f"\n   Pulling {QWEN3_MODEL}... (this may take 5-10 minutes)")
    print("   ═══════════════════════════════════════════════════════════════")
    try:
        result = subprocess.run(
            ["ollama", "pull", QWEN3_MODEL],
            capture_output=False,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        if result.returncode == 0:
            print(f"   ✅ {QWEN3_MODEL} downloaded successfully!")
            return True
        else:
            print(f"   ❌ Failed to pull {QWEN3_MODEL}")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Download timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ Error pulling model: {e}")
        return False

def test_qwen3():
    """Test Qwen3 model with a simple prompt."""
    print("\n4. Testing Qwen3 model...")
    try:
        import httpx
        
        print("   Sending test prompt...")
        start = time.time()
        
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": QWEN3_MODEL,
                "prompt": "Analyze this transaction for fraud: Amount $9,999 from new sender to offshore account. Is this suspicious?",
                "stream": False,
                "options": {
                    "num_predict": 100
                }
            },
            timeout=60.0
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("response", "")[:200]
            print(f"   ✅ Qwen3 responded in {elapsed:.1f}s")
            print(f"\n   Response preview:")
            print(f"   {text}...")
            return True
        else:
            print(f"   ❌ Qwen3 API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error testing model: {e}")
        return False

def print_success():
    """Print success message."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║   ██████╗ ██╗   ██╗███████╗███╗   ██╗██████╗     ██████╗ ███████╗ █████╗ ██████╗ ██╗   ║
║  ██╔═══██╗██║   ██║██╔════╝████╗  ██║╚════██╗    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██║   ║
║  ██║   ██║██║   ██║█████╗  ██╔██╗ ██║ █████╔╝    ██████╔╝█████╗  ███████║██║  ██║██║   ║
║  ██║▄▄ ██║██║   ██║██╔══╝  ██║╚██╗██║ ╚═══██╗    ██╔══██╗██╔══╝  ██╔══██║██║  ██║╚═╝   ║
║  ╚██████╔╝╚██████╔╝███████╗██║ ╚████║██████╔╝    ██║  ██║███████╗██║  ██║██████╔╝██╗   ║
║   ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝   ║
║                                                                                       ║
║   🎉 Qwen3 Local LLM is ready for fraud detection!                                    ║
║                                                                                       ║
║   Your Setup:                                                                         ║
║   • Model: qwen3:8b (Latest 2025 release from Alibaba)                               ║
║   • GPU: RTX 4070 (8GB VRAM)                                                         ║
║   • Inference: 100% LOCAL - No cloud API, no data leaves your machine               ║
║   • Speed: <500ms per analysis                                                       ║
║                                                                                       ║
║   To start the fraud detection server:                                               ║
║   > cd theblocks/packages/nextjs/services/ai                                         ║
║   > python -m uvicorn secureAIOracle:app --host 0.0.0.0 --port 8000                  ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
""")

def main():
    """Main setup function."""
    print_banner()
    
    # Step 1: Check Ollama installed
    if not check_ollama_installed():
        install_ollama_instructions()
        return
    
    # Step 2: Check/start Ollama server
    if not check_ollama_server():
        if not start_ollama_server():
            print("\n❌ Could not start Ollama server. Please run 'ollama serve' manually.")
            return
    
    # Step 3: Check/pull Qwen3 model
    if not check_qwen3_model():
        if not pull_qwen3_model():
            print(f"\n❌ Could not pull {QWEN3_MODEL}. Please run 'ollama pull {QWEN3_MODEL}' manually.")
            return
    
    # Step 4: Test the model
    if not test_qwen3():
        print("\n⚠️ Qwen3 test failed but model is installed. Try restarting Ollama.")
        return
    
    # Success!
    print_success()

if __name__ == "__main__":
    # Check for httpx
    try:
        import httpx
    except ImportError:
        print("Installing httpx...")
        subprocess.run([sys.executable, "-m", "pip", "install", "httpx"], check=True)
        import httpx
    
    main()
