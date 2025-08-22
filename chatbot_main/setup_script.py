#!/usr/bin/env python3
"""
Setup script for GRU Chatbot
Run: python setup.py
"""

import os
import subprocess
import sys
import platform

def run_command(command, description):
    """Run a shell command and handle errors with nice logs"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            command, shell=True, check=True,
            capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully!")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Ensure Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def main():
    print("🤖 GRU Chatbot Setup Script")
    print("This will install dependencies, train, and test the chatbot\n")
    
    # 1. Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # 2. Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n❌ Failed to install dependencies. Please check your pip installation.")
        sys.exit(1)
    
    # 3. Check for training data
    if not os.path.exists("sample_pairs.csv"):
        print("\n⚠️  sample_pairs.csv not found. The training script will create default data...")
    
    # 4. Train the model
    print("\n🎓 Training the GRU model (this may take a few minutes)...")
    if not run_command("python train_gru_chatbot.py --epochs 10", "Training GRU model"):
        print("\n❌ Training failed. Please check errors above.")
        sys.exit(1)
    
    # 5. Test the trained model
    print("\n🧪 Testing the trained model...")
    test_code = r'''
import sys
sys.path.append(".")
from inference_utils import load_artifacts, greedy_decode
model, tok, meta = load_artifacts("artifacts")
resp = greedy_decode(model, tok, meta, "hello")
print("Test response:", resp)
'''
    # Handle platform-specific command wrapping
    if platform.system() == "Windows":
        test_cmd = f'python -c "{test_code}"'
    else:
        test_cmd = f"python3 -c '{test_code}'"
    
    if not run_command(test_cmd, "Testing trained model"):
        print("\n❌ Model testing failed.")
        sys.exit(1)
    
    # 6. Success message
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\nYour GRU chatbot is ready! Here's how to use it:\n")
    print("1. Streamlit Web Interface:")
    print("   streamlit run chatbot_streamlit.py")
    print("   (Open http://localhost:8501 in your browser)")
    
    print("\n2. Flask REST API:")
    print("   python chatbot_flask.py")
    print("   (Open http://127.0.0.1:8000 in your browser)")
    
    print("\n3. Retrain with your own data:")
    print("   - Edit sample_pairs.csv with your conversation pairs")
    print("   - Run: python train_gru_chatbot.py --epochs 20")
    
    print("\n4. Example API Usage:")
    print('   curl -X POST http://127.0.0.1:8000/chat \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message": "hello"}\'')
    
    print("\nFiles created:")
    print("   📁 artifacts/          - Trained model + vocab")
    print("   📄 sample_pairs.csv    - Training data")
    print("   🤖 Scripts ready: chatbot_flask.py, chatbot_streamlit.py")

if __name__ == "__main__":
    main()
# Ensure the script is executable
if __name__ == "__main__":
    main()