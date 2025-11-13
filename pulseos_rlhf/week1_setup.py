"""
Week 1 Day 1-2: Environment Setup and Verification

This script verifies that all required dependencies are installed and
the environment is ready for TRL-based RLHF training.
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    return True

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False

def main():
    """Run environment checks."""
    print("=" * 80)
    print("PulseOS RLHF - Environment Setup Check")
    print("=" * 80)
    print()
    
    # Check Python version
    print("1. Checking Python version...")
    python_ok = check_python_version()
    print()
    
    # Check required packages
    print("2. Checking required packages...")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    
    print()
    
    # Check GPU
    print("3. Checking GPU availability...")
    gpu_available = check_gpu()
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if python_ok and all_installed:
        print("✓ All required packages are installed")
    else:
        print("✗ Some packages are missing")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
    
    if gpu_available:
        print("✓ GPU available - training will be faster")
    else:
        print("⚠️  No GPU detected - training will be slower")
        print("  Consider using Google Colab free tier (T4 GPU) if local fails")
    
    print()
    print("Next steps:")
    print("  1. If all packages installed: Run week1_sentiment.py")
    print("  2. If packages missing: pip install -r requirements.txt")
    print("  3. If local fails: Use Colab notebook (01_sentiment_demo.ipynb)")
    print("=" * 80)
    
    return python_ok and all_installed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


