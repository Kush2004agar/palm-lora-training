"""
Validate that everything is set up correctly
"""

import sys
import subprocess

def check_imports():
    """Check if all required packages are installed"""
    print("🔍 Checking imports...")
    
    required = [
        'torch',
        'torchvision',
        'diffusers',
        'transformers',
        'accelerate',
        'peft',
        'PIL',
        'numpy',
        'tqdm'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports OK!")
    return True

def check_cuda():
    """Check CUDA availability"""
    import torch
    
    print("\n🔍 Checking CUDA...")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("  ⚠️  CUDA not available - training will be slow on CPU")
        return True

def check_directories():
    """Check if required directories exist"""
    import os
    
    print("\n🔍 Checking directories...")
    
    dirs = ['./palm_dataset', './output', './logs']
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path} exists")
        else:
            print(f"  ℹ️  {dir_path} - will be created during training")
    
    return True

def main():
    print("=" * 50)
    print("Palm LoRA Setup Validation")
    print("=" * 50)
    
    checks = [
        check_imports(),
        check_cuda(),
        check_directories()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("✅ Setup validation passed!")
        print("\nYou're ready to train! Run:")
        print("  python prepare_dataset.py --source <your_images>")
        print("  python train_lora.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()