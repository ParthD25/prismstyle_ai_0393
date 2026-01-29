#!/usr/bin/env python3
"""
DeepFashion2 Training Environment Setup Script

This script sets up the complete training environment for the DeepFashion2
clothing classification model used in PrismStyle AI.

Requirements:
- Python 3.8-3.11
- NVIDIA GPU with CUDA support
- 16GB+ RAM
- 100GB+ free storage

Usage:
    python setup_training.py --check          # Check environment
    python setup_training.py --setup          # Full setup
    python setup_training.py --create-dirs    # Create directory structure only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

def print_header(message):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {message}")
    print("=" * 60)

def print_status(message, status="INFO"):
    """Print a status message"""
    symbols = {"OK": "[OK]", "FAIL": "[X]", "INFO": "[*]", "WARN": "[!]"}
    print(f"{symbols.get(status, '[*]')} {message}")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and 8 <= version.minor <= 11:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} detected", "OK")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} not supported. Use 3.8-3.11", "FAIL")
        return False

def check_nvidia_gpu():
    """Check for NVIDIA GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print_status(f"NVIDIA GPU detected: {gpu_info}", "OK")
            return True
    except FileNotFoundError:
        pass
    
    print_status("No NVIDIA GPU detected. Training will be slow on CPU.", "WARN")
    return False

def check_cuda():
    """Check CUDA installation"""
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path and os.path.exists(cuda_path):
        print_status(f"CUDA found at: {cuda_path}", "OK")
        return True
    
    # Try to find nvcc
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [l for l in result.stdout.split("\n") if "release" in l]
            if version_line:
                print_status(f"CUDA detected: {version_line[0].strip()}", "OK")
                return True
    except FileNotFoundError:
        pass
    
    print_status("CUDA not found. Install CUDA 11.8 or 12.1 for GPU support", "WARN")
    return False

def check_disk_space(path="."):
    """Check available disk space"""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    
    if free_gb >= 100:
        print_status(f"Available disk space: {free_gb:.1f} GB", "OK")
        return True
    else:
        print_status(f"Low disk space: {free_gb:.1f} GB (need 100GB+)", "WARN")
        return False

def check_memory():
    """Check available RAM"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        
        if total_gb >= 16:
            print_status(f"Total RAM: {total_gb:.1f} GB", "OK")
            return True
        else:
            print_status(f"RAM: {total_gb:.1f} GB (16GB+ recommended)", "WARN")
            return False
    except ImportError:
        print_status("Install psutil to check RAM: pip install psutil", "INFO")
        return True

def create_directories(base_path="."):
    """Create required directory structure"""
    dirs = [
        "data",
        "data/deepfashion2",
        "data/deepfashion2/train",
        "data/deepfashion2/validation",
        "models",
        "models/checkpoints",
        "converted_models",
        "logs",
        "notebooks"
    ]
    
    print_header("Creating Directory Structure")
    
    for dir_name in dirs:
        dir_path = Path(base_path) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print_status(f"Created: {dir_path}", "OK")
    
    # Create .gitkeep files
    gitkeep_dirs = ["data/deepfashion2/train", "data/deepfashion2/validation", 
                    "models/checkpoints", "converted_models", "logs", "notebooks"]
    
    for dir_name in gitkeep_dirs:
        gitkeep_path = Path(base_path) / dir_name / ".gitkeep"
        gitkeep_path.touch()
    
    print_status("Directory structure created successfully", "OK")

def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print_status("Dependencies installed successfully", "OK")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies: {e}", "FAIL")
        return False

def check_tensorflow():
    """Check TensorFlow installation and GPU support"""
    try:
        import tensorflow as tf
        
        version = tf.__version__
        print_status(f"TensorFlow {version} installed", "OK")
        
        # Check GPU support
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_status(f"TensorFlow GPU support: {len(gpus)} GPU(s) available", "OK")
            for gpu in gpus:
                print_status(f"  - {gpu.name}", "INFO")
            return True
        else:
            print_status("TensorFlow GPU support not available", "WARN")
            return False
            
    except ImportError:
        print_status("TensorFlow not installed. Run: pip install tensorflow", "FAIL")
        return False

def verify_dataset_structure():
    """Verify dataset directory structure"""
    print_header("Verifying Dataset Structure")
    
    import config
    
    required_dirs = [
        config.DATASET_PATH,
        f"{config.DATASET_PATH}/train",
        f"{config.DATASET_PATH}/validation"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print_status(f"Found: {dir_path}", "OK")
        else:
            print_status(f"Missing: {dir_path}", "WARN")
            all_exist = False
    
    if not all_exist:
        print_status("\nDataset not found. Please download DeepFashion2:", "INFO")
        print_status("1. Get from: https://github.com/switchablenorms/DeepFashion2", "INFO")
        print_status("2. Password: 2019Deepfashion2", "INFO")
        print_status(f"3. Extract to: {config.DATASET_PATH}", "INFO")
        return False
    
    # Check for training categories
    train_path = f"{config.DATASET_PATH}/train"
    if os.path.exists(train_path):
        categories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        if categories:
            print_status(f"Found {len(categories)} categories in training set", "OK")
            for cat in categories[:5]:
                print_status(f"  - {cat}", "INFO")
            if len(categories) > 5:
                print_status(f"  ... and {len(categories) - 5} more", "INFO")
        else:
            print_status("No categories found in training directory", "WARN")
            return False
    
    return True

def run_environment_check():
    """Run complete environment check"""
    print_header("PrismStyle AI - Training Environment Check")
    
    checks = [
        ("Python Version", check_python_version),
        ("NVIDIA GPU", check_nvidia_gpu),
        ("CUDA Installation", check_cuda),
        ("Disk Space", check_disk_space),
        ("RAM", check_memory),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Check TensorFlow if installed
    results["TensorFlow"] = check_tensorflow()
    
    # Summary
    print_header("Environment Check Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_check in results.items():
        status = "OK" if passed_check else "WARN"
        print_status(f"{name}: {'Passed' if passed_check else 'Warning/Failed'}", status)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if passed >= 4:
        print_status("\nEnvironment is ready for training!", "OK")
    else:
        print_status("\nSome checks failed. Review warnings above.", "WARN")
    
    return passed == total

def main():
    parser = argparse.ArgumentParser(
        description='DeepFashion2 Training Environment Setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_training.py --check          # Check environment
  python setup_training.py --setup          # Full setup
  python setup_training.py --create-dirs    # Create directories only

After setup, run training with:
  python train_model.py --epochs 50 --batch-size 32

Monitor with TensorBoard:
  tensorboard --logdir logs/
        """
    )
    
    parser.add_argument('--check', action='store_true', help='Check environment only')
    parser.add_argument('--setup', action='store_true', help='Full environment setup')
    parser.add_argument('--create-dirs', action='store_true', help='Create directories only')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies only')
    parser.add_argument('--verify-dataset', action='store_true', help='Verify dataset structure')
    
    args = parser.parse_args()
    
    if args.check:
        run_environment_check()
    elif args.create_dirs:
        create_directories()
    elif args.install_deps:
        install_dependencies()
    elif args.verify_dataset:
        verify_dataset_structure()
    elif args.setup:
        print_header("PrismStyle AI - Full Training Setup")
        
        # Step 1: Check environment
        run_environment_check()
        
        # Step 2: Create directories
        create_directories()
        
        # Step 3: Install dependencies
        install_dependencies()
        
        # Step 4: Verify TensorFlow
        check_tensorflow()
        
        # Step 5: Check dataset
        verify_dataset_structure()
        
        print_header("Setup Complete!")
        print_status("Next steps:", "INFO")
        print_status("1. Download DeepFashion2 dataset if not present", "INFO")
        print_status("2. Run: python train_model.py --epochs 50", "INFO")
        print_status("3. Monitor: tensorboard --logdir logs/", "INFO")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
