"""
Comprehensive Setup Verification Script
Tests all components: Python, CUDA, PyTorch, GPU, and dependencies
"""

import sys
import subprocess

print("=" * 60)
print("COMPREHENSIVE SETUP VERIFICATION")
print("=" * 60)
print()

# 1. Python Version Check
print("1. PYTHON VERSION CHECK")
print("-" * 60)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print()

# 2. CUDA Installation Check
print("2. CUDA INSTALLATION CHECK")
print("-" * 60)
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"[OK] CUDA Toolkit: {line.strip()}")
                break
    else:
        print("[X] CUDA Toolkit not found via nvcc")
except FileNotFoundError:
    print("[X] CUDA Toolkit (nvcc) not found in PATH")
except Exception as e:
    print(f"[X] Error checking CUDA: {e}")
print()

# 3. NVIDIA Driver Check
print("3. NVIDIA DRIVER CHECK")
print("-" * 60)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"[OK] {line.strip()}")
            if 'CUDA Version' in line:
                print(f"[OK] {line.strip()}")
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                gpu_info = line.split('|')
                if len(gpu_info) > 1:
                    print(f"[OK] GPU: {gpu_info[1].strip()}")
    else:
        print("[X] nvidia-smi not available")
except FileNotFoundError:
    print("[X] nvidia-smi not found")
except Exception as e:
    print(f"[X] Error checking NVIDIA driver: {e}")
print()

# 4. PyTorch Installation Check
print("4. PYTORCH INSTALLATION CHECK")
print("-" * 60)
try:
    import torch
    print(f"[OK] PyTorch Version: {torch.__version__}")
    print(f"[OK] PyTorch Location: {torch.__file__}")
except ImportError:
    print("[X] PyTorch not installed")
    sys.exit(1)
print()

# 5. GPU Detection and Details
print("5. GPU DETECTION & DETAILS")
print("-" * 60)
try:
    if torch.cuda.is_available():
        print(f"[OK] CUDA Available: True")
        print(f"[OK] CUDA Version: {torch.version.cuda}")
        print(f"[OK] cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"[OK] Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            print(f"    Capability: {torch.cuda.get_device_capability(i)}")
            print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"    Current Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"    Current Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("[X] CUDA Available: False")
        print("  GPU not detected by PyTorch")
except Exception as e:
    print(f"[X] Error checking GPU: {e}")
print()

# 6. GPU Computation Test
print("6. GPU COMPUTATION TEST")
print("-" * 60)
try:
    if torch.cuda.is_available():
        print("Running GPU computation test...")
        
        # Create test tensors
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # Perform computation
        import time
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed_time = time.time() - start_time
        
        print(f"[OK] GPU computation successful!")
        print(f"  Matrix multiplication (1000x1000): {elapsed_time*1000:.2f} ms")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
    else:
        print("[X] Skipping GPU test - CUDA not available")
except Exception as e:
    print(f"[X] GPU computation test failed: {e}")
print()

# 7. Installed Dependencies Check
print("7. INSTALLED DEPENDENCIES CHECK")
print("-" * 60)
packages_to_check = [
    'torch', 'torchvision', 'torchaudio',
    'cv2', 'mediapipe', 'whisper', 'transformers', 'flask',
    'numpy', 'PIL'
]

for package in packages_to_check:
    try:
        if package == 'cv2':
            import cv2
            print(f"[OK] opencv-python: {cv2.__version__}")
        elif package == 'PIL':
            from PIL import Image
            import PIL
            print(f"[OK] Pillow: {PIL.__version__}")
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'installed')
            print(f"[OK] {package}: {version}")
    except ImportError:
        print(f"[X] {package}: Not installed")
    except Exception as e:
        print(f"[X] {package}: Error - {e}")
print()

# 8. Summary
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
if torch.cuda.is_available():
    print("[OK] All systems operational!")
    print("[OK] GPU acceleration is ready to use")
    print(f"[OK] Your {torch.cuda.get_device_name(0)} is ready for AI workloads")
else:
    print("[WARNING] GPU not detected - check CUDA installation")
print("=" * 60)
