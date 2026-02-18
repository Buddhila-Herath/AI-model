"""
Quick GPU Test Script
Run this anytime to verify GPU is working and test computation speed
"""

import torch
import time

print("=" * 60)
print("GPU QUICK TEST")
print("=" * 60)
print()

# Check if CUDA is available
if not torch.cuda.is_available():
    print("[ERROR] CUDA not available!")
    print("Check your CUDA and PyTorch installation.")
    exit(1)

# GPU Information
print("GPU Information:")
print("-" * 60)
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Free Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
print()

# Test 1: Simple computation
print("Test 1: Simple GPU Computation")
print("-" * 60)
device = torch.device("cuda:0")
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
elapsed = (time.time() - start) * 1000

print(f"[OK] Matrix multiplication (1000x1000) completed in {elapsed:.2f} ms")
print(f"Result shape: {z.shape}")
print(f"Result device: {z.device}")
print()

# Test 2: Neural network forward pass
print("Test 2: Neural Network Forward Pass")
print("-" * 60)
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
).to(device)

input_data = torch.randn(32, 1000).to(device)

start = time.time()
output = model(input_data)
torch.cuda.synchronize()
elapsed = (time.time() - start) * 1000

print(f"[OK] Neural network forward pass completed in {elapsed:.2f} ms")
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
print()

# Test 3: Memory usage
print("Test 3: Memory Usage")
print("-" * 60)
print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
print()

# Cleanup
del x, y, z, model, input_data, output
torch.cuda.empty_cache()

print("=" * 60)
print("[SUCCESS] All GPU tests passed!")
print(f"Your {torch.cuda.get_device_name(0)} is working perfectly!")
print("=" * 60)
