import torch

print("--- GPU CHECK ---")
if torch.cuda.is_available():
    print(f"Success! Your RTX 3050 is detected.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("Error: GPU not found. Check your CUDA installation.")
