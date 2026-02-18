# How to Check and Verify Your Setup

## Quick Commands to Verify Everything

### 1. Activate Virtual Environment
Always activate your virtual environment first:
```powershell
cd Documents\VivaAIProject
.\venv\Scripts\Activate.ps1
```

### 2. Check Python Version
```powershell
python --version
```
**Expected:** Python 3.12.8

### 3. Check CUDA Installation
```powershell
nvcc --version
```
**Expected:** CUDA 12.6

### 4. Check NVIDIA Driver
```powershell
nvidia-smi
```
**Expected:** Shows your RTX 3050 GPU and driver version

### 5. Quick GPU Test (Recommended)
```powershell
python test_gpu.py
```
**Expected:** Shows GPU info and runs computation tests

### 6. Comprehensive Verification
```powershell
python verify_setup.py
```
**Expected:** Full system check including all components

### 7. Simple GPU Check
```powershell
python check_gpu.py
```
**Expected:** Basic GPU detection message

## What Each Script Does

### `test_gpu.py` - Quick GPU Test
- Shows GPU information
- Tests matrix multiplication
- Tests neural network forward pass
- Shows memory usage
- **Use this for quick verification**

### `verify_setup.py` - Full System Check
- Checks Python version
- Verifies CUDA installation
- Checks NVIDIA driver
- Verifies PyTorch installation
- Shows detailed GPU information
- Tests GPU computation
- Lists all installed packages
- **Use this for complete verification**

### `check_gpu.py` - Basic Check
- Simple GPU detection
- **Use this for minimal check**

## Expected Output Examples

### ✅ Success Output:
```
[OK] CUDA Available: True
[OK] GPU computation successful!
Device name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

### ❌ Error Output:
```
[ERROR] CUDA not available!
```

## Troubleshooting

### If GPU is not detected:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA: `nvcc --version`
3. Reinstall PyTorch with CUDA:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

### If scripts don't run:
1. Make sure virtual environment is activated (you should see `(venv)` in prompt)
2. Check Python version: `python --version`
3. Reinstall dependencies: `pip install -r requirements.txt`

## Quick Reference

| Check | Command |
|-------|---------|
| Python | `python --version` |
| CUDA | `nvcc --version` |
| GPU Driver | `nvidia-smi` |
| Quick GPU Test | `python test_gpu.py` |
| Full Verification | `python verify_setup.py` |
| Basic GPU Check | `python check_gpu.py` |

## Your Current Status

✅ Python 3.12.8 - Installed  
✅ CUDA 12.6 - Installed  
✅ PyTorch 2.10.0+cu126 - Installed  
✅ GPU RTX 3050 - Detected and Working  
✅ GPU Computation - Tested and Working  

**Everything is ready for AI development!** 🚀
