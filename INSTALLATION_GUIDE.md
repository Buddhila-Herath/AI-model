# Complete Installation Guide - Step by Step

## 🎯 RECOMMENDED VERSIONS (February 2025)

### ✅ Python Version
**Install: Python 3.12.x** (Latest Stable)
- **Why**: Best balance of stability and features
- **Download**: https://www.python.org/downloads/
- **Alternative**: Python 3.11.x (also very stable, widely tested)
- **Minimum**: Python 3.10 (required by PyTorch)
- **Maximum**: Python 3.13 (supported but newer)

### ✅ CUDA Version
**Install: CUDA Toolkit 12.6** (Recommended)
- **Why**: Latest stable CUDA version with full PyTorch support
- **Download**: https://developer.nvidia.com/cuda-downloads
- **Alternative**: CUDA 12.8 (newest, but 12.6 is more stable)
- **Note**: CUDA 12.1 will work but is older; 12.6 is better supported

### ✅ PyTorch Version
**Will install: PyTorch 2.7.0** (Latest Stable)
- Automatically installed with the pip command below
- Supports CUDA 12.6 perfectly

### ✅ Visual Studio Code
**Install: Latest Version**
- **Download**: https://code.visualstudio.com/
- **Extension**: Install "Python" extension by Microsoft

---

## 📋 STEP-BY-STEP INSTALLATION

### Step 1: Install Python 3.12

1. Go to: **https://www.python.org/downloads/**
2. Click the big yellow "Download Python 3.12.x" button
3. Run the installer
4. ⚠️ **CRITICAL**: Check the box **"Add Python to PATH"** at the bottom
5. Click "Install Now"
6. Wait for installation to complete

**Verify Installation:**
- Open Command Prompt (cmd)
- Type: `python --version`
- Should show: `Python 3.12.x`

---

### Step 2: Install CUDA Toolkit 12.6

1. Go to: **https://developer.nvidia.com/cuda-downloads**
2. Select:
   - **Operating System**: Windows
   - **Architecture**: x86_64
   - **Version**: Windows 10/11
   - **Installer Type**: exe (local)
3. Download the installer (it's large, ~3GB)
4. Run the installer
5. Choose "Express Installation" (recommended)
6. Wait for installation (takes 10-15 minutes)

**Verify Installation:**
- Open Command Prompt
- Type: `nvcc --version`
- Should show CUDA version 12.6

**Check GPU Driver:**
- Type: `nvidia-smi`
- Should show your RTX 3050 and driver version
- **Note**: Your NVIDIA driver must support CUDA 12.6 (usually driver 550+)

---

### Step 3: Install Visual Studio Code

1. Go to: **https://code.visualstudio.com/**
2. Click "Download for Windows"
3. Run the installer
4. Use default settings (or customize if you prefer)
5. After installation, open VS Code

**Install Python Extension:**
1. In VS Code, click the Extensions icon (left sidebar, 4 squares)
2. Search for: "Python"
3. Install the one by **Microsoft** (official)
4. Restart VS Code if prompted

---

### Step 4: Set Up Your Project

1. Open Command Prompt
2. Navigate to your project:
   ```cmd
   cd Documents\VivaAIProject
   ```

3. Activate the virtual environment:
   ```cmd
   venv\Scripts\activate
   ```
   You should see `(venv)` at the start of your command line.

4. Upgrade pip:
   ```cmd
   python -m pip install --upgrade pip
   ```

---

### Step 5: Install PyTorch with CUDA 12.6

**If you installed CUDA 12.6** (recommended):
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**If you installed CUDA 12.1** (alternative):
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**If you installed CUDA 11.8** (older but stable):
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Wait for installation** (takes 5-10 minutes, downloads ~2GB)

---

### Step 6: Install Other Dependencies

Still in the activated virtual environment `(venv)`, run:

```cmd
pip install -r requirements.txt
```

This installs:
- opencv-python (computer vision)
- mediapipe (hand/pose detection)
- openai-whisper (speech recognition)
- transformers (AI models)
- flask (web framework)

---

### Step 7: Verify Everything Works

Run the GPU check script:

```cmd
python check_gpu.py
```

**Expected Output:**
```
--- GPU CHECK ---
Success! Your RTX 3050 is detected.
Device name: NVIDIA GeForce RTX 3050
```

If you see this, **everything is working perfectly!** 🎉

---

## 🔧 TROUBLESHOOTING

### Problem: "Python not found"
- **Solution**: Python wasn't added to PATH. Reinstall Python and check "Add Python to PATH"

### Problem: "CUDA not detected"
- **Solution 1**: Check NVIDIA driver: `nvidia-smi` - update if needed
- **Solution 2**: Verify CUDA installation: `nvcc --version`
- **Solution 3**: Install PyTorch CPU version first, then CUDA version

### Problem: "pip install fails"
- **Solution**: Make sure virtual environment is activated (see `(venv)` in prompt)
- **Solution**: Try: `python -m pip install --upgrade pip` first

### Problem: "RTX 3050 not detected"
- **Solution**: Update NVIDIA drivers from https://www.nvidia.com/drivers
- **Solution**: Make sure CUDA toolkit matches your driver version

---

## 📝 QUICK REFERENCE

| Component | Version | Download Link |
|-----------|---------|--------------|
| Python | 3.12.x | https://www.python.org/downloads/ |
| CUDA | 12.6 | https://developer.nvidia.com/cuda-downloads |
| VS Code | Latest | https://code.visualstudio.com/ |
| PyTorch | 2.7.0 | (auto-installed via pip) |

---

## ✅ CHECKLIST

- [ ] Python 3.12 installed with PATH checked
- [ ] CUDA 12.6 installed
- [ ] VS Code installed with Python extension
- [ ] Virtual environment activated
- [ ] PyTorch with CUDA installed
- [ ] Other dependencies installed
- [ ] GPU check script runs successfully

---

**You're all set!** Start coding your AI project! 🚀
