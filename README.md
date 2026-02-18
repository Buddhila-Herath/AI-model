# VivaAIProject

AI Research Project with RTX 3050 GPU Support

## 📖 Quick Start

**See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for complete step-by-step instructions!**

## Prerequisites

Before starting, ensure you have installed:

1. **Python 3.12** (recommended) or Python 3.11/3.10
   - Download from: https://www.python.org/downloads/
   - ⚠️ **CRITICAL**: Check "Add Python to PATH" during installation

2. **Visual Studio Code** (from code.visualstudio.com)

3. **CUDA Toolkit 12.6** (recommended) or CUDA 12.1/11.8
   - Download from: https://developer.nvidia.com/cuda-downloads

## Setup Instructions

### Step 1: Create Virtual Environment

Open Command Prompt and navigate to this folder:

```bash
cd Documents\VivaAIProject
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command line.

### Step 2: Install Dependencies

**Important**: Install PyTorch with CUDA support first:

**For CUDA 12.6 (RECOMMENDED):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**For CUDA 12.1:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install other dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Verify GPU Setup

Run the GPU check script:

```bash
python check_gpu.py
```

You should see confirmation that your RTX 3050 is detected.

## Project Structure

```
VivaAIProject/
├── venv/              # Virtual environment (created after Step 1)
├── check_gpu.py      # GPU verification script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Notes

- Always activate the virtual environment before working: `venv\Scripts\activate`
- The virtual environment keeps project dependencies isolated from your system Python
