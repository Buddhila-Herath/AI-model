# Setup script for VivaAIProject
# Run this script in PowerShell: .\setup.ps1

Write-Host "=== VivaAIProject Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python from python.org" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host ""
Write-Host "=== Installation Instructions ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Install PyTorch with CUDA 12.1 support:" -ForegroundColor Yellow
Write-Host "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor White
Write-Host ""
Write-Host "2. Install other dependencies:" -ForegroundColor Yellow
Write-Host "   pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "3. Verify GPU setup:" -ForegroundColor Yellow
Write-Host "   python check_gpu.py" -ForegroundColor White
Write-Host ""
Write-Host "Setup script completed!" -ForegroundColor Green
Write-Host "Remember to activate the virtual environment before working:" -ForegroundColor Cyan
Write-Host "   venv\Scripts\activate" -ForegroundColor White
