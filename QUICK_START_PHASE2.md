# Quick Start: Phase 2 Testing

## 🚀 Quick Commands

### Activate Environment (Always do this first!)
```powershell
cd Documents\VivaAIProject
.\venv\Scripts\Activate.ps1
```

### Option 1: Interactive Menu (Easiest)
```powershell
python run_phase2_tests.py
```
Then select which test to run from the menu.

### Option 2: Run Tests Directly

#### Test Vision Module
```powershell
python test_vision.py
```
- Opens your webcam
- Shows face mesh detection
- Press **ESC** to exit
- Shows FPS and performance stats

#### Test Audio Module
```powershell
# If you have test_audio.wav file:
python test_audio.py

# Or record audio on the fly:
python test_audio.py --record
```
- Loads Whisper model
- Transcribes audio
- Shows GPU memory usage
- Displays transcription results

---

## 📋 What Each Test Does

### Vision Test (`test_vision.py`)
✅ Tests MediaPipe Face Mesh  
✅ Real-time facial landmark detection  
✅ Measures FPS and detection rate  
✅ Checks GPU performance  

**Success**: FPS ≥ 20 means your GPU handles real-time vision well!

### Audio Test (`test_audio.py`)
✅ Tests Whisper speech recognition  
✅ Checks if 6GB VRAM can hold the model  
✅ Measures transcription speed  
✅ Shows GPU memory usage  

**Success**: Processing speed ≥ 2x real-time means efficient transcription!

---

## 📊 Understanding Results

### Vision Performance
- **≥ 30 FPS**: Excellent - Smooth real-time processing
- **20-29 FPS**: Good - Suitable for real-time use
- **10-19 FPS**: Acceptable - May need optimization
- **< 10 FPS**: Slow - Check system resources

### Audio Performance  
- **≥ 2x real-time**: Excellent - Very fast
- **1-2x real-time**: Good - Real-time capable
- **0.5-1x real-time**: Acceptable - May lag
- **< 0.5x real-time**: Slow - Use smaller model

### GPU Memory
- **< 2 GB**: Excellent - Plenty of room
- **2-4 GB**: Good - Normal usage
- **4-5 GB**: High - Monitor closely
- **> 5 GB**: Critical - May cause errors

---

## 🎯 What to Check

After running tests, verify:

1. **Latency**: How fast does processing happen?
   - Vision: Should be smooth (≥20 FPS)
   - Audio: Should be faster than real-time (≥2x)

2. **VRAM Usage**: Check Task Manager
   - Open Task Manager → Performance → GPU
   - Watch "Dedicated GPU Memory"
   - Should NOT stay at 100% (6.0/6.0 GB)

3. **Accuracy**: Does it work correctly?
   - Vision: Face landmarks appear correctly
   - Audio: Transcription matches what you said

---

## 📁 Files Created

- `test_vision.py` - Vision module test
- `test_audio.py` - Audio module test  
- `run_phase2_tests.py` - Interactive test runner
- `PHASE2_TEST_GUIDE.md` - Detailed guide
- `QUICK_START_PHASE2.md` - This file

---

## ⚡ Next Steps

After testing both modules:

1. ✅ Note your performance metrics
2. ✅ Understand how they "feel" on your GPU
3. ✅ Plan Phase 3: Build the full pipeline
4. ✅ Optimize if needed based on results

---

## 🆘 Troubleshooting

**Vision test doesn't open camera?**
- Check if another app is using camera
- Try: `python test_vision.py`

**Audio test says file not found?**
- Create `test_audio.wav` OR
- Use: `python test_audio.py --record`

**Low performance?**
- Close other GPU applications
- Check Task Manager for GPU usage
- Restart your computer

**Need help?**
- Read `PHASE2_TEST_GUIDE.md` for detailed info
- Check `HOW_TO_CHECK.md` for GPU verification

---

**Ready? Start testing!** 🎬

```powershell
python run_phase2_tests.py
```
