# Phase 2: Check-and-Understand Flow

## Overview
Test Vision and Audio modules individually to understand how they perform on your RTX 3050 GPU.

---

## Step 1: Test Vision Module (The Eye of the AI)

### What it does:
- Tests MediaPipe Face Mesh for real-time facial landmark detection
- Checks if your GPU can handle real-time video processing
- Measures FPS and detection accuracy

### How to run:
```powershell
# Activate virtual environment first
cd Documents\VivaAIProject
.\venv\Scripts\Activate.ps1

# Run vision test
python test_vision.py
```

### What to expect:
1. **Camera opens** - Your webcam should activate
2. **Face detection** - Green landmarks appear on your face
3. **FPS counter** - Shows processing speed
4. **Status messages** - Prints stats every 30 frames

### Success criteria:
- ✅ **FPS ≥ 20**: Excellent performance
- ✅ **FPS ≥ 10**: Good performance  
- ⚠️ **FPS < 10**: May need optimization

### Controls:
- **ESC key**: Exit the test

### Expected output:
```
VISION TEST RESULTS
============================================================
Total frames processed: 300
Average FPS: 25.50
Face detection rate: 95.0%
Test duration: 11.76 seconds
GPU Memory Used: 45.23 MB

[SUCCESS] Vision module is working well!
Your RTX 3050 can handle real-time facial landmark detection.
```

---

## Step 2: Test Audio Module (The Ears of the AI)

### What it does:
- Tests Whisper speech recognition model
- Checks if your 6GB VRAM can hold the model
- Measures transcription speed and accuracy

### How to run:

#### Option A: Use existing audio file
```powershell
# Place a test audio file named 'test_audio.wav' in the project folder
# Then run:
python test_audio.py
```

#### Option B: Record audio on the fly
```powershell
# Record 5 seconds of audio using your microphone
python test_audio.py --record
```

### What to expect:
1. **Model loading** - Downloads 'base' model on first run (takes ~1 minute)
2. **GPU memory check** - Shows VRAM usage
3. **Transcription** - Processes your audio file
4. **Results** - Shows transcribed text and performance metrics

### Success criteria:
- ✅ **Processing speed ≥ 2x real-time**: Excellent
- ✅ **Processing speed ≥ 1x real-time**: Good
- ⚠️ **Processing speed < 1x real-time**: May need smaller model

### Expected output:
```
TRANSCRIPTION RESULTS
============================================================
Text: Hello, this is a test of the Whisper speech recognition system.

Language: en
Processing time: 2.45 seconds
Audio duration: 5.00 seconds
Processing speed: 2.04x real-time

GPU Memory Used: 1.23 GB
Memory increase: 1.23 GB

[SUCCESS] Audio module is working well!
Your RTX 3050 can handle Whisper transcription efficiently.
```

---

## Understanding the Results

### Vision Module Performance

| FPS Range | Status | Meaning |
|-----------|--------|---------|
| ≥ 30 FPS | Excellent | Can handle real-time processing easily |
| 20-29 FPS | Good | Suitable for real-time applications |
| 10-19 FPS | Acceptable | May need optimization for smooth video |
| < 10 FPS | Slow | Check system resources or reduce quality |

### Audio Module Performance

| Speed | Status | Meaning |
|-------|--------|---------|
| ≥ 2x real-time | Excellent | Very fast transcription |
| 1-2x real-time | Good | Suitable for real-time use |
| 0.5-1x real-time | Acceptable | May lag for long audio |
| < 0.5x real-time | Slow | Consider using smaller model |

### GPU Memory Usage

| Memory Used | Status | Action |
|-------------|--------|--------|
| < 2 GB | Excellent | Plenty of room for other operations |
| 2-4 GB | Good | Normal usage for AI models |
| 4-5 GB | High | Monitor for memory issues |
| > 5 GB | Critical | May cause out-of-memory errors |

---

## Troubleshooting

### Vision Module Issues

**Problem**: Camera doesn't open
- **Solution**: Check if another app is using the camera
- **Solution**: Try changing camera index: `cv2.VideoCapture(1)`

**Problem**: Low FPS
- **Solution**: Close other applications using GPU
- **Solution**: Reduce camera resolution in code
- **Solution**: Check Task Manager for GPU usage

**Problem**: No face detected
- **Solution**: Ensure good lighting
- **Solution**: Face the camera directly
- **Solution**: Check camera is working in other apps

### Audio Module Issues

**Problem**: Model download fails
- **Solution**: Check internet connection
- **Solution**: Model downloads automatically, wait for completion

**Problem**: Out of memory error
- **Solution**: Use smaller model: Change `"base"` to `"tiny"` or `"small"`
- **Solution**: Close other GPU applications
- **Solution**: Use CPU instead: Set `device = "cpu"`

**Problem**: Audio file not found
- **Solution**: Create `test_audio.wav` in project folder
- **Solution**: Use `--record` flag to record audio

**Problem**: Slow transcription
- **Solution**: Use `fp16=True` (already enabled for GPU)
- **Solution**: Use smaller Whisper model
- **Solution**: Check GPU is being used (should show CUDA device)

---

## Next Steps After Testing

Once both modules work:

1. **Note the performance metrics** - FPS, processing speed, memory usage
2. **Understand the "feel"** - How smooth is vision? How fast is audio?
3. **Plan Phase 3** - Build the full pipeline with these modules
4. **Optimize if needed** - Adjust model sizes based on your results

---

## Quick Reference

| Test | Command | Expected Duration |
|------|---------|-------------------|
| Vision | `python test_vision.py` | Run until ESC pressed |
| Audio | `python test_audio.py` | ~5-10 seconds |
| Audio (record) | `python test_audio.py --record` | ~10 seconds |

---

## Files Created

- `test_vision.py` - Vision module test script
- `test_audio.py` - Audio module test script
- `PHASE2_TEST_GUIDE.md` - This guide

---

**Ready to test? Start with Vision, then Audio!** 🚀
