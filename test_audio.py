"""
Phase 2: Audio Module Test - "The Ears of the AI"
Tests Whisper speech recognition on RTX 3050 GPU
Checks if VRAM (6GB) can hold the Whisper model
"""

import whisper
import torch
import os
import sys
import time

print("=" * 60)
print("AUDIO MODULE TEST - Whisper Speech Recognition")
print("=" * 60)
print()

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"[OK] GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA Available: True")
    print(f"[OK] Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("[INFO] GPU not available, using CPU (will be slower)")
print()

# Check for test audio file
test_audio_file = "test_audio.wav"
if not os.path.exists(test_audio_file):
    print(f"[WARNING] Test audio file '{test_audio_file}' not found!")
    print()
    print("You have two options:")
    print("1. Record a test audio file:")
    print("   - Use any audio recording software")
    print("   - Save it as 'test_audio.wav' in this directory")
    print("   - Recommended: 5-10 seconds of speech")
    print()
    print("2. Use your microphone to record:")
    print("   - Run this script with --record flag")
    print("   - Example: python test_audio.py --record")
    print()
    
    # Check if user wants to record
    if "--record" in sys.argv:
        print("Recording audio...")
        try:
            import sounddevice as sd
            import soundfile as sf
            
            duration = 5  # seconds
            sample_rate = 16000
            
            print(f"Recording for {duration} seconds...")
            print("Speak now!")
            
            audio_data = sd.rec(int(duration * sample_rate), 
                              samplerate=sample_rate, 
                              channels=1, 
                              dtype='float32')
            sd.wait()  # Wait until recording is finished
            
            sf.write(test_audio_file, audio_data, sample_rate)
            print(f"[OK] Audio saved to {test_audio_file}")
        except ImportError:
            print("[ERROR] sounddevice not installed. Install it with:")
            print("  pip install sounddevice soundfile")
            sys.exit(1)
    else:
        print("[INFO] Skipping test. Create 'test_audio.wav' or use --record flag.")
        sys.exit(0)

print(f"[OK] Found test audio file: {test_audio_file}")
print()

# Load Whisper model
print("Loading Whisper model...")
print("[INFO] Using 'base' model - optimized for 6GB VRAM")
print("[INFO] This may take a minute on first run...")

start_load = time.time()
try:
    model = whisper.load_model("base")
    load_time = time.time() - start_load
    print(f"[OK] Model loaded in {load_time:.2f} seconds")
except Exception as e:
    print(f"[ERROR] Failed to load Whisper model: {e}")
    print("[INFO] The model will be downloaded automatically on first use.")
    sys.exit(1)

# Move model to GPU if available
if torch.cuda.is_available():
    print("Moving model to GPU...")
    model = model.to(device)
    print(f"[OK] Model moved to {device}")
    
    # Check GPU memory
    memory_before = torch.cuda.memory_allocated(0) / 1024**3
    print(f"GPU Memory Used: {memory_before:.2f} GB")
    print()

# Transcribe audio
print("=" * 60)
print("TRANSCRIBING AUDIO")
print("=" * 60)
print()

start_transcribe = time.time()

try:
    # Use fp16=True to save VRAM on RTX 3050
    result = model.transcribe(
        test_audio_file, 
        fp16=torch.cuda.is_available(),  # Use fp16 on GPU to save memory
        language="en"  # Optional: specify language
    )
    
    transcribe_time = time.time() - start_transcribe
    
    print("[OK] Transcription completed!")
    print()
    print("=" * 60)
    print("TRANSCRIPTION RESULTS")
    print("=" * 60)
    print(f"Text: {result['text']}")
    print()
    print(f"Language: {result.get('language', 'unknown')}")
    print(f"Processing time: {transcribe_time:.2f} seconds")
    
    # Show audio file info
    import soundfile as sf
    audio_info = sf.info(test_audio_file)
    audio_duration = audio_info.duration
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Processing speed: {audio_duration / transcribe_time:.2f}x real-time")
    print()
    
    # Check GPU memory after processing
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated(0) / 1024**3
        memory_used = memory_after - memory_before if memory_before > 0 else memory_after
        print(f"GPU Memory Used: {memory_after:.2f} GB")
        print(f"Memory increase: {memory_used:.2f} GB")
        print()
        
        # Check if memory usage is acceptable
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_percent = (memory_after / total_memory) * 100
        
        if memory_percent < 50:
            print("[SUCCESS] GPU memory usage is excellent!")
        elif memory_percent < 75:
            print("[OK] GPU memory usage is acceptable")
        else:
            print("[WARNING] GPU memory usage is high. Consider using 'tiny' or 'small' model.")
    
    print()
    print("=" * 60)
    print("AUDIO TEST RESULTS")
    print("=" * 60)
    
    if transcribe_time < audio_duration * 2:
        print("[SUCCESS] Audio module is working well!")
        print("Your RTX 3050 can handle Whisper transcription efficiently.")
    elif transcribe_time < audio_duration * 5:
        print("[OK] Audio module is working, but could be faster.")
    else:
        print("[WARNING] Audio processing is slow. Check your system resources.")
    
    print()
    print("=" * 60)

except Exception as e:
    print(f"[ERROR] Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] GPU memory cleared")
