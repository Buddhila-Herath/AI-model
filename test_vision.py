"""
Phase 2: Vision Module Test - "The Eye of the AI"
Tests MediaPipe FaceLandmarker on RTX 3050 GPU
Checks if GPU can handle real-time facial landmark detection
Uses the new MediaPipe Tasks API (0.10.32+)
"""

import cv2
import mediapipe as mp
import torch
import time
import sys
import os
import numpy as np

print("=" * 60)
print("VISION MODULE TEST - Face Landmark Detection")
print("=" * 60)
print()

# Check GPU
if torch.cuda.is_available():
    print(f"[OK] GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA Available: True")
else:
    print("[INFO] GPU not available, using CPU")
print()

# Check model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("[ERROR] face_landmarker.task model not found!")
    print("Downloading model...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("[OK] Model downloaded")

print("[OK] Model file found: face_landmarker.task")
print()

# Initialize FaceLandmarker with the new Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

latest_result = None
latest_timestamp = 0

def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    result_callback=on_result
)

print("Initializing FaceLandmarker...")
landmarker = FaceLandmarker.create_from_options(options)
print("[OK] FaceLandmarker initialized")
print()

# Initialize webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam!")
    print("Make sure your camera is connected and not used by another app.")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[OK] Webcam opened ({width}x{height})")
print()
print("=" * 60)
print("VISION TEST RUNNING - Press ESC to exit")
print("=" * 60)
print()

frame_count = 0
face_detected_count = 0
start_time = time.time()

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def draw_landmarks(image, landmarks, w, h):
    """Draw face landmarks on the image."""
    if not landmarks:
        return
    
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    
    for i in range(len(FACE_OVAL) - 1):
        idx1 = FACE_OVAL[i]
        idx2 = FACE_OVAL[i + 1]
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
            pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
            cv2.line(image, pt1, pt2, (0, 255, 0), 1)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_count += 1
        timestamp_ms = int(time.time() * 1000)

        # Convert to MediaPipe Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Send to FaceLandmarker (async)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw results from latest callback
        if latest_result and latest_result.face_landmarks:
            face_detected_count += 1
            for face_landmarks in latest_result.face_landmarks:
                draw_landmarks(image, face_landmarks, width, height)

            num_landmarks = len(latest_result.face_landmarks[0])
            cv2.putText(image, f"Face Detected! ({num_landmarks} landmarks)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No Face Detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Vision Test - Face Landmarks (Press ESC to exit)", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Print stats every 30 frames
        if frame_count % 30 == 0:
            fps = frame_count / elapsed if elapsed > 0 else 0
            det_rate = (face_detected_count / frame_count) * 100
            print(f"Frames: {frame_count} | FPS: {fps:.1f} | Detection rate: {det_rate:.1f}%")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    if frame_count > 0 and elapsed > 0:
        avg_fps = frame_count / elapsed
        det_rate = (face_detected_count / frame_count) * 100

        print()
        print("=" * 60)
        print("VISION TEST RESULTS")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Face detection rate: {det_rate:.1f}%")
        print(f"Test duration: {elapsed:.2f} seconds")
        print()

        if avg_fps >= 20:
            print("[SUCCESS] Vision module is working great!")
            print("Your RTX 3050 handles real-time face landmark detection smoothly.")
        elif avg_fps >= 10:
            print("[OK] Vision module is working. Performance is acceptable.")
        else:
            print("[WARNING] Vision module is slow. Check system resources.")
        print("=" * 60)
