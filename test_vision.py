"""
Phase 2: Vision Module Test - "The Eye of the AI"
Tests MediaPipe Face Mesh on RTX 3050 GPU
Checks if GPU can handle real-time facial landmark detection
"""

import cv2
import mediapipe as mp
import torch
import time
import sys

print("=" * 60)
print("VISION MODULE TEST - Face Mesh Detection")
print("=" * 60)
print()

# Check GPU availability
if torch.cuda.is_available():
    print(f"[OK] GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA Available: True")
else:
    print("[INFO] GPU not available, using CPU")
print()

# Initialize MediaPipe Face Mesh
print("Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print("[OK] Face Mesh initialized")
print()

# Initialize webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam!")
    print("Make sure your camera is connected and not being used by another application.")
    sys.exit(1)

print("[OK] Webcam opened")
print()

# Get camera properties
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera Resolution: {width}x{height}")
print(f"Camera FPS: {fps}")
print()

print("=" * 60)
print("VISION TEST STARTING")
print("Press ESC to exit")
print("=" * 60)
print()

frame_count = 0
start_time = time.time()
face_detected_count = 0

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("[WARNING] Failed to read frame")
            continue
        
        frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = face_mesh.process(image_rgb)
        
        # Draw face landmarks if detected
        if results.multi_face_landmarks:
            face_detected_count += 1
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Show status on frame
            cv2.putText(image, "Face Detected - Vision Working!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            current_fps = frame_count / elapsed
            cv2.putText(image, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Vision Test - Face Mesh Detection (Press ESC to exit)', image)
        
        # Check for ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            detection_rate = (face_detected_count / frame_count) * 100
            
            print(f"Frames processed: {frame_count} | "
                  f"FPS: {fps:.1f} | "
                  f"Face detection rate: {detection_rate:.1f}%")
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1024**2
                print(f"  GPU Memory Used: {memory_used:.2f} MB")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    elapsed = time.time() - start_time
    if elapsed > 0:
        avg_fps = frame_count / elapsed
        detection_rate = (face_detected_count / frame_count) * 100 if frame_count > 0 else 0
        
        print()
        print("=" * 60)
        print("VISION TEST RESULTS")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Face detection rate: {detection_rate:.1f}%")
        print(f"Test duration: {elapsed:.2f} seconds")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**2
            print(f"GPU Memory Used: {memory_used:.2f} MB")
        
        print()
        if avg_fps >= 20:
            print("[SUCCESS] Vision module is working well!")
            print("Your RTX 3050 can handle real-time facial landmark detection.")
        elif avg_fps >= 10:
            print("[OK] Vision module is working, but performance could be better.")
        else:
            print("[WARNING] Vision module is slow. Check your system resources.")
        print("=" * 60)
