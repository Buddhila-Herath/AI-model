"""
EAR & Blink Detection Live Test
Opens your webcam and shows real-time EAR values + blink counting.
Press ESC to exit. A summary prints at the end.
"""

import cv2
import mediapipe as mp
import time
import sys
import os

from modules.vision_engine import EyeAnalyzer

print("=" * 60)
print("EAR & BLINK DETECTION - Live Webcam Test")
print("=" * 60)
print()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face_landmarker.task ...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH,
    )
    print("[OK] Model downloaded")

print(f"[OK] Model: {MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

latest_result = None

def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    result_callback=on_result,
)

landmarker = FaceLandmarker.create_from_options(options)
print("[OK] FaceLandmarker ready")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam!")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[OK] Webcam opened ({w}x{h})")
print()
print("Press ESC to stop. Try blinking -- watch the counter!")
print("=" * 60)

eye_analyzer = EyeAnalyzer(ear_threshold=0.20, min_blink_frames=2)
frame_count = 0
start_time = time.time()

LEFT_EYE_IDX = EyeAnalyzer.LEFT_EYE
RIGHT_EYE_IDX = EyeAnalyzer.RIGHT_EYE

def draw_eye_contour(image, landmarks, indices, color, img_w, img_h):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    for i in range(len(pts)):
        cv2.line(image, pts[i], pts[(i + 1) % len(pts)], color, 1)

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame_count += 1
        ts = int(time.time() * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_img, ts)

        if latest_result and latest_result.face_landmarks:
            lms = latest_result.face_landmarks[0]

            ear_data = eye_analyzer.process_frame(lms, ts)

            draw_eye_contour(frame, lms, LEFT_EYE_IDX, (0, 255, 0), w, h)
            draw_eye_contour(frame, lms, RIGHT_EYE_IDX, (0, 255, 0), w, h)

            avg_ear = ear_data["avg_ear"]
            blinks = ear_data["blink_count"]
            blinking = ear_data["is_blinking"]

            bar_color = (0, 0, 255) if blinking else (0, 255, 0)
            bar_len = int(avg_ear * 400)
            cv2.rectangle(frame, (10, 80), (10 + bar_len, 100), bar_color, -1)
            cv2.rectangle(frame, (10, 80), (170, 100), (255, 255, 255), 1)

            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blinks}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if blinking:
                cv2.putText(frame, "BLINK!", (w - 160, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            elapsed_sec = time.time() - start_time
            if elapsed_sec > 5:
                bpm = blinks / (elapsed_sec / 60.0)
                cv2.putText(frame, f"BPM: {bpm:.1f}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        else:
            cv2.putText(frame, "No Face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("EAR Blink Test (ESC to exit)", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")

finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    if elapsed > 0:
        summary = eye_analyzer.get_session_summary(elapsed, 30)
        print()
        print("=" * 60)
        print("EAR & BLINK TEST RESULTS")
        print("=" * 60)
        print(f"  Duration          : {elapsed:.1f}s")
        print(f"  Frames processed  : {frame_count}")
        print(f"  Total blinks      : {summary['total_blinks']}")
        print(f"  Blinks per minute : {summary['blinks_per_minute']:.1f}")
        print(f"  Blink rate status : {summary['blink_rate_status'].upper()}")
        print(f"  Avg EAR           : {summary['avg_ear']:.4f}")
        print(f"  EAR std dev       : {summary['ear_std']:.4f}")
        print(f"  Avg blink duration: {summary['avg_blink_duration_ms']:.0f} ms")
        print()
        trend = summary["ear_trend"]
        print(f"  EAR trend (temporal):")
        print(f"    First third avg : {trend['first_third_avg']:.4f}")
        print(f"    Middle third avg: {trend['middle_third_avg']:.4f}")
        print(f"    Last third avg  : {trend['last_third_avg']:.4f}")
        print("=" * 60)

        if 12 <= summary["blinks_per_minute"] <= 22:
            print("[OK] Normal blink rate - you look calm and composed!")
        elif summary["blinks_per_minute"] < 12:
            print("[INFO] Low blink rate - could indicate focus or staring")
        else:
            print("[INFO] High blink rate - could indicate stress or fatigue")
        print()
