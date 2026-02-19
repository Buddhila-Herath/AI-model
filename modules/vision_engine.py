"""
Vision Engine - "The Eye of the AI"
Analyzes video for gaze direction, emotion, and head movement stability.
Returns confidence scores based on eye contact and head stability.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)


class VisionEngine:
    """Processes video frames to extract visual confidence indicators."""

    # Key landmark indices for gaze and head pose estimation
    # MediaPipe FaceMesh uses 478 landmarks
    LEFT_EYE_IRIS = [468, 469, 470, 471, 472]
    RIGHT_EYE_IRIS = [473, 474, 475, 476, 477]
    LEFT_EYE_CORNERS = [33, 133]
    RIGHT_EYE_CORNERS = [362, 263]
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    FOREHEAD = 10

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "face_landmarker.task"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        self.model_path = model_path
        self.frame_results = []
        self.head_positions = []

    def analyze_video(self, video_path):
        """
        Analyze a video file and return vision-based confidence metrics.

        Args:
            video_path: Path to the video file.

        Returns:
            dict with keys:
                - eye_contact_score (0-100): How consistently the person looks at camera
                - head_stability_score (0-100): How stable the head position is
                - emotion_scores: Dict of detected emotion indicators
                - frame_count: Total frames analyzed
                - face_detected_ratio: Percentage of frames with a face detected
                - details: Per-frame analysis data for research logging
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Analyzing video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )

        landmarker = FaceLandmarker.create_from_options(options)

        self.frame_results = []
        self.head_positions = []
        face_detected_count = 0
        frame_idx = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp_ms = int((frame_idx / fps) * 1000)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                frame_data = {
                    "frame": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "face_detected": False,
                    "gaze_centered": False,
                    "head_pitch": 0.0,
                    "head_yaw": 0.0,
                }

                if result.face_landmarks:
                    face_detected_count += 1
                    landmarks = result.face_landmarks[0]
                    frame_data["face_detected"] = True

                    gaze_offset = self._estimate_gaze(landmarks, width, height)
                    frame_data["gaze_offset"] = gaze_offset
                    frame_data["gaze_centered"] = gaze_offset < 0.15

                    pitch, yaw = self._estimate_head_pose(landmarks, width, height)
                    frame_data["head_pitch"] = pitch
                    frame_data["head_yaw"] = yaw
                    self.head_positions.append((pitch, yaw))

                    if result.face_blendshapes:
                        frame_data["blendshapes"] = {
                            bs.category_name: bs.score
                            for bs in result.face_blendshapes[0]
                        }

                self.frame_results.append(frame_data)

        finally:
            landmarker.close()
            cap.release()

        elapsed = time.time() - start_time
        logger.info(f"Video analysis complete: {frame_idx} frames in {elapsed:.2f}s")

        return self._compute_scores(frame_idx, face_detected_count, fps)

    def _estimate_gaze(self, landmarks, img_w, img_h):
        """Estimate how centered the gaze is (0 = looking at camera, 1 = looking away)."""
        left_iris = landmarks[self.LEFT_EYE_IRIS[0]]
        right_iris = landmarks[self.RIGHT_EYE_IRIS[0]]

        left_inner = landmarks[self.LEFT_EYE_CORNERS[0]]
        left_outer = landmarks[self.LEFT_EYE_CORNERS[1]]
        right_inner = landmarks[self.RIGHT_EYE_CORNERS[0]]
        right_outer = landmarks[self.RIGHT_EYE_CORNERS[1]]

        left_ratio = self._iris_position_ratio(left_iris, left_inner, left_outer)
        right_ratio = self._iris_position_ratio(right_iris, right_inner, right_outer)

        avg_ratio = (left_ratio + right_ratio) / 2.0
        return abs(avg_ratio - 0.5) * 2.0

    def _iris_position_ratio(self, iris, inner_corner, outer_corner):
        """Calculate iris position as ratio between eye corners (0=inner, 1=outer)."""
        eye_width = np.sqrt(
            (outer_corner.x - inner_corner.x) ** 2 +
            (outer_corner.y - inner_corner.y) ** 2
        )
        if eye_width < 1e-6:
            return 0.5

        iris_dist = np.sqrt(
            (iris.x - inner_corner.x) ** 2 +
            (iris.y - inner_corner.y) ** 2
        )
        return np.clip(iris_dist / eye_width, 0.0, 1.0)

    def _estimate_head_pose(self, landmarks, img_w, img_h):
        """Estimate head pitch and yaw from landmark positions."""
        nose = landmarks[self.NOSE_TIP]
        chin = landmarks[self.CHIN]
        left_ear = landmarks[self.LEFT_EAR]
        right_ear = landmarks[self.RIGHT_EAR]
        forehead = landmarks[self.FOREHEAD]

        # Yaw: horizontal rotation (left-right head turn)
        ear_midpoint_x = (left_ear.x + right_ear.x) / 2.0
        yaw = (nose.x - ear_midpoint_x) * 2.0

        # Pitch: vertical rotation (looking up/down)
        face_height = abs(forehead.y - chin.y)
        if face_height > 1e-6:
            nose_relative = (nose.y - forehead.y) / face_height
            pitch = (nose_relative - 0.6) * 2.0
        else:
            pitch = 0.0

        return pitch, yaw

    def _compute_scores(self, total_frames, face_detected_count, fps):
        """Compute final confidence scores from frame-level analysis."""
        if total_frames == 0:
            return self._empty_result()

        face_ratio = face_detected_count / total_frames

        # Eye contact score
        gaze_centered_frames = sum(
            1 for f in self.frame_results if f.get("gaze_centered", False)
        )
        eye_contact_raw = gaze_centered_frames / max(face_detected_count, 1)
        eye_contact_score = round(np.clip(eye_contact_raw * 100, 0, 100), 2)

        # Head stability score (lower variance = more stable)
        head_stability_score = 100.0
        if len(self.head_positions) > 1:
            pitches = [p[0] for p in self.head_positions]
            yaws = [p[1] for p in self.head_positions]
            pitch_var = np.var(pitches)
            yaw_var = np.var(yaws)
            total_var = pitch_var + yaw_var
            head_stability_score = round(np.clip(100 - (total_var * 500), 0, 100), 2)

        # Emotion analysis from blendshapes
        emotion_scores = self._aggregate_emotions()

        return {
            "eye_contact_score": eye_contact_score,
            "head_stability_score": head_stability_score,
            "emotion_scores": emotion_scores,
            "frame_count": total_frames,
            "face_detected_ratio": round(face_ratio * 100, 2),
            "fps": fps,
            "details": self.frame_results,
        }

    def _aggregate_emotions(self):
        """Aggregate blendshape-based emotion indicators across all frames."""
        emotion_keys = [
            "browDownLeft", "browDownRight", "browInnerUp",
            "eyeSquintLeft", "eyeSquintRight",
            "jawOpen", "mouthSmileLeft", "mouthSmileRight",
            "mouthFrownLeft", "mouthFrownRight",
        ]

        aggregated = {k: [] for k in emotion_keys}
        for frame in self.frame_results:
            bs = frame.get("blendshapes", {})
            for key in emotion_keys:
                if key in bs:
                    aggregated[key].append(bs[key])

        return {
            key: round(float(np.mean(vals)), 4) if vals else 0.0
            for key, vals in aggregated.items()
        }

    def _empty_result(self):
        return {
            "eye_contact_score": 0.0,
            "head_stability_score": 0.0,
            "emotion_scores": {},
            "frame_count": 0,
            "face_detected_ratio": 0.0,
            "fps": 0,
            "details": [],
        }
