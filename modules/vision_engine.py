"""
Vision Engine - "The Eye of the AI"
Analyzes video for gaze direction, emotion, head movement stability, and
blink behavior using Eye Aspect Ratio (EAR).
Returns confidence scores based on eye contact, head stability, and blink metrics.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EyeAnalyzer:
    """
    Eye Aspect Ratio (EAR) based blink detector and eye-openness tracker.

    EAR formula:  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    When the eye is open EAR stays roughly constant (~0.25-0.35).
    During a blink EAR drops sharply toward zero.  A blink is registered
    only on the *transition* from open -> closed, preventing double-counts.
    """

    # MediaPipe FaceMesh landmark indices (478-point model)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # p1..p6

    def __init__(self, ear_threshold=0.20, min_blink_frames=2):
        """
        Args:
            ear_threshold: EAR value below which the eye is considered closed.
            min_blink_frames: Minimum consecutive low-EAR frames to confirm a
                              blink (filters sensor noise / micro-twitches).
        """
        self.ear_threshold = ear_threshold
        self.min_blink_frames = min_blink_frames

        self.blink_count = 0
        self.consecutive_low_frames = 0
        self.is_blinking = False

        self.ear_history: list[float] = []
        self.blink_timestamps_ms: list[int] = []
        self.blink_durations_frames: list[int] = []

        self._current_blink_length = 0
        # Sliding window for short-term EAR trend (last 30 frames ~ 1 sec @ 30fps)
        self._ear_window: deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------------
    # Core EAR calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _landmark_distance(p1, p2):
        """Euclidean distance between two MediaPipe NormalizedLandmark points."""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def calculate_ear(self, landmarks, eye_indices):
        """
        Compute EAR for a single eye.

        Args:
            landmarks: List of 478 MediaPipe NormalizedLandmarks.
            eye_indices: 6-element list [p1, p2, p3, p4, p5, p6].

        Returns:
            float: The Eye Aspect Ratio value.
        """
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vertical_a = self._landmark_distance(p2, p6)
        vertical_b = self._landmark_distance(p3, p5)
        horizontal = self._landmark_distance(p1, p4)

        if horizontal < 1e-6:
            return 0.0

        return (vertical_a + vertical_b) / (2.0 * horizontal)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(self, landmarks, timestamp_ms):
        """
        Analyse one frame's landmarks and return blink/EAR data.

        Args:
            landmarks: List of 478 NormalizedLandmark from MediaPipe.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            dict with keys:
                - left_ear, right_ear, avg_ear: EAR values for the frame
                - is_blinking: Whether a blink is in progress
                - blink_count: Running total of confirmed blinks
                - short_term_avg_ear: Mean EAR over the last ~1 second
        """
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        self.ear_history.append(avg_ear)
        self._ear_window.append(avg_ear)

        # State machine: open -> closing -> closed -> opening -> open
        if avg_ear < self.ear_threshold:
            self.consecutive_low_frames += 1
            self._current_blink_length += 1

            if (self.consecutive_low_frames >= self.min_blink_frames
                    and not self.is_blinking):
                self.is_blinking = True
                self.blink_count += 1
                self.blink_timestamps_ms.append(timestamp_ms)
        else:
            if self.is_blinking:
                self.blink_durations_frames.append(self._current_blink_length)
            self.is_blinking = False
            self.consecutive_low_frames = 0
            self._current_blink_length = 0

        return {
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "avg_ear": round(avg_ear, 4),
            "is_blinking": self.is_blinking,
            "blink_count": self.blink_count,
            "short_term_avg_ear": round(
                float(np.mean(self._ear_window)), 4
            ),
        }

    # ------------------------------------------------------------------
    # Session-level analytics (called once after all frames processed)
    # ------------------------------------------------------------------

    def get_session_summary(self, total_duration_sec, fps):
        """
        Compute aggregate blink/EAR statistics for the whole video.

        Args:
            total_duration_sec: Video duration in seconds.
            fps: Video frame rate.

        Returns:
            dict of summary metrics for scoring and research logging.
        """
        duration_min = total_duration_sec / 60.0 if total_duration_sec > 0 else 1.0
        blinks_per_minute = self.blink_count / duration_min if duration_min > 0 else 0.0

        avg_ear = float(np.mean(self.ear_history)) if self.ear_history else 0.0
        ear_std = float(np.std(self.ear_history)) if self.ear_history else 0.0

        avg_blink_dur_frames = (
            float(np.mean(self.blink_durations_frames))
            if self.blink_durations_frames else 0.0
        )
        avg_blink_dur_ms = avg_blink_dur_frames / fps * 1000.0 if fps > 0 else 0.0

        # Temporal pattern: split EAR history into thirds
        thirds = self._split_into_thirds(self.ear_history)
        ear_trend = {
            "first_third_avg": round(float(np.mean(thirds[0])), 4) if thirds[0] else 0.0,
            "middle_third_avg": round(float(np.mean(thirds[1])), 4) if thirds[1] else 0.0,
            "last_third_avg": round(float(np.mean(thirds[2])), 4) if thirds[2] else 0.0,
        }

        # Normal blink rate is ~15-20/min.  <10 or >30 may signal stress/drowsiness.
        blink_rate_status = "normal"
        if blinks_per_minute < 10:
            blink_rate_status = "low"
        elif blinks_per_minute > 30:
            blink_rate_status = "high"

        return {
            "total_blinks": self.blink_count,
            "blinks_per_minute": round(blinks_per_minute, 2),
            "blink_rate_status": blink_rate_status,
            "avg_ear": round(avg_ear, 4),
            "ear_std": round(ear_std, 4),
            "avg_blink_duration_ms": round(avg_blink_dur_ms, 1),
            "ear_trend": ear_trend,
            "blink_timestamps_ms": self.blink_timestamps_ms,
        }

    def reset(self):
        """Clear all state for a fresh analysis run."""
        self.blink_count = 0
        self.consecutive_low_frames = 0
        self.is_blinking = False
        self.ear_history.clear()
        self.blink_timestamps_ms.clear()
        self.blink_durations_frames.clear()
        self._current_blink_length = 0
        self._ear_window.clear()

    @staticmethod
    def _split_into_thirds(lst):
        n = len(lst)
        if n == 0:
            return [], [], []
        t = n // 3
        return lst[:t], lst[t:2*t], lst[2*t:]


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

    def __init__(self, model_path=None, ear_threshold=0.20):
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
        self.eye_analyzer = EyeAnalyzer(ear_threshold=ear_threshold)

    def analyze_video(self, video_path):
        """
        Analyze a video file and return vision-based confidence metrics.

        Args:
            video_path: Path to the video file.

        Returns:
            dict with keys:
                - eye_contact_score (0-100)
                - head_stability_score (0-100)
                - blink_summary: EAR & blink analytics from EyeAnalyzer
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
        self.eye_analyzer.reset()
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

                    # --- EAR / Blink analysis ---
                    ear_data = self.eye_analyzer.process_frame(
                        landmarks, timestamp_ms
                    )
                    frame_data.update(ear_data)

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

        total_duration_sec = frame_idx / fps if fps > 0 else 0
        return self._compute_scores(
            frame_idx, face_detected_count, fps, total_duration_sec
        )

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

    def _compute_scores(self, total_frames, face_detected_count, fps,
                         total_duration_sec=0):
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

        # Blink / EAR summary
        blink_summary = self.eye_analyzer.get_session_summary(
            total_duration_sec, fps
        )

        return {
            "eye_contact_score": eye_contact_score,
            "head_stability_score": head_stability_score,
            "blink_summary": blink_summary,
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
            "blink_summary": {
                "total_blinks": 0,
                "blinks_per_minute": 0.0,
                "blink_rate_status": "normal",
                "avg_ear": 0.0,
                "ear_std": 0.0,
                "avg_blink_duration_ms": 0.0,
                "ear_trend": {
                    "first_third_avg": 0.0,
                    "middle_third_avg": 0.0,
                    "last_third_avg": 0.0,
                },
                "blink_timestamps_ms": [],
            },
            "emotion_scores": {},
            "frame_count": 0,
            "face_detected_ratio": 0.0,
            "fps": 0,
            "details": [],
        }
