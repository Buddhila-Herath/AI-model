"""
Fusion Engine - "The Brain of the AI"
Combines Vision and Audio scores into a unified viva performance grade.
Uses weighted scoring with configurable thresholds.
"""

import numpy as np
import time
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Default weights for final score calculation (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "eye_contact": 0.18,
    "head_stability": 0.08,
    "blink_behaviour": 0.07,
    "fluency": 0.28,
    "clarity": 0.19,
    "filler_penalty": 0.10,
    "pace": 0.10,
}

GRADE_THRESHOLDS = {
    "A+": 90, "A": 80, "B+": 70, "B": 60,
    "C+": 50, "C": 40, "D": 30, "F": 0,
}


class FusionEngine:
    """Fuses vision and audio analysis into a final viva performance assessment."""

    def __init__(self, weights=None, log_dir=None):
        """
        Args:
            weights: Dict of score component weights. Uses defaults if None.
            log_dir: Directory to save research logs. Uses 'logs/' if None.
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= total_weight

    def evaluate(self, vision_result, audio_result):
        """
        Combine vision and audio results into a final performance assessment.

        Args:
            vision_result: Dict from VisionEngine.analyze_video()
            audio_result: Dict from AudioEngine.analyze_audio()

        Returns:
            dict with keys:
                - overall_score (0-100): Final weighted performance score
                - grade: Letter grade (A+ to F)
                - component_scores: Individual component scores
                - feedback: List of human-readable feedback strings
                - strengths: List of identified strengths
                - improvements: List of suggested improvements
                - metadata: Processing metadata
        """
        start_time = time.time()

        component_scores = self._calculate_components(vision_result, audio_result)
        overall_score = self._weighted_score(component_scores)
        grade = self._assign_grade(overall_score)
        feedback = self._generate_feedback(component_scores, vision_result, audio_result)

        result = {
            "overall_score": overall_score,
            "grade": grade,
            "component_scores": component_scores,
            "feedback": feedback["messages"],
            "strengths": feedback["strengths"],
            "improvements": feedback["improvements"],
            "metadata": {
                "vision_frames_analyzed": vision_result.get("frame_count", 0),
                "audio_duration_seconds": audio_result.get("duration_seconds", 0),
                "audio_word_count": audio_result.get("word_count", 0),
                "processing_time": round(time.time() - start_time, 3),
                "weights_used": self.weights,
                "timestamp": datetime.now().isoformat(),
            },
        }

        self._save_log(result, vision_result, audio_result)
        return result

    def _calculate_components(self, vision, audio):
        """Calculate individual component scores."""

        eye_contact = vision.get("eye_contact_score", 0)
        head_stability = vision.get("head_stability_score", 0)
        fluency = audio.get("fluency_score", 0)
        clarity = audio.get("clarity_score", 0)

        # Filler word penalty (fewer fillers = higher score)
        filler_ratio = audio.get("filler_word_ratio", 0)
        filler_score = max(0, 100 - (filler_ratio * 500))

        # Pace score (optimal range: 120-150 wpm)
        wpm = audio.get("words_per_minute", 0)
        pace_score = self._pace_score(wpm)

        # Blink behaviour score (normal ~15-20 bpm is ideal)
        blink_score = self._blink_score(vision.get("blink_summary", {}))

        return {
            "eye_contact": round(eye_contact, 2),
            "head_stability": round(head_stability, 2),
            "blink_behaviour": round(blink_score, 2),
            "fluency": round(fluency, 2),
            "clarity": round(clarity, 2),
            "filler_penalty": round(np.clip(filler_score, 0, 100), 2),
            "pace": round(pace_score, 2),
        }

    def _pace_score(self, wpm):
        """Score speaking pace. Optimal range is 120-150 wpm."""
        if wpm <= 0:
            return 0.0
        if 120 <= wpm <= 150:
            return 100.0
        elif 100 <= wpm < 120:
            return 80 + (wpm - 100) * 1.0
        elif 150 < wpm <= 180:
            return 80 + (180 - wpm) * 0.67
        elif 80 <= wpm < 100:
            return 60 + (wpm - 80) * 1.0
        elif 180 < wpm <= 200:
            return 60 + (200 - wpm) * 1.0
        else:
            return max(0, 40 - abs(wpm - 135) * 0.3)

    def _blink_score(self, blink_summary):
        """
        Score blink behaviour.  Normal relaxed blink rate is ~15-20/min.
        Very low (<8) may indicate staring / stress; very high (>35) may
        indicate anxiety or fatigue.  Avg blink duration ~100-400 ms is normal.
        """
        bpm = blink_summary.get("blinks_per_minute", 0)
        avg_dur = blink_summary.get("avg_blink_duration_ms", 200)

        if bpm <= 0:
            return 50.0  # no blink data (e.g. no face detected)

        # Rate component (peak 100 at 15-20 bpm, decays outward)
        if 12 <= bpm <= 22:
            rate_score = 100.0
        elif 8 <= bpm < 12:
            rate_score = 70 + (bpm - 8) * 7.5
        elif 22 < bpm <= 30:
            rate_score = 70 + (30 - bpm) * 3.75
        elif bpm < 8:
            rate_score = max(30, 70 - (8 - bpm) * 10)
        else:
            rate_score = max(20, 70 - (bpm - 30) * 5)

        # Duration component (normal blink ~100-400 ms)
        if 80 <= avg_dur <= 400:
            dur_score = 100.0
        elif avg_dur < 80:
            dur_score = 70.0
        else:
            dur_score = max(40, 100 - (avg_dur - 400) * 0.15)

        return np.clip(rate_score * 0.7 + dur_score * 0.3, 0, 100)

    def _weighted_score(self, components):
        """Compute the weighted overall score."""
        score = sum(
            components.get(key, 0) * weight
            for key, weight in self.weights.items()
        )
        return round(np.clip(score, 0, 100), 2)

    def _assign_grade(self, score):
        """Assign a letter grade based on the overall score."""
        for grade, threshold in GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"

    def _generate_feedback(self, components, vision, audio):
        """Generate human-readable feedback, strengths, and improvements."""
        messages = []
        strengths = []
        improvements = []

        # Eye contact feedback
        ec = components["eye_contact"]
        if ec >= 80:
            strengths.append("Excellent eye contact throughout the presentation")
        elif ec >= 60:
            messages.append("Good eye contact, but could be more consistent")
        else:
            improvements.append("Maintain more consistent eye contact with the camera")

        # Head stability feedback
        hs = components["head_stability"]
        if hs >= 80:
            strengths.append("Steady head position showing composure")
        elif hs < 50:
            improvements.append("Try to keep your head more stable during the viva")

        # Fluency feedback
        fl = components["fluency"]
        if fl >= 80:
            strengths.append("Very fluent speech with good flow")
        elif fl >= 60:
            messages.append("Speech is reasonably fluent with minor hesitations")
        else:
            improvements.append("Practice speaking more fluently with fewer pauses")

        # Filler words feedback
        filler_count = audio.get("filler_word_count", 0)
        if filler_count == 0:
            strengths.append("No filler words detected - very clean speech")
        elif filler_count <= 3:
            messages.append(f"Only {filler_count} filler word(s) detected - good control")
        else:
            improvements.append(
                f"Reduce filler words ({filler_count} detected: um, uh, like, etc.)"
            )

        # Speaking pace feedback
        wpm = audio.get("words_per_minute", 0)
        if 120 <= wpm <= 150:
            strengths.append(f"Ideal speaking pace ({wpm:.0f} words per minute)")
        elif wpm > 180:
            improvements.append(f"Speaking too fast ({wpm:.0f} wpm). Aim for 120-150 wpm")
        elif wpm < 80 and wpm > 0:
            improvements.append(f"Speaking too slow ({wpm:.0f} wpm). Aim for 120-150 wpm")

        # Blink behaviour feedback
        blink_summary = vision.get("blink_summary", {})
        bpm = blink_summary.get("blinks_per_minute", 0)
        blink_status = blink_summary.get("blink_rate_status", "normal")
        bb = components.get("blink_behaviour", 50)

        if bb >= 80:
            strengths.append(
                f"Natural blink rate ({bpm:.0f}/min) suggesting calm composure"
            )
        elif blink_status == "low" and bpm > 0:
            improvements.append(
                f"Very low blink rate ({bpm:.0f}/min) - this may indicate "
                "staring or tension. Try to relax your eyes"
            )
        elif blink_status == "high":
            improvements.append(
                f"Elevated blink rate ({bpm:.0f}/min) - may indicate nervousness. "
                "Practice deep breathing before your viva"
            )

        # Clarity feedback
        cl = components["clarity"]
        if cl >= 80:
            strengths.append("Clear and well-articulated speech")
        elif cl < 50:
            improvements.append("Try to speak more clearly and articulate words better")

        # Pause feedback
        pause_count = audio.get("pause_count", 0)
        longest_pause = audio.get("longest_pause", 0)
        if pause_count > 3:
            improvements.append(
                f"Too many long pauses ({pause_count}). "
                f"Longest pause: {longest_pause:.1f}s"
            )

        if not messages and not strengths and not improvements:
            messages.append("Analysis complete. Review individual scores for details.")

        return {
            "messages": messages,
            "strengths": strengths,
            "improvements": improvements,
        }

    def _save_log(self, result, vision_result, audio_result):
        """Save analysis results to a JSON log file for thesis research data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"viva_analysis_{timestamp}.json")

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "result": {
                "overall_score": result["overall_score"],
                "grade": result["grade"],
                "component_scores": result["component_scores"],
                "feedback": result["feedback"],
                "strengths": result["strengths"],
                "improvements": result["improvements"],
            },
            "vision_summary": {
                "eye_contact_score": vision_result.get("eye_contact_score"),
                "head_stability_score": vision_result.get("head_stability_score"),
                "face_detected_ratio": vision_result.get("face_detected_ratio"),
                "frame_count": vision_result.get("frame_count"),
                "blink_summary": vision_result.get("blink_summary", {}),
            },
            "audio_summary": {
                "word_count": audio_result.get("word_count"),
                "duration_seconds": audio_result.get("duration_seconds"),
                "words_per_minute": audio_result.get("words_per_minute"),
                "filler_word_count": audio_result.get("filler_word_count"),
                "fluency_score": audio_result.get("fluency_score"),
                "clarity_score": audio_result.get("clarity_score"),
                "transcription": audio_result.get("transcription"),
            },
        }

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Research log saved: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save log: {e}")
