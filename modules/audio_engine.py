"""
Audio Engine - "The Ears of the AI"
Transcribes audio via Whisper, analyzes speech patterns.
Detects filler words (um, ah, uh), pauses, and speaking pace.
"""

import whisper
import torch
import numpy as np
import os
import re
import time
import logging

logger = logging.getLogger(__name__)

FILLER_WORDS = {
    "um", "uh", "uhm", "uhh", "umm", "er", "err", "ah", "ahh",
    "like", "you know", "so", "basically", "actually", "literally",
    "i mean", "kind of", "sort of", "right",
}


class AudioEngine:
    """Processes audio to extract speech quality and confidence indicators."""

    def __init__(self, model_size="base", device=None):
        """
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                        'base' is optimal for RTX 3050 6GB VRAM.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = None

    def load_model(self):
        """Load Whisper model onto the specified device."""
        if self.model is not None:
            return

        logger.info(f"Loading Whisper '{self.model_size}' model on {self.device}...")
        start = time.time()
        self.model = whisper.load_model(self.model_size).to(self.device)
        elapsed = time.time() - start
        logger.info(f"Whisper model loaded in {elapsed:.2f}s")

        if self.device == "cuda":
            mem_gb = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory used by Whisper: {mem_gb:.2f} GB")

    def analyze_audio(self, audio_path):
        """
        Analyze an audio file and return speech quality metrics.

        Args:
            audio_path: Path to the audio file (wav, mp3, etc.).

        Returns:
            dict with keys:
                - transcription: Full text transcription
                - word_count: Total words spoken
                - duration_seconds: Audio duration
                - words_per_minute: Speaking pace
                - filler_word_count: Number of filler words detected
                - filler_word_ratio: Ratio of filler words to total words
                - filler_words_found: List of detected filler words with timestamps
                - pause_count: Number of significant pauses (>1.5s)
                - longest_pause: Duration of the longest pause in seconds
                - fluency_score: 0-100 score based on speech fluency
                - clarity_score: 0-100 score based on speech clarity
                - segments: Whisper segment-level data for research
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.load_model()

        logger.info(f"Transcribing: {audio_path}")
        start = time.time()

        result = self.model.transcribe(
            audio_path,
            fp16=(self.device == "cuda"),
            language="en",
            word_timestamps=True,
        )

        elapsed = time.time() - start
        logger.info(f"Transcription completed in {elapsed:.2f}s")

        transcription = result["text"].strip()
        segments = result.get("segments", [])

        duration = self._get_audio_duration(segments, audio_path)
        words = self._extract_words(transcription)
        word_count = len(words)
        wpm = (word_count / duration * 60) if duration > 0 else 0

        filler_analysis = self._detect_fillers(transcription, segments)
        pause_analysis = self._detect_pauses(segments)
        fluency_score = self._compute_fluency_score(
            word_count, filler_analysis["filler_word_count"],
            pause_analysis["pause_count"], wpm, duration
        )
        clarity_score = self._compute_clarity_score(segments)

        return {
            "transcription": transcription,
            "word_count": word_count,
            "duration_seconds": round(duration, 2),
            "words_per_minute": round(wpm, 2),
            "filler_word_count": filler_analysis["filler_word_count"],
            "filler_word_ratio": filler_analysis["filler_word_ratio"],
            "filler_words_found": filler_analysis["filler_words_found"],
            "pause_count": pause_analysis["pause_count"],
            "longest_pause": pause_analysis["longest_pause"],
            "average_pause": pause_analysis["average_pause"],
            "fluency_score": fluency_score,
            "clarity_score": clarity_score,
            "processing_time": round(elapsed, 2),
            "segments": segments,
        }

    def _get_audio_duration(self, segments, audio_path):
        """Get audio duration from segments or file info."""
        if segments:
            return segments[-1]["end"]
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 0.0

    def _extract_words(self, text):
        """Extract clean word list from transcription."""
        cleaned = re.sub(r"[^\w\s']", "", text.lower())
        return [w for w in cleaned.split() if w]

    def _detect_fillers(self, transcription, segments):
        """Detect filler words and their approximate positions."""
        text_lower = transcription.lower()
        words = self._extract_words(transcription)
        total_words = len(words)

        found_fillers = []

        for segment in segments:
            seg_text = segment.get("text", "").lower().strip()
            seg_words = self._extract_words(seg_text)

            for word in seg_words:
                if word in FILLER_WORDS:
                    found_fillers.append({
                        "word": word,
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                    })

        # Also check multi-word fillers
        for filler in FILLER_WORDS:
            if " " in filler and filler in text_lower:
                count = text_lower.count(filler)
                for _ in range(count):
                    found_fillers.append({"word": filler, "start": 0, "end": 0})

        filler_count = len(found_fillers)
        filler_ratio = round(filler_count / max(total_words, 1), 4)

        return {
            "filler_word_count": filler_count,
            "filler_word_ratio": filler_ratio,
            "filler_words_found": found_fillers,
        }

    def _detect_pauses(self, segments):
        """Detect significant pauses between speech segments."""
        if len(segments) < 2:
            return {"pause_count": 0, "longest_pause": 0.0, "average_pause": 0.0}

        pauses = []
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i - 1]["end"]
            if gap > 0.3:  # Only count gaps > 300ms
                pauses.append(gap)

        significant_pauses = [p for p in pauses if p > 1.5]

        return {
            "pause_count": len(significant_pauses),
            "longest_pause": round(max(pauses, default=0.0), 2),
            "average_pause": round(float(np.mean(pauses)) if pauses else 0.0, 2),
        }

    def _compute_fluency_score(self, word_count, filler_count, pause_count, wpm, duration):
        """
        Compute fluency score (0-100).
        Penalizes excessive fillers, long pauses, and abnormal speaking pace.
        """
        if duration < 1 or word_count < 5:
            return 0.0

        score = 100.0

        # Penalty for filler words (each filler costs up to 3 points)
        filler_ratio = filler_count / max(word_count, 1)
        score -= min(filler_ratio * 300, 40)

        # Penalty for long pauses
        score -= min(pause_count * 5, 25)

        # Penalty for speaking too fast (>180 wpm) or too slow (<80 wpm)
        if wpm > 180:
            score -= min((wpm - 180) * 0.3, 15)
        elif wpm < 80:
            score -= min((80 - wpm) * 0.3, 15)

        return round(np.clip(score, 0, 100), 2)

    def _compute_clarity_score(self, segments):
        """
        Compute clarity score based on Whisper's confidence.
        Higher average confidence = clearer speech.
        """
        if not segments:
            return 0.0

        confidences = []
        for seg in segments:
            avg_logprob = seg.get("avg_logprob", -1.0)
            no_speech_prob = seg.get("no_speech_prob", 0.0)
            confidence = np.exp(avg_logprob) * (1 - no_speech_prob)
            confidences.append(confidence)

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        clarity = np.clip(avg_confidence * 150, 0, 100)
        return round(clarity, 2)

    def cleanup(self):
        """Release GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Audio engine cleaned up")
