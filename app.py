"""
Viva AI Performance Analyzer - Main Flask Server
The entry point that ties Vision, Audio, and Fusion engines together.
"""

import os
import time
import logging
import subprocess
import uuid
from flask import Flask, render_template, request, jsonify

from modules.vision_engine import VisionEngine
from modules.audio_engine import AudioEngine
from modules.fusion_engine import FusionEngine

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "webm", "wav", "mp3", "m4a", "mkv"}

# Ensure ffmpeg is findable -- refresh PATH from system environment
os.environ["PATH"] = (
    os.environ.get("PATH", "") + ";" +
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin")
)

# Initialize engines (lazy-loaded)
vision_engine = None
audio_engine = None
fusion_engine = FusionEngine()


def get_vision_engine():
    global vision_engine
    if vision_engine is None:
        vision_engine = VisionEngine()
    return vision_engine


def get_audio_engine():
    global audio_engine
    if audio_engine is None:
        audio_engine = AudioEngine(model_size="base")
    return audio_engine


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(video_path, audio_path):
    """Extract audio track from video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr[:500]}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint. Accepts video/audio file upload."""
    start_time = time.time()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload MP4, AVI, MOV, WebM, WAV, or MP3."}), 400

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_filename = f"{file_id}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)
    file.save(saved_path)
    logger.info(f"File saved: {saved_path} ({os.path.getsize(saved_path) / 1024 / 1024:.1f} MB)")

    try:
        is_video = ext in {"mp4", "avi", "mov", "webm", "mkv"}
        is_audio = ext in {"wav", "mp3", "m4a"}

        vision_result = None
        audio_result = None

        if is_video:
            # Extract audio from video
            audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_audio.wav")
            logger.info("Extracting audio from video...")
            extract_audio(saved_path, audio_path)

            # Analyze vision (video)
            logger.info("Running vision analysis...")
            ve = get_vision_engine()
            vision_result = ve.analyze_video(saved_path)

            # Analyze audio
            logger.info("Running audio analysis...")
            ae = get_audio_engine()
            audio_result = ae.analyze_audio(audio_path)

        elif is_audio:
            # Audio-only analysis
            logger.info("Running audio-only analysis...")
            ae = get_audio_engine()
            audio_result = ae.analyze_audio(saved_path)

            # Provide default vision scores for audio-only
            vision_result = {
                "eye_contact_score": 0,
                "head_stability_score": 0,
                "blink_summary": {
                    "total_blinks": 0,
                    "blinks_per_minute": 0.0,
                    "blink_rate_status": "normal",
                    "avg_ear": 0.0,
                    "ear_std": 0.0,
                    "avg_blink_duration_ms": 0.0,
                    "ear_trend": {},
                    "blink_timestamps_ms": [],
                },
                "emotion_scores": {},
                "frame_count": 0,
                "face_detected_ratio": 0,
                "fps": 0,
                "details": [],
            }

        # Fuse results
        logger.info("Running fusion engine...")
        final_result = fusion_engine.evaluate(vision_result, audio_result)

        # Add extra metadata for frontend
        final_result["metadata"]["transcription"] = audio_result.get("transcription", "")
        final_result["metadata"]["audio_word_count"] = audio_result.get("word_count", 0)
        final_result["metadata"]["words_per_minute"] = audio_result.get("words_per_minute", 0)
        final_result["metadata"]["audio_duration_seconds"] = audio_result.get("duration_seconds", 0)
        final_result["metadata"]["filler_word_count"] = audio_result.get("filler_word_count", 0)

        # Blink / EAR metrics
        blink = vision_result.get("blink_summary", {})
        final_result["metadata"]["total_blinks"] = blink.get("total_blinks", 0)
        final_result["metadata"]["blinks_per_minute"] = blink.get("blinks_per_minute", 0)
        final_result["metadata"]["blink_rate_status"] = blink.get("blink_rate_status", "normal")
        final_result["metadata"]["avg_ear"] = blink.get("avg_ear", 0)
        final_result["metadata"]["avg_blink_duration_ms"] = blink.get("avg_blink_duration_ms", 0)
        final_result["metadata"]["ear_trend"] = blink.get("ear_trend", {})

        elapsed = time.time() - start_time
        final_result["metadata"]["total_processing_time"] = round(elapsed, 2)
        logger.info(f"Analysis complete in {elapsed:.2f}s. Score: {final_result['overall_score']}, Grade: {final_result['grade']}")

        return jsonify(final_result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded files
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(file_id):
                try:
                    os.remove(os.path.join(UPLOAD_DIR, f))
                except OSError:
                    pass


@app.route("/health")
def health():
    """Health check endpoint."""
    import torch
    return jsonify({
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


if __name__ == "__main__":
    logger.info("Starting Viva AI Performance Analyzer...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    app.run(debug=True, host="0.0.0.0", port=5000)
