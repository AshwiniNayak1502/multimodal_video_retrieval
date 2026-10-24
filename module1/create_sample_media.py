"""Utility script to generate realistic sample videos for Module 1 testing."""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEOS_DIR = BASE_DIR / "input" / "videos"
INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def generate_speech_wav(wav_path: Path, text: str) -> bool:
    """Synthesize speech using Windows SpeechSynthesizer."""
    ps_script = (
        f'Add-Type -AssemblyName System.Speech; '
        f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        f'$s.SetOutputToWaveFile("{wav_path.as_posix()}"); '
        f'$s.Speak("{text}"); '
        f'$s.Dispose()'
    )
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0


def create_sample_videos():
    temp_wav = BASE_DIR / "temp_speech.wav"
    speech_text = (
        "Welcome to the multimodal video retrieval project. "
        "Module one extracts representative video frames and converts speech into structured transcripts."
    )

    print(f"Generating synthetic speech audio at {temp_wav}...")
    success = generate_speech_wav(temp_wav, speech_text)
    if not success:
        print("PowerShell TTS failed, falling back to sine audio via ffmpeg.")
        # Fallback to ffmpeg sine wave if TTS fails
        audio_input = ["-f", "lavfi", "-i", "sine=frequency=440:duration=8"]
    else:
        audio_input = ["-i", str(temp_wav)]

    # 1. Create sample_presentation.mp4 (video with speech audio)
    out_mp4 = INPUT_VIDEOS_DIR / "sample_presentation.mp4"
    print(f"Creating sample video with audio: {out_mp4}...")
    cmd_mp4 = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=8:size=640x360:rate=25",
        *audio_input,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd_mp4, check=True, capture_output=True)
    print(f"Created: {out_mp4} ({out_mp4.stat().st_size} bytes)")

    # 2. Create sample_silent_clip.mkv (video without audio)
    out_mkv = INPUT_VIDEOS_DIR / "sample_silent_clip.mkv"
    print(f"Creating sample silent video: {out_mkv}...")
    cmd_mkv = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "smptebars=duration=6:size=640x360:rate=25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_mkv),
    ]
    subprocess.run(cmd_mkv, check=True, capture_output=True)
    print(f"Created: {out_mkv} ({out_mkv.stat().st_size} bytes)")

    # Clean up temporary speech wav
    if temp_wav.exists():
        temp_wav.unlink()


if __name__ == "__main__":
    create_sample_videos()
