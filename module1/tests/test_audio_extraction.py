"""Unit tests for single-pass audio extraction via FFmpeg."""

import wave
from pathlib import Path

from src.audio_extraction import extract_audio
from src.id_generation import generate_video_id


def test_extract_audio_with_audio_track(
    sample_video_with_audio: Path, tmp_path: Path
):
    """Extracts a valid 16kHz mono WAV audio file from a video with audio."""
    video_id = generate_video_id(sample_video_with_audio)
    audio_dir = tmp_path / "audio"

    result = extract_audio(
        video_path=sample_video_with_audio,
        video_id=video_id,
        output_audio_dir=audio_dir,
    )

    assert result.has_audio is True
    assert result.audio_path is not None
    wav_path = Path(result.audio_path)
    assert wav_path.exists()
    assert wav_path.stat().st_size > 0

    # Verify WAV header specifications (16kHz, mono, 16-bit PCM)
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2  # 16-bit


def test_extract_audio_graceful_on_no_audio(
    sample_video_no_audio: Path, tmp_path: Path
):
    """Gracefully handles video without audio track, skipping extraction without error."""
    video_id = generate_video_id(sample_video_no_audio)
    audio_dir = tmp_path / "audio"

    result = extract_audio(
        video_path=sample_video_no_audio,
        video_id=video_id,
        output_audio_dir=audio_dir,
    )

    assert result.has_audio is False
    assert result.audio_path is None
    assert result.error_message is None
