"""Unit tests for WhisperX local audio transcription."""

import json
import subprocess
from pathlib import Path

from src.audio_extraction import extract_audio
from src.id_generation import generate_video_id
from src.transcription import LocalTranscriber


def test_transcription_structure(sample_video_with_audio: Path, tmp_path: Path):
    """Verifies that transcribing audio yields valid structured JSON with required keys."""
    video_id = generate_video_id(sample_video_with_audio)
    audio_dir = tmp_path / "audio"
    transcripts_dir = tmp_path / "transcripts"

    audio_res = extract_audio(
        video_path=sample_video_with_audio,
        video_id=video_id,
        output_audio_dir=audio_dir,
    )
    assert audio_res.has_audio is True
    assert audio_res.audio_path is not None

    transcriber = LocalTranscriber(model_name="tiny")
    t_res = transcriber.transcribe_audio(
        audio_path=audio_res.audio_path,
        video_id=video_id,
        output_dir=transcripts_dir,
    )

    assert t_res.video_id == video_id
    assert Path(t_res.transcript_path).exists()

    with open(t_res.transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["video_id"] == video_id
    assert "language" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)


def test_transcription_non_speech_silence_handling(tmp_path: Path):
    """Verifies that audio with no speech (e.g. pure tone) produces an empty transcript without crashing."""
    transcripts_dir = tmp_path / "transcripts"
    audio_path = tmp_path / "pure_tone.wav"

    # Generate 2s 440Hz sine wave WAV
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    transcriber = LocalTranscriber(model_name="tiny")
    res = transcriber.transcribe_audio(
        audio_path=audio_path,
        video_id="vid_test_tone",
        output_dir=transcripts_dir,
    )

    assert res.video_id == "vid_test_tone"
    assert res.segment_count == 0
    assert res.word_count == 0
    assert Path(res.transcript_path).exists()
