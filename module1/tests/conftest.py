"""Shared fixtures for Module 1 test suite."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from src.config import Module1Config


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    """Verify ffmpeg binary is available on system PATH."""
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def test_media_dir(tmp_path_factory) -> Path:
    """Session-scoped directory for generated test video media."""
    media_dir = tmp_path_factory.mktemp("test_media")
    return Path(media_dir)


@pytest.fixture(scope="session")
def sample_video_with_audio(test_media_dir: Path, ffmpeg_available: bool) -> Path:
    """Provide a clean MP4 video containing video and spoken audio tracks."""
    if not ffmpeg_available:
        pytest.skip("FFmpeg is required for test video generation.")

    # Check if pre-generated sample presentation video exists
    workspace_sample = Path(__file__).resolve().parent.parent / "input" / "videos" / "sample_presentation.mp4"
    if workspace_sample.exists():
        return workspace_sample

    video_path = test_media_dir / "valid_test_video.mp4"
    if not video_path.exists():
        # Synthesize speech audio using powershell
        temp_wav = test_media_dir / "temp_tts.wav"
        ps_cmd = (
            f'Add-Type -AssemblyName System.Speech; '
            f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.SetOutputToWaveFile("{temp_wav.as_posix()}"); '
            f'$s.Speak("This is a test of multimodal retrieval."); '
            f'$s.Dispose()'
        )
        res_ps = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True)
        if res_ps.returncode == 0 and temp_wav.exists():
            audio_args = ["-i", str(temp_wav)]
        else:
            audio_args = ["-f", "lavfi", "-i", "sine=frequency=440:duration=4"]

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=4:size=320x240:rate=25",
            *audio_args,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-shortest",
            str(video_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            pytest.fail(f"Failed to generate test video: {res.stderr.decode()}")

    return video_path


@pytest.fixture(scope="session")
def sample_video_no_audio(test_media_dir: Path, ffmpeg_available: bool) -> Path:
    """Generate a clean 3-second MKV video with NO audio stream."""
    if not ffmpeg_available:
        pytest.skip("FFmpeg is required for test video generation.")

    video_path = test_media_dir / "valid_no_audio.mkv"
    if not video_path.exists():
        # 3 seconds, 320x240, 25fps, testsrc video, no audio (-an)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(video_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            pytest.fail(f"Failed to generate no-audio video: {res.stderr.decode()}")

    return video_path


@pytest.fixture(scope="session")
def sample_corrupt_video(test_media_dir: Path) -> Path:
    """Create a simulated corrupt / unreadable .mp4 file."""
    corrupt_path = test_media_dir / "corrupt_video.mp4"
    if not corrupt_path.exists():
        # Write 4096 bytes of invalid header data
        corrupt_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096))
    return corrupt_path


@pytest.fixture
def temp_config(tmp_path: Path) -> Module1Config:
    """Create a temporary Module1Config isolated in tmp_path."""
    cfg = Module1Config(
        base_dir=tmp_path,
        input_videos_dir=tmp_path / "input" / "videos",
        output_dir=tmp_path / "output",
        frames_dir=tmp_path / "output" / "frames",
        audio_dir=tmp_path / "output" / "audio",
        transcripts_dir=tmp_path / "output" / "transcripts",
        metadata_dir=tmp_path / "output" / "metadata",
        frame_interval_seconds=2.0,
        whisper_model="tiny",
    )
    cfg.ensure_directories()
    return cfg
