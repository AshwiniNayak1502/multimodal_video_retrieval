"""Speech-to-text transcription module using local WhisperX."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import whisperx

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_VAD_METHOD,
    DEFAULT_WHISPER_MODEL,
    get_default_compute_type,
    get_default_device,
)

logger = logging.getLogger("module1.transcription")


@dataclass
class WordAlignment:
    """Word-level timestamped token."""

    word: str
    start: Optional[float]
    end: Optional[float]
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert word alignment to JSON dictionary."""
        return asdict(self)


@dataclass
class SpeechSegment:
    """Speech segment with start time, end time, text, and optional word timestamps."""

    segment_id: int
    start_time: float
    end_time: float
    text: str
    words: List[Dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert segment to JSON dictionary."""
        return asdict(self)


@dataclass
class TranscriptResult:
    """Summary of transcription result for a video."""

    video_id: str
    transcript_path: str
    language: str
    segment_count: int
    word_count: int
    has_word_alignment: bool
    data: Dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert transcript result to JSON dictionary."""
        return asdict(self)


class LocalTranscriber:
    """Manages WhisperX model lifecycle and audio transcription operations."""

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        vad_method: str = DEFAULT_VAD_METHOD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device or get_default_device()
        self.compute_type = compute_type or get_default_compute_type(
            self.device
        )
        self.vad_method = vad_method
        self.batch_size = batch_size
        self._model = None
        self._align_models: Dict[str, tuple] = {}

    def _get_model(self):
        """Lazily load the WhisperX model."""
        if self._model is None:
            logger.info(
                "Loading WhisperX model '%s' (device=%s, compute_type=%s, vad=%s)...",
                self.model_name,
                self.device,
                self.compute_type,
                self.vad_method,
            )
            self._model = whisperx.load_model(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                vad_method=self.vad_method,
            )
            logger.info("WhisperX model '%s' loaded successfully.", self.model_name)
        return self._model

    def _get_align_model(self, language_code: str):
        """Cache and retrieve alignment model for a given language code."""
        if language_code not in self._align_models:
            logger.info(
                "Loading WhisperX alignment model for language '%s' on %s...",
                language_code,
                self.device,
            )
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
            self._align_models[language_code] = (align_model, align_metadata)
        return self._align_models[language_code]

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        video_id: str,
        output_dir: Union[str, Path],
    ) -> TranscriptResult:
        """Transcribe extracted WAV audio file into structured timestamped JSON.

        Args:
            audio_path: Path to mono 16kHz WAV audio file.
            video_id: Deterministic video identifier.
            output_dir: Directory where transcript JSON will be stored.

        Returns:
            TranscriptResult with details and file paths.

        Raises:
            FileNotFoundError: If audio file does not exist.
        """
        a_path = Path(audio_path).resolve()
        if not a_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {a_path}")

        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = target_dir / f"{video_id}.json"

        logger.info(
            "Transcribing audio for video %s from %s...", video_id, a_path.name
        )

        model = self._get_model()
        audio = whisperx.load_audio(str(a_path))

        # Base Whisper transcription
        try:
            result = model.transcribe(audio, batch_size=self.batch_size)
            detected_language = result.get("language", "en")
            raw_segments = result.get("segments", [])
        except IndexError:
            logger.warning(
                "No speech segments detected in audio for %s (audio contains non-speech or silence). Returning empty transcript.",
                video_id,
            )
            detected_language = "unknown"
            raw_segments = []

        logger.info(
            "Transcription completed for %s: detected language='%s', initial segments=%d",
            video_id,
            detected_language,
            len(raw_segments),
        )

        # Word-level alignment
        has_word_alignment = False
        aligned_segments = raw_segments

        if raw_segments:
            try:
                align_model, align_metadata = self._get_align_model(
                    detected_language
                )
                aligned_result = whisperx.align(
                    raw_segments,
                    align_model,
                    align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                aligned_segments = aligned_result.get(
                    "segments", raw_segments
                )
                has_word_alignment = True
                logger.info(
                    "Word-level alignment succeeded for video %s (%d segments aligned)",
                    video_id,
                    len(aligned_segments),
                )
            except Exception as align_exc:
                logger.warning(
                    "Word alignment skipped or failed for %s (%s). Using segment timestamps.",
                    video_id,
                    align_exc,
                )
                aligned_segments = raw_segments
                has_word_alignment = False

        # Build clean structured segments
        structured_segments: List[Dict[str, Any]] = []
        total_word_count = 0

        for idx, seg in enumerate(aligned_segments):
            seg_start = round(float(seg.get("start", 0.0)), 3)
            seg_end = round(float(seg.get("end", seg_start)), 3)
            seg_text = seg.get("text", "").strip()

            words_data: List[Dict[str, Any]] = []
            for w in seg.get("words", []):
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                w_start = (
                    round(float(w["start"]), 3)
                    if "start" in w and w["start"] is not None
                    else None
                )
                w_end = (
                    round(float(w["end"]), 3)
                    if "end" in w and w["end"] is not None
                    else None
                )
                w_score = (
                    round(float(w["score"]), 4)
                    if "score" in w and w["score"] is not None
                    else None
                )
                words_data.append(
                    {
                        "word": word_text,
                        "start": w_start,
                        "end": w_end,
                        "score": w_score,
                    }
                )

            total_word_count += len(words_data)
            structured_segments.append(
                {
                    "segment_id": idx,
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "text": seg_text,
                    "words": words_data,
                }
            )

        transcript_payload = {
            "video_id": video_id,
            "audio_path": str(a_path),
            "language": detected_language,
            "has_word_alignment": has_word_alignment,
            "segment_count": len(structured_segments),
            "word_count": total_word_count,
            "segments": structured_segments,
        }

        # Write formatted JSON to disk
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript_payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "Saved transcript to %s (%d segments, %d words)",
            transcript_file,
            len(structured_segments),
            total_word_count,
        )

        return TranscriptResult(
            video_id=video_id,
            transcript_path=str(transcript_file.resolve()),
            language=detected_language,
            segment_count=len(structured_segments),
            word_count=total_word_count,
            has_word_alignment=has_word_alignment,
            data=transcript_payload,
        )


def transcribe(
    audio_path: Union[str, Path],
    video_id: str,
    output_dir: Union[str, Path],
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    vad_method: str = DEFAULT_VAD_METHOD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TranscriptResult:
    """Convenience functional wrapper around LocalTranscriber."""
    transcriber = LocalTranscriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        vad_method=vad_method,
        batch_size=batch_size,
    )
    return transcriber.transcribe_audio(
        audio_path=audio_path, video_id=video_id, output_dir=output_dir
    )
