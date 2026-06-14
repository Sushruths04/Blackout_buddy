from __future__ import annotations


def transcribe(audio_path: str) -> str:
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Offline speech recognition is optional and not installed."
        ) from exc

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=1)
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize(text: str, lang: str) -> str:
    raise RuntimeError(
        "Piper voice output is not configured. Text guidance remains available."
    )
