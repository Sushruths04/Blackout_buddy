from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PRIMARY_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
PRIMARY_FILE = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
MULTILINGUAL_REPO = "CohereLabs/tiny-aya-global-GGUF"
MULTILINGUAL_FILE = "tiny-aya-global-q4_k_m.gguf"

SYSTEM_PROMPT = """You are Blackout Buddy, an offline emergency field-guide assistant.
Use only the supplied source material. Never invent a medical fact, dosage, diagnosis,
or procedure. Give calm, short, numbered actions. Preserve every important warning.
State uncertainty plainly. Encourage reaching trained help whenever any route is
available. Keep the answer under 220 words."""


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    filename: str
    path_env: str


SPECS = {
    "primary": ModelSpec(PRIMARY_REPO, PRIMARY_FILE, "BLACKOUT_PRIMARY_MODEL_PATH"),
    "multilingual": ModelSpec(
        MULTILINGUAL_REPO,
        MULTILINGUAL_FILE,
        "BLACKOUT_MULTILINGUAL_MODEL_PATH",
    ),
}

_models: dict[str, object] = {}
_errors: dict[str, str] = {}
_lock = threading.Lock()


def _allow_download() -> bool:
    return os.getenv("BLACKOUT_ALLOW_MODEL_DOWNLOAD", "0") == "1"


def _resolve_model_path(spec: ModelSpec) -> Path | None:
    explicit = os.getenv(spec.path_env)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
        _errors[spec.path_env] = f"Configured file does not exist: {path}"

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(spec.repo_id, spec.filename)
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)
    except Exception:
        pass

    if not _allow_download():
        return None

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=spec.repo_id, filename=spec.filename))


def get_model(model: str = "primary", *, low_power: bool = False):
    cache_key = f"{model}:{'low' if low_power else 'normal'}"
    if cache_key in _models:
        return _models[cache_key]
    if cache_key in _errors:
        return None

    with _lock:
        if cache_key in _models:
            return _models[cache_key]
        try:
            spec = SPECS[model]
            model_path = _resolve_model_path(spec)
            if model_path is None:
                _errors[cache_key] = (
                    "No local GGUF found. Safe Field-Guide mode is active."
                )
                return None
            from llama_cpp import Llama

            lora_path = os.getenv("BLACKOUT_LORA_PATH") if model == "primary" else None
            lora_file = Path(lora_path).expanduser() if lora_path else None
            lora_options = (
                {"lora_path": str(lora_file)}
                if lora_file is not None and lora_file.is_file()
                else {}
            )
            instance = Llama(
                model_path=str(model_path),
                n_ctx=2048 if low_power else 4096,
                n_threads=max(1, os.cpu_count() or 1),
                n_batch=128 if low_power else 256,
                verbose=False,
                **lora_options,
            )
            _models[cache_key] = instance
            return instance
        except Exception as exc:
            _errors[cache_key] = f"{type(exc).__name__}: {exc}"
            return None


def model_status(model: str = "primary", *, low_power: bool = False) -> str:
    cache_key = f"{model}:{'low' if low_power else 'normal'}"
    if cache_key in _models:
        return "ready"
    if cache_key in _errors:
        return _errors[cache_key]
    spec = SPECS[model]
    path = _resolve_model_path(spec)
    return "local model available" if path else "Safe Field-Guide mode"


def ask_stream(
    question: str,
    context: str,
    *,
    model: str = "primary",
    language: str = "en",
    low_power: bool = False,
    max_tokens: int | None = None,
) -> Iterator[str]:
    instance = get_model(model, low_power=low_power)
    if instance is None:
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Reply in language code '{language}'.\n\n"
                f"SOURCE MATERIAL:\n{context}\n\n"
                f"USER SITUATION:\n{question}"
            ),
        },
    ]
    stream = instance.create_chat_completion(
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=max_tokens or (160 if low_power else 240),
        stream=True,
    )
    for event in stream:
        token = event["choices"][0].get("delta", {}).get("content", "")
        if token:
            yield token
