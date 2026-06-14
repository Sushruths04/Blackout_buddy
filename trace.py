from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_DIR = Path(__file__).resolve().parent / "traces"
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)")


def _scrub(value: str) -> str:
    value = EMAIL_RE.sub("[email removed]", value)
    return PHONE_RE.sub("[phone removed]", value)


def log_trace(
    turns: list[dict[str, str]],
    *,
    route: dict[str, Any] | None = None,
    outcome: str = "completed",
    session_id: str | None = None,
) -> str:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = session_id or str(uuid.uuid4())
    record = {
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "route": route or {},
        "turns": [
            {"role": turn["role"], "content": _scrub(turn["content"])[:2000]}
            for turn in turns
        ],
    }
    path = TRACE_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return trace_id


def publish(repo_id: str) -> None:
    from huggingface_hub import HfApi

    if not repo_id:
        raise ValueError("A dataset repo_id is required.")
    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(TRACE_DIR),
        path_in_repo="data",
    )
