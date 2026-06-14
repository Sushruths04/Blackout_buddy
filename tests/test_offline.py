from pathlib import Path

import llm
from trace import _scrub


def test_model_download_is_opt_in(monkeypatch):
    monkeypatch.delenv("BLACKOUT_ALLOW_MODEL_DOWNLOAD", raising=False)
    assert llm._allow_download() is False


def test_application_has_no_http_client_imports():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("import requests", "import httpx", "import aiohttp", "urllib.request")
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(item in source for item in forbidden), path.name


def test_trace_scrubs_basic_contact_details():
    scrubbed = _scrub("Email me at person@example.com or call +49 123 456 7890")
    assert "person@example.com" not in scrubbed
    assert "+49 123 456 7890" not in scrubbed
