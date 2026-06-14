---
title: Blackout Buddy
emoji: 🔦
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
tags:
  - hackathon
  - build-small
  - backyard-ai
  - nvidia
  - cohere
  - llama-cpp
  - modal
  - off-grid
  - offline
  - emergency
  - fine-tuned
  - tiny-titan
  - off-brand
---

# Blackout Buddy

Blackout Buddy is an offline emergency field guide with optional local-LLM
enrichment. Reviewed, source-attributed cards render immediately; a local GGUF
model can add a short explanation using only retrieved card content.

The core experience works without a model and without a network connection.

## Why it exists

Emergency information is most valuable when connectivity is least reliable.
Blackout Buddy prioritizes:

- instant reviewed guidance before generation;
- no cloud inference or runtime API dependency;
- graceful fallback when model loading fails;
- large controls and a low-cognitive-load dark interface;
- local, scrubbed traces for the guided triage flow.

## Safety boundary

This project provides general preparedness and first-aid information. It is not a
diagnostic device and does not replace trained care. Reach emergency services,
poison control, or another trained responder whenever any route is available.

Knowledge-base cards are original paraphrases with links to authoritative sources.
They require ongoing clinical and regional review before real-world deployment.

## Architecture

```text
Gradio UI
  -> offline retrieval over data/emergency_kb.json
  -> immediate reviewed card
  -> optional llama.cpp enrichment using retrieved text only
  -> local PII-scrubbed JSONL trace

Training:
  reviewed KB -> Modal GPU LoRA job -> Hugging Face model repo
```

No production request is sent to Modal. Modal is used only for one-off training.

## Run locally

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`.

Without a local GGUF, the app starts in Safe Field-Guide mode. That is a supported
operating mode, not an error.

## Add local models

Verified model files are listed in [MODELS.md](MODELS.md).

```powershell
$env:BLACKOUT_PRIMARY_MODEL_PATH="D:\models\NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
$env:BLACKOUT_MULTILINGUAL_MODEL_PATH="D:\models\tiny-aya-global-q4_k_m.gguf"
$env:BLACKOUT_LORA_PATH="D:\models\blackout-buddy-lora.gguf"
python app.py
```

The app checks explicit paths and the existing Hugging Face cache. It does not
download implicitly. To permit a one-time startup download:

```powershell
$env:BLACKOUT_ALLOW_MODEL_DOWNLOAD="1"
python app.py
```

After the files are cached, set the flag back to `0` and disconnect the network.

## Modal-only training

```powershell
python -m pip install modal==1.5.0
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_your_write_token
modal run finetune/modal_train.py --output-repo your-name/blackout-buddy-nemotron-lora
```

See [finetune/README.md](finetune/README.md) for resource and persistence details.
The published PEFT adapter must be converted to llama.cpp's GGUF LoRA format before
setting `BLACKOUT_LORA_PATH`; details are in [MODELS.md](MODELS.md).

## Tests

```powershell
pytest -q
python -m compileall -q .
```

## Hackathon badge evidence

| Badge | Evidence |
|---|---|
| Off the Grid | No cloud inference; cards and local GGUF runtime work disconnected |
| Well-Tuned | Modal LoRA pipeline publishes a project adapter to Hugging Face |
| Off-Brand | Custom emergency dashboard CSS and interaction design |
| Llama Champion | Optional GGUF models run through `llama-cpp-python` |
| Sharing is Caring | PII-scrubbed triage traces can be published as a Hub dataset |
| Field Notes | Publish the project report and link it here before submission |
| Tiny Titan | NVIDIA Nemotron 3 Nano is 3.97B parameters |

Do not claim badges whose linked Hub artifacts have not been published.

## Deployment note

A free CPU Space may not have enough RAM or storage for a 2.84 GB GGUF plus the
Gradio process. The field-guide build remains useful on CPU Basic; model-backed
enrichment should be validated on the chosen Space hardware before the demo.

## License

Application code is Apache-2.0. Third-party models and source material retain
their own terms. In particular, Tiny Aya Global GGUF is CC BY-NC 4.0. Review
[MODELS.md](MODELS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
