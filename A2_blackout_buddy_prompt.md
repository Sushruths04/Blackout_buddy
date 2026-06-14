# A2 — "Blackout Buddy" | Claude Code Build Prompt
## Build Small Hackathon 2026 | Backyard AI Track

---

## Mission
Build a fully offline, no-internet emergency advisor that runs on llama.cpp. When the internet is down, power is out, or you're in a disaster zone, the model is your first-responder guide: first aid, water purification, evacuation, fire, medical triage — in whatever language the user needs. This is the rare app where "offline-first" is the product, not a checkbox.

---

## Models (ONLY sponsor models)

| Role | Model ID | Params | Sponsor |
|---|---|---|---|
| Primary advisor | `nvidia/Nemotron-3-Nano-4B-Instruct` (GGUF) | 4B | NVIDIA |
| Multilingual fallback | `CohereLabs/tiny-aya-global-GGUF` (Q4_K_M) | 3.35B | Cohere |
| Fine-tuned LoRA on emergency data | Custom LoRA on Nemotron-Nano-4B | — | NVIDIA |

**Tiny Titan:** Nemotron-Nano-4B (4B exactly) qualifies. Publish LoRA to HF to claim Well-Tuned badge.

**llama.cpp:** Both models run via llama.cpp — no PyTorch, no cloud, no GPU required. Runs on CPU.

---

## Badge stack (this is the Bonus Quest Champion play)

| Badge | How |
|---|---|
| ✅ Off the Grid | No cloud APIs — llama.cpp only, runs fully local |
| ✅ Well-Tuned | Fine-tune LoRA on emergency/first-aid dataset, publish to HF |
| ✅ Llama Champion | Both models run via llama.cpp runtime |
| ✅ Off-Brand | Custom dark-mode emergency UI (not default Gradio) |
| ✅ Open Trace | Publish agent trace dataset to HF Hub |
| ✅ Field Notes | Write blog post: "why offline AI for emergencies matters" |

**6 out of 6 badges.** This is the Bonus Quest Champion strategy — no other app will stack this many.

---

## Tech stack
- **llama-cpp-python** (Python bindings for llama.cpp) — no PyTorch needed
- **Gradio 5.x** with `gr.Server` for custom UI
- **Modal for training only** — one-off LoRA jobs run on Modal; production inference never calls Modal
- **HF Spaces CPU tier** — if the GGUF model is too large for CPU Space, use A10G Space with llama.cpp still (stays off-grid from API standpoint)
- **LoRA training**: run on Modal with `peft` before deployment; upload trained weights to HF

---

## Directory structure
```
blackout-buddy/
├── app.py                    # Gradio entry point
├── llm.py                    # llama-cpp-python wrapper
├── emergency_kb.py           # Structured emergency knowledge prompts
├── finetune/
│   ├── modal_train.py        # Modal GPU LoRA job; never used by production
│   ├── build_dataset.py      # Deterministic dataset export/inspection
│   ├── dataset.jsonl         # Emergency Q&A pairs (see dataset section)
│   └── README.md             # How to reproduce the fine-tune
├── requirements.txt          # llama-cpp-python, gradio, huggingface_hub
├── .env.example              # HF_TOKEN (for model download only)
└── README.md                 # HF Space frontmatter
```

---

## README.md — EXACT frontmatter
```yaml
---
title: Blackout Buddy
emoji: 🔦
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
tags:
  - hackathon
  - build-small
  - backyard-ai
  - nvidia/Nemotron-3-Nano-4B
  - cohere/tiny-aya-global
  - llama-cpp
  - off-grid
  - offline
  - emergency
  - fine-tuned
  - tiny-titan
  - off-brand
---
```

---

## Fine-tune dataset (emergency_finetune_dataset.jsonl)
Create 200-300 Q&A pairs covering:
- First aid: CPR, choking, burns, bleeding, fractures, shock
- Water: purification methods, finding water, contamination signs
- Fire: evacuation, types of fires, what NOT to do
- Natural disasters: earthquake, flood, tornado, hurricane steps
- Medical: allergic reaction, stroke signs (FAST), diabetic emergency
- Navigation: without GPS, reading terrain

Format:
```json
{"instruction": "Someone is choking and cannot speak. What do I do?", "response": "Perform the Heimlich maneuver immediately: 1. Stand behind them..."}
{"instruction": "How do I purify water if I have no filter or tablets?", "response": "Boiling is most reliable: bring water to a rolling boil for 1 minute (3 minutes above 6500ft)..."}
```

Fine-tune command (run on Modal before submission):
```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_write_token
modal run finetune/modal_train.py \
  --output-repo build-small-hackathon/blackout-buddy-nemotron-lora \
  --epochs 3
```

Modal persists checkpoints in the `blackout-buddy-training` Volume and uploads the
adapter directly to the selected Hugging Face repository.

---

## llm.py — llama.cpp wrapper
```python
from llama_cpp import Llama
import os

_model = None

def get_model():
    global _model
    if _model is None:
        # Download GGUF from HF Hub if not cached
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="nvidia/Nemotron-3-Nano-4B-Instruct-GGUF",  # check exact HF repo
            filename="Nemotron-3-Nano-4B-Instruct-Q4_K_M.gguf"
        )
        _model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )
    return _model

SYSTEM_PROMPT = """You are Blackout Buddy — an offline emergency advisor. 
The user may be in a crisis with no internet. Give clear, numbered, actionable steps.
Keep responses under 300 words. Never suggest calling services as a first step — 
assume all services are unavailable. Be calm, direct, and specific."""

def ask(question: str, lang: str = "en") -> str:
    model = get_model()
    prompt = f"[INST] {question} [/INST]"
    result = model(
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{question} [/INST]",
        max_tokens=400,
        stop=["</s>", "[INST]"],
        echo=False
    )
    return result["choices"][0]["text"].strip()
```

---

## emergency_kb.py — structured quick-access prompts
```python
SCENARIOS = {
    "🩹 Someone is bleeding badly": "How do I stop severe bleeding with no medical supplies?",
    "💧 Water is unsafe": "How do I purify water without equipment or electricity?",
    "🔥 Fire in the building": "Fire emergency evacuation steps — what order?",
    "❤️ Person is unconscious": "Person is unconscious and not breathing. CPR steps.",
    "🌊 Flash flood warning": "Flash flood starting now — immediate action steps.",
    "🏥 Signs of a stroke": "Person may be having a stroke — what to do without ambulance?",
    "🌡️ Heatstroke": "Someone has heatstroke in extreme heat — treatment steps.",
    "🧊 Hypothermia": "Person is severely cold and shaking — hypothermia treatment.",
    "🐍 Snake bite": "Snake bite on leg — treatment steps with no hospital available.",
    "⚡ Power outage (extended)": "Power has been out for days — survival priorities."
}
```

---

## app.py — Gradio UI spec
```python
import gradio as gr
from llm import ask
from emergency_kb import SCENARIOS

# Custom dark emergency CSS
CSS = """
body { background: #1a0000; color: #ff9900; font-family: monospace; font-size: 18px; }
.scenario-btn { background: #2a0000; border: 1px solid #ff4400; color: #ff9900; 
                font-size: 16px; margin: 4px; padding: 10px; border-radius: 6px; }
.answer-box { background: #0a0a0a; border: 2px solid #ff4400; 
              font-size: 18px; line-height: 1.9; padding: 16px; }
.gr-button-primary { background: #cc2200 !important; font-size: 20px; }
"""

with gr.Blocks(css=CSS, title="🔦 Blackout Buddy") as demo:
    gr.Markdown("# 🔦 Blackout Buddy\n**Offline emergency advisor — no internet required.**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Quick Scenarios")
            scenario_btns = []
            for label, prompt in SCENARIOS.items():
                btn = gr.Button(label, elem_classes=["scenario-btn"])
                scenario_btns.append((btn, prompt))

        with gr.Column(scale=2):
            lang = gr.Dropdown(
                choices=["English", "Hindi", "German", "Spanish", "French", "Arabic"],
                value="English", label="Language"
            )
            question = gr.Textbox(
                label="Describe your emergency",
                placeholder="e.g. My child swallowed something and is choking...",
                lines=3
            )
            ask_btn = gr.Button("⚡ GET HELP", variant="primary")
            answer = gr.Textbox(
                label="Emergency guidance",
                lines=12,
                elem_classes=["answer-box"],
                interactive=False
            )
            gr.Markdown("⚠️ *This is AI guidance. In a true emergency, always attempt to reach trained help.*")

    def handle_ask(q, lang_choice):
        if not q.strip():
            return "Please describe your emergency above."
        return ask(q, lang=lang_choice[:2].lower())

    ask_btn.click(handle_ask, inputs=[question, lang], outputs=answer)

    for btn, prompt in scenario_btns:
        btn.click(lambda p=prompt: p, outputs=question).then(
            handle_ask, inputs=[question, lang], outputs=answer
        )

demo.launch()
```

---

## TODO 1 — Multilingual auto-detect + Tiny Aya fallback
Detect the input language automatically (use `langdetect` library — tiny, no model needed). If non-English and non-Hindi, route through `CohereLabs/tiny-aya-global-GGUF` instead of Nemotron for better multilingual coverage. This makes the app genuinely useful across language communities (EU migrants, disaster tourists, etc.) and adds the Cohere sponsor to your stack.

## TODO 2 — Agent trace dataset + offline symptom checker
After the base app works: add an "Agent Mode" where the app asks the user 3-5 clarifying questions before giving guidance ("Is the person conscious? Are they breathing? Any known allergies?"). Log each multi-turn session as a trace and publish to `build-small-hackathon/blackout-buddy-traces` dataset — this earns the Open Trace badge. The agent loop runs entirely via llama.cpp, no cloud.

---

## Non-negotiables
- The app MUST work if you pull the network cable. No `requests` calls at runtime, only at model-download time.
- Model download happens at Space startup (cache in `/data` if persistent storage is on). Show download progress.
- If llama.cpp inference takes >10s on CPU, show a spinner with "Thinking... (offline model, takes a few seconds)"
- Test on CPU only before submitting — judges may run it on a CPU Space
- GGUF model file: if `nvidia/Nemotron-3-Nano-4B-Instruct-GGUF` doesn't exist on HF yet, use community conversion or fall back to `CohereLabs/tiny-aya-global-GGUF:Q4_K_M` as primary
