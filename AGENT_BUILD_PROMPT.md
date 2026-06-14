# Master Build Prompt — Blackout Buddy
_Paste this into a coding agent (Claude Code, Codex, Cursor, OpenCode). It builds the project end-to-end._

---

## ROLE
You are a senior Python/ML engineer. Build **Blackout Buddy**, a fully offline emergency advisor for the Build Small Hackathon 2026 (Backyard AI / off-grid track). It runs a quantized 4B model via **llama.cpp** (no PyTorch at runtime, no cloud, CPU-only) behind a custom-themed **Gradio 5.x** UI, grounded on a curated verified first-aid knowledge base. Work in `D:\Project\Hugging_face_app\Blackout_buddy`.

## ABSOLUTE NON-NEGOTIABLES
1. **No network calls at runtime.** Network is allowed ONCE — to download models/assets at first boot, then cache. The app MUST work with the network cable pulled. Audit for stray `requests`/HTTP at inference time.
2. **Never crash to a blank/error for the user.** If the model fails to load, fall back to a "Safe Field-Guide mode" that serves verified KB cards with a banner.
3. **Ground the model.** Do NOT let it free-generate medical advice. Retrieve verified KB snippets and have the model rewrite/personalize them. Keep a prominent AI-guidance disclaimer.
4. **Verify model repos exist on HF before coding against them.** If a GGUF is missing, pick a real community GGUF or convert one; document the choice. Fallback primary model is allowed.
5. **CPU-only, streaming, pre-warmed.** Cap tokens, low temperature for factual output.

## STEP 0 — Verify & pin
- Confirm on HF that the chosen GGUFs exist (primary 4B + multilingual fallback). Record exact `repo_id`/`filename` in `MODELS.md`. If unavailable, choose a real alternative ≤4B Q4 and note it.

## STEP 1 — Scaffold
Create:
```
app.py · llm.py · router.py · kb.py · agent.py · voice.py · trace.py · emergency_kb.py
data/emergency_kb.json · data/index.npy (optional)
finetune/modal_train.py · finetune/build_dataset.py · finetune/dataset.jsonl · finetune/README.md
requirements.txt · packages.txt · .env.example · README.md · MODELS.md
```
`requirements.txt`: `llama-cpp-python`, `gradio>=5`, `huggingface_hub`, `langdetect`, `rank-bm25` (+ optional `faster-whisper`/whisper.cpp bindings, `piper-tts`). Pin versions. Prefer prebuilt llama-cpp-python wheels (Windows-friendly).

## STEP 2 — LLM layer (`llm.py`)
- Singleton llama.cpp model, downloaded via `hf_hub_download`, cached to `/data` if writable.
- `ask_stream(prompt, *, model="primary", max_tokens=350, temperature=0.3, grammar=None) -> Iterator[str]` using llama.cpp streaming.
- Pre-warm at import (load + 1 token). `n_threads = os.cpu_count()`, `n_ctx` 4096 (2048 in low-power).
- Load LoRA adapter via `lora_path` if present, else base model.
- System prompt: calm, numbered, imperative, <300 words, assume all external services unavailable, "stabilize and reach trained help if possible."

## STEP 3 — Knowledge base + retrieval (`kb.py`, `data/emergency_kb.json`)
- Author 30–60 verified emergency cards (First Aid, Water, Fire, Disaster, Medical, Navigation) as original paraphrases of authoritative guidance; include a `source` field per card.
- `card(scenario_key) -> str` returns instant verified steps for the 10 quick scenarios.
- `search(query, k=3) -> list[Snippet]` retrieval. Use sentence-transformers MiniLM if available; **default to `rank-bm25`** (pure-python, dependency-light, bulletproof offline).
- Grounded prompt builder: system + retrieved snippets + user question.

## STEP 4 — Router (`router.py`)
- `langdetect` for language. EN/HI → primary (Nemotron); other languages → multilingual fallback (tiny-aya). Manual override from UI dropdown wins.
- Light intent detection (emergency type) to bias retrieval.

## STEP 5 — Gradio UI (`app.py`)
- `gr.Blocks` with custom dark "night-vision emergency" theme via `gr.themes.Base` override + injected CSS (palette: bg `#0B0B0F`, signal `#FF8A00`, critical `#FF3B30`, safe `#28C76F`, ink `#F5E9D8`; fonts Inter + JetBrains Mono; base 18px, line-height 1.8; 56px+ targets).
- Header: title `🔦 Blackout Buddy` + green `● OFFLINE-READY` status chip + Low-Power & Flashlight toggles.
- **Tabs:** Quick Help · Talk to Buddy · Field Guide.
  - **Quick Help:** left = 2-col grid of 10 emoji scenario buttons; right = language dropdown (Auto default), multiline input + mic, giant `⚡ GET HELP` primary button, streaming answer panel (terminal style), "🔊 Read aloud", disclaimer footer. Scenario click renders instant KB card immediately, then LLM enrichment streams.
  - **Talk to Buddy (Agent):** chat triage; Buddy asks 3–5 clarifying questions (big Yes/No/Not-sure buttons), then a highlighted "ACTION PLAN". Log session via `trace.py`.
  - **Field Guide:** searchable accordion of verified cards by category with source attribution.
- Wire streaming generators (`yield`); `demo.queue(default_concurrency_limit=1)`. Mobile CSS media queries. Spinner text for >2s.

## STEP 6 — Triage agent (`agent.py`)
- State machine: gather → ask next question → when enough info, produce grounded guidance.
- Constrain structured steps with a **GBNF grammar** so the small model emits valid `{"action":"ask"|"answer", ...}` JSON.
- Runs entirely via llama.cpp. Append each session to `traces/*.jsonl`.

## STEP 7 — Voice (`voice.py`, optional & lazy)
- `transcribe(audio)` via whisper.cpp/faster-whisper tiny.
- `synthesize(text, lang)` via Piper → wav for `gr.Audio(autoplay=True)`.
- Lazy-import; if unavailable, hide voice controls gracefully — never block core app.

## STEP 8 — Traces & dataset (`trace.py`)
- Append-only JSONL logger; `publish(repo_id)` uploads to a HF dataset (Open Trace badge). Strip PII.

## STEP 9 — Fine-tune (`finetune/`)
- `dataset.jsonl`: 200–300 emergency Q&A pairs (CPR, choking, burns, bleeding, fractures, shock, water purification, fire, earthquake/flood/tornado/hurricane, stroke FAST, allergic reaction, diabetic emergency, snakebite, heatstroke, hypothermia, no-GPS navigation).
- `modal_train.py`: Modal GPU LoRA job via `peft` (r=16, ~3 epochs), persistent Modal Volume checkpoints, and direct Hub upload. Training must run on Modal, never locally or in the Space. `finetune/README.md` documents reproduction.

## STEP 10 — Optimization & resilience
- Pre-warm, KV/context tuning, quant ladder (Q3_K/Q4_K_M/Q8 selectable), low-power path.
- Safe Field-Guide fallback on model failure. CPU-only latency test. RAM-budget check.

## STEP 11 — Docs, deploy, demo
- `README.md` with the exact frontmatter (title Blackout Buddy, emoji 🔦, colorFrom orange, colorTo red, sdk gradio, sdk_version "5.0", app_file app.py, tags incl. sponsors + hackathon + off-grid + fine-tuned + tiny-titan + off-brand) followed by full docs, screenshots, badges table, install (Space + air-gapped local), usage, safety/disclaimer, license (Apache-2.0; verify content/model license compatibility), contributors.
- Deploy to HF Space (CPU Basic first; bump tier only if needed — stay zero-runtime-network).
- Produce a 90–120s demo plan: unplug the network → tap a card → speak "someone is choking" → triage agent → ask in another language.

## DEFINITION OF DONE
- Runs locally and on a Space with the network disconnected.
- 10 scenarios (instant cards + streaming enrichment), grounded RAG, multilingual routing, triage agent with traces, optional voice, custom theme, safe fallback.
- LoRA + trace dataset published to HF; README + frontmatter + screenshots done.
- All claimed badges backed by live artifacts. No runtime network calls.

## BUILD ORDER (each step independently demo-able — always have something to submit)
core advisor → KB grounding + instant cards → multilingual routing → LoRA → custom theme/UI → voice → triage agent + traces → optimize → docs/deploy/demo.

Ask me to clarify only if a model repo cannot be found or a hard dependency won't install on this platform. Otherwise, proceed autonomously and report progress per step.
