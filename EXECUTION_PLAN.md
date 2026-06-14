# Blackout Buddy — Master Execution Plan
**Build Small Hackathon 2026 · Backyard AI Track · Offline Emergency Advisor**
_Senior architect expansion of `A2_blackout_buddy_prompt.md` — v1.0 (2026-06-14)_

---

## 0. The One-Sentence Thesis
> **An emergency AI that works precisely when everything else has failed — no internet, no power grid, no cell signal — by running a fine-tuned 4B model on llama.cpp over a curated, verified first-aid knowledge base, with voice in/out so a panicking person never has to type.**

The single most important strategic upgrade over the original spec: **do not let the small model free-generate medical advice.** Ground it in a curated offline knowledge base (RAG) and use it to *rewrite and personalize verified content*. This simultaneously (a) fixes the #1 risk of small models — hallucinated, dangerous advice; (b) massively increases technical depth and demo credibility; and (c) guarantees the demo never dies, because verified cards render instantly even if the model fails to load.

---

## 1. Project Analysis

### Executive Summary
Blackout Buddy is a fully offline, CPU-only emergency first-responder advisor. It answers first-aid, water, fire, disaster, medical-triage, and navigation questions in multiple languages, with zero network dependency at runtime. It runs a quantized NVIDIA Nemotron Nano 4B (with a Cohere tiny-aya multilingual fallback) via llama.cpp, fine-tuned with a LoRA on emergency data. The product's killer property is also its category: **offline-first is the feature, not a checkbox.** The demo's hero moment is literally unplugging the network and watching it keep working.

### Problem Statement
When disaster strikes — earthquakes, floods, hurricanes, grid failures, war zones, remote wilderness — the internet and cell networks are the *first* things to fail, exactly when life-saving information is most needed. Existing AI assistants (ChatGPT, Gemini, Siri) are useless offline. Printed first-aid manuals are static, monolingual, and not searchable under stress. There is a real gap: **interactive, conversational, multilingual emergency guidance that survives infrastructure collapse.**

### Target Users
1. **Disaster-zone residents** (post-earthquake/flood/hurricane) — primary.
2. **Wilderness/backcountry travelers** — hikers, campers, off-roaders with no signal.
3. **Rural & low-connectivity communities** — chronic intermittent infrastructure.
4. **Migrants / disaster tourists / refugees** — need guidance in a non-local language.
5. **Preppers & emergency-response volunteers** — want a resilient offline tool.
6. **Edge/embedded deployers** — NGOs putting it on Raspberry Pis, old phones, solar kits.

### Market Need
- 3+ billion people experience periodic connectivity loss; natural disasters displace tens of millions yearly.
- Red Cross / WHO first-aid content is public-domain and authoritative but locked in PDFs.
- On-device AI is now feasible: a 4B Q4 model fits in ~2.5 GB RAM and runs on a CPU.
- No mainstream consumer product positions itself as "the AI that works when the grid is down."

### Competitive Advantage
| Vs. | Their weakness | Our edge |
|---|---|---|
| ChatGPT/Gemini apps | Dead offline | Runs with the cable pulled |
| Printed manuals | Static, monolingual, not interactive | Conversational, multilingual, triage |
| First-aid apps (offline DBs) | Keyword lookup, no reasoning | LLM reasoning grounded on verified KB |
| Other hackathon entries | "Offline" as a bullet point | Offline *is* the entire value prop + 6 badges |

### Innovation Score (self-assessed)
- **Originality: 8.5/10** — offline-survival framing + grounded small-model RAG + voice is uncommon.
- **Technical depth: 9/10** — quantization, LoRA, llama.cpp, offline RAG, GBNF-constrained triage, offline TTS/STT.
- **Usefulness: 9.5/10** — genuinely life-relevant, not a toy.
- **Demo impact: 9.5/10** — the unplug-the-cable moment is unforgettable.
- **Feasibility on small compute: 9/10** — everything chosen runs on CPU.

### Hackathon Winning Potential
**High.** The project naturally stacks 6 badges + Tiny Titan, aligns perfectly with the Backyard AI / off-grid theme, has a visceral demo, and a real mission. The risks are model-availability (verify exact GGUF repos) and CPU latency (mitigated by streaming + instant KB cards). With the RAG + voice upgrades, it moves from "good badge-farming entry" to "memorable, defensible winner."

---

## 2. Product Vision

### Long-Term Vision
The default emergency layer for any disconnected device — preloaded on relief-org tablets, solar-powered community kiosks, and consumer phones' "emergency mode." A trusted, auditable, multilingual survival companion that never depends on a server.

### Future Roadmap (post-hackathon)
- **v1 (hackathon):** Text + voice advisor, grounded RAG, triage agent, multilingual fallback, LoRA.
- **v2:** Vision lite — photograph a wound/rash, small VLM describes severity (offline).
- **v3:** Bluetooth/LoRa **mesh sharing** — one device with the model relays guidance to nearby phones with none.
- **v4:** Region packs — localized hazards (monsoon, wildfire, avalanche) + local emergency protocols.
- **v5:** Native mobile (llama.cpp on Android/iOS), "Emergency Mode" OS integration, watch companion.

### Scalability Opportunities
Horizontal by *content packs* (languages, regions, hazard types) rather than by servers — because there are no servers. Distribution scales via app stores, NGO deployments, and SD-card images.

### Edge-Device Deployment
- **Raspberry Pi 4/5 (4–8 GB):** runs Q4 4B at a few tok/s — viable.
- **Android via Termux / MLC / llama.cpp:** flagship phones run it comfortably.
- **Old laptops / solar kits / Meshtastic nodes:** the target backyard scenario.
- Package as a single folder with bundled GGUF for true air-gapped install.

### Small-Model Optimization Strategy
1. **Quantization:** Q4_K_M default; offer Q8 for accuracy on capable hardware, Q3_K for tiny RAM.
2. **LoRA fine-tune** on emergency Q&A → tighter, more reliable domain answers; merge or load adapter.
3. **Grounded RAG** → model rewrites retrieved verified text, drastically cutting hallucination and letting a *small* model punch above its weight.
4. **GBNF grammar-constrained decoding** for the triage agent → forces valid structured outputs (no rambling).
5. **Prompt economy:** short system prompt, capped `max_tokens`, streaming, KV-cache reuse, pre-warmed model at boot.
6. **Low-Power Mode:** reduced `n_ctx`, fewer threads, shorter answers for battery-critical situations.

---

## 3. Technical Architecture

### System Architecture (logical)
```
┌──────────────────────────────────────────────────────────────┐
│                     GRADIO UI (Blocks)                         │
│  Quick-Scenario grid · Free-text box · Voice in · Voice out   │
│  Triage "Agent Mode" · Language selector · Low-Power toggle   │
└───────────────┬──────────────────────────────────────────────┘
                │ python calls (in-process, no HTTP to cloud)
        ┌───────▼─────────┐     ┌──────────────────────────────┐
        │   ROUTER /       │────▶│  KB + RETRIEVER (offline)    │
        │   ORCHESTRATOR   │     │  curated cards + embeddings/ │
        │ (intent, lang,   │     │  BM25 over verified content  │
        │  agent loop)     │◀────│  → top-k grounded context    │
        └───────┬──────────┘     └──────────────────────────────┘
                │ grounded prompt
        ┌───────▼──────────┐
        │  LLM LAYER        │  Nemotron-Nano-4B (Q4) + LoRA  ──┐
        │ (llama-cpp-python)│  Cohere tiny-aya fallback (mlt) │
        │  streaming, GBNF  │                                 │
        └───────┬──────────┘                                  │
                │ tokens                                       │
   ┌────────────▼───────┐  ┌──────────────┐  ┌────────────────▼─┐
   │ Whisper.cpp (STT)  │  │ Piper (TTS)  │  │ Trace logger →   │
   │  voice → text      │  │ text → audio │  │ JSONL → HF dataset│
   └────────────────────┘  └──────────────┘  └──────────────────┘

   NETWORK USED ONLY ONCE: model/asset download at first boot → cached.
   At runtime: ZERO outbound calls. Pull the cable → still works.
```

### Frontend Architecture
- **Gradio 5.x `gr.Blocks`** with custom dark-emergency CSS theme (`gr.themes.Base` override + injected CSS).
- Three primary views via `gr.Tabs`: **Quick Help**, **Talk to Buddy (Agent)**, **Field Guide (offline cards)**.
- Glanceable, high-contrast, large tap targets (panic-grade UX, see §4).
- Streaming answer rendering via generator functions (`yield`).
- Optional voice: `gr.Audio(sources=["microphone"])` in, `gr.Audio(autoplay=True)` out.

### Backend Architecture (all in-process Python, no external services)
- `app.py` — Gradio composition + event wiring + streaming handlers.
- `llm.py` — llama-cpp-python singleton, streaming `ask()`/`ask_stream()`, GBNF grammar loader, LoRA attach.
- `router.py` — language detect (`langdetect`/`fasttext-lite`), model routing (Nemotron vs tiny-aya), intent classification.
- `kb.py` — curated emergency cards (dict/JSON) + retriever (sentence-transformers MiniLM **or** pure-python BM25 fallback to stay dependency-light).
- `agent.py` — triage state machine: asks N clarifying questions, then emits grounded guidance; GBNF-constrained.
- `voice.py` — whisper.cpp (STT) + Piper (TTS) wrappers, lazy-loaded, optional.
- `trace.py` — append-only JSONL session logger + HF upload helper (Open Trace badge).
- `emergency_kb.py` — quick scenario map (already drafted in spec).
- `finetune/` — LoRA training on Modal only; never invoked by the offline Space.

### AI Model Architecture
- **Primary:** Nemotron Nano 4B Instruct, GGUF Q4_K_M, + emergency LoRA (load adapter via llama.cpp `lora_path`, or ship a merged GGUF).
- **Multilingual fallback:** Cohere tiny-aya-global GGUF Q4_K_M for non-EN/HI inputs.
- **Retriever embeddings:** `all-MiniLM-L6-v2` (90 MB) via llama.cpp embedding mode or onnx; **fallback:** rank-bm25 (no model, pure python) to keep it bulletproof offline.
- **STT:** whisper.cpp `tiny`/`base` (~75–150 MB).
- **TTS:** Piper (~20–60 MB per voice).
- **Decoding controls:** `temperature 0.3` (factual), `max_tokens 350`, stop tokens, GBNF grammar for triage JSON.

### Data Flow (typical request)
```
user input (text or voice)
   └─ if voice → whisper.cpp → text
        └─ langdetect → choose model (Nemotron | tiny-aya)
             └─ retriever.search(query) → top-k verified KB snippets
                  └─ build grounded prompt (system + KB context + question)
                       └─ llama.cpp stream tokens → UI (yield)
                            └─ optional Piper TTS → autoplay audio
                                 └─ trace.append(session) → JSONL
```

### API Structure (internal contracts — no public HTTP API)
```python
# llm.py
def ask_stream(prompt: str, *, model="primary", max_tokens=350, grammar=None) -> Iterator[str]
# kb.py
def search(query: str, k: int = 3) -> list[Snippet]         # Snippet(title, text, source)
def card(scenario_key: str) -> str                           # instant verified answer
# router.py
def route(text: str) -> Route                                # Route(lang, model, intent)
# agent.py
def triage_step(history: list[Turn], user_msg: str) -> AgentReply   # ask | answer
# voice.py
def transcribe(audio_path: str) -> str
def synthesize(text: str, lang: str) -> str                  # returns wav path
# trace.py
def log(session_id: str, turns: list[Turn]) -> None
def publish(repo_id: str) -> None
```

### Storage Strategy
- **Models/assets:** HF Hub download → cached to `~/.cache/huggingface` or Space `/data` persistent volume.
- **KB:** versioned JSON/JSONL in repo (`data/emergency_kb.json`) + prebuilt embedding index (`data/index.npy`) committed so no build step is needed at boot.
- **Traces:** local `traces/*.jsonl`, periodically pushed to a HF dataset repo.
- **No database** — flat files only (off-grid principle).

### Deployment Strategy
- **Primary:** Hugging Face Space, `sdk: gradio`, **CPU Basic** tier first (proves off-grid). If GGUF too heavy/slow, **CPU Upgrade** or **A10G** — still off-grid from an *API* standpoint.
- Model download at first startup with visible progress; cache to `/data`.
- `requirements.txt` pinned; system deps (e.g., for whisper.cpp/piper) via `packages.txt` or prebuilt wheels.
- **Air-gapped build** for the demo: bundle GGUFs locally so judges can run with network truly off.

### Performance Optimization Plan
1. **Pre-warm** model at app launch (load + 1 dummy token) so first real query is fast.
2. **Stream tokens** (`yield`) — perceived latency drops dramatically.
3. **Instant KB cards** for the 10 quick scenarios — sub-100ms, no model needed; LLM only enriches.
4. **Cap context & tokens**, reuse KV cache, set `n_threads = os.cpu_count()`.
5. **Lazy-load** voice models only when the voice tab is used.
6. **Low-Power Mode** toggle: `n_ctx 2048`, `max_tokens 200`.
7. **Quant ladder** selectable: Q3_K (tiny) / Q4_K_M (default) / Q8 (accurate).

---

## 4. UI/UX Design Plan — "Panic-Grade" Emergency UX

### Design Philosophy
**"Calm under fire."** A person using this is scared, possibly in the dark, possibly with shaking hands, possibly one-handed while holding a victim. The UI must be: **glanceable, huge-targeted, high-contrast, low-cognitive-load, and reassuring.** Aesthetic = "emergency dashboard meets night-vision terminal": dark background, warm amber/orange signal color, red for critical, monospace for that rugged off-grid feel. Every screen answers one question: *what do I do RIGHT NOW?*

### Core Principles
- **STOP · THINK · ACT** framing on every answer — numbered, imperative, short.
- **One primary action per screen.** The giant `⚡ GET HELP` button.
- **Thumb-reachable**, 56px+ targets, generous spacing for trembling hands.
- **Dark-mode by default** (battery + night usability), optional high-brightness "flashlight" mode.
- **No clutter, no jargon, no walls of text** — chunked steps with whitespace.
- **Always-visible safety footer** (AI guidance disclaimer + "seek trained help if reachable").

### User Journeys
1. **Cold panic (fastest path):** open → tap a Quick Scenario card → instant verified steps appear (KB) → LLM enriches/streams below → optional "read aloud."
2. **Describe-it path:** type or **speak** the emergency → grounded answer streams → tap follow-up.
3. **Guided triage (Agent Mode):** "Talk to Buddy" → it asks 3–5 yes/no questions → tailored guidance → session logged as a trace.
4. **Browse Field Guide:** offline cards for when you just want to read/prepare.
5. **Language switch:** auto-detected; manual override dropdown for mixed-language households.

### Wireframe Descriptions
- **Header:** `🔦 Blackout Buddy` + live **status chip**: `● OFFLINE-READY` (green dot) to reassure "this works disconnected." Optional `⚡ Low-Power` toggle and 🔆 brightness.
- **Quick Help tab (default):** left = 2-column grid of 10 big emoji scenario buttons; right = language selector, large multiline input with mic button, the giant `⚡ GET HELP` button, and a tall streaming answer panel styled as a terminal. Disclaimer footer.
- **Talk to Buddy tab:** chat-style triage. Buddy's questions appear as cards with big Yes/No/"Not sure" buttons; final guidance in a highlighted "ACTION PLAN" block; "🔊 Read aloud" button.
- **Field Guide tab:** searchable accordion of verified cards grouped by category (First Aid · Water · Fire · Disaster · Medical · Navigation), each with source attribution (Red Cross/WHO).

### Dashboard Layout (Quick Help)
```
┌─────────────────────────────────────────────────────────────┐
│ 🔦 BLACKOUT BUDDY            ● OFFLINE-READY   ⚡Low  🔆Bright │
├───────────────┬─────────────────────────────────────────────┤
│ QUICK         │  Language [Auto ▾]                           │
│ SCENARIOS     │  ┌───────────────────────────────────────┐  │
│ 🩹 Bleeding   │  │ Describe your emergency…        🎤 mic │  │
│ 💧 Water bad  │  └───────────────────────────────────────┘  │
│ 🔥 Fire       │            ┌───────────────────┐            │
│ ❤️ Unconscious│            │   ⚡  GET  HELP    │            │
│ 🌊 Flood      │            └───────────────────┘            │
│ 🏥 Stroke     │  ┌──── EMERGENCY GUIDANCE ──────────────┐   │
│ 🌡️ Heatstroke │  │ 1. …                                 │   │
│ 🧊 Hypotherm. │  │ 2. …               (streaming)        │   │
│ 🐍 Snakebite  │  │ 🔊 Read aloud                         │   │
│ ⚡ Outage     │  └──────────────────────────────────────┘   │
│               │  ⚠️ AI guidance — reach trained help if able │
└───────────────┴─────────────────────────────────────────────┘
```

### Color Palette
| Token | Hex | Use |
|---|---|---|
| `bg-base` | `#0B0B0F` | app background (near-black, OLED-friendly) |
| `bg-panel` | `#141118` | cards/panels |
| `signal` | `#FF8A00` | primary amber text/icons |
| `critical` | `#FF3B30` | warnings, primary button, danger |
| `safe` | `#28C76F` | "offline-ready" status, success |
| `muted` | `#9A8C7A` | secondary text |
| `ink` | `#F5E9D8` | high-contrast body text |
Accent gradient `colorFrom: orange → colorTo: red` matches the Space card.

### Typography
- **Display/UI:** `Inter` or `Space Grotesk` (clean, modern, readable under stress).
- **Answer/terminal body:** `JetBrains Mono` / `IBM Plex Mono` — rugged, monospaced, "field equipment" feel, excellent legibility.
- Base size **18px**, answer steps **19–20px**, generous line-height **1.8** for scan-ability.

### Accessibility
- WCAG AA+ contrast (amber on near-black exceeds 7:1).
- **Voice in/out** is itself a major accessibility win (low literacy, low vision, hands busy).
- Large targets, keyboard navigable, ARIA labels on icon buttons.
- Plain-language, short-sentence outputs (constrained by system prompt).
- Reduced-motion respect; no flashing (no seizure risk).
- High-contrast "flashlight" mode for daylight readability.

### Mobile Responsiveness
- Single-column stack on narrow screens; scenario grid collapses to a horizontally scrollable chip row or 2-wide grid.
- Sticky `⚡ GET HELP` button.
- Touch-first, thumb-zone layout. Gradio is responsive by default; reinforce with CSS media queries.

### Demo-Friendly Interactions
- **Streaming** answers (visible "thinking → answering").
- **The unplug moment:** a recorded/live clip toggling network off while it keeps working.
- **Voice demo:** speak "someone is choking" → hear spoken steps back.
- **Instant cards:** tapping a scenario shows verified steps with zero perceptible delay (great on camera).
- **Status chip** turning/staying green `OFFLINE-READY` reinforces the narrative on screen.

---

## 5. Gradio Implementation Plan

### App Structure
```
app.py
 ├─ build_theme()              # gr.themes.Base override + CSS injection
 ├─ Blocks(theme, css, title)
 │   ├─ Header row (title + status chip + toggles)
 │   ├─ gr.Tabs
 │   │   ├─ Tab "Quick Help"      → quick_help_ui()
 │   │   ├─ Tab "Talk to Buddy"   → agent_ui()
 │   │   └─ Tab "Field Guide"     → field_guide_ui()
 │   └─ Footer (disclaimer)
 └─ demo.queue().launch()      # queue() enables streaming + concurrency control
```

### Component Hierarchy (Quick Help)
```
Row
├─ Column(scale=1)  Quick Scenarios
│   └─ Button × 10  (elem_classes=["scenario-btn"])
└─ Column(scale=2)
    ├─ Dropdown(language, value="Auto")
    ├─ Row: Textbox(question) + Audio(microphone)   # voice optional
    ├─ Button("⚡ GET HELP", variant="primary")
    ├─ Markdown/Textbox(answer, streaming)           # the answer panel
    ├─ Button("🔊 Read aloud") + Audio(autoplay)     # TTS optional
    └─ Markdown(disclaimer)
```

### Pages / Navigation
Three tabs (Quick Help / Talk to Buddy / Field Guide). No router needed — Gradio Tabs. Keep nav flat: an emergency app must never make users hunt.

### User Interaction Flow (event wiring)
- `ask_btn.click(handle_ask_stream, [question, lang, low_power], answer)` — **generator** that `yield`s tokens.
- Each scenario `btn.click(lambda p=prompt: p, None, question).then(handle_ask_stream, ...)` — fills box then answers. (Also render the instant KB card immediately for zero-latency feel.)
- `mic.stop_recording(transcribe, mic, question)` → then optionally auto-ask.
- `read_btn.click(synthesize, [answer, lang], tts_audio)`.
- Agent tab: `submit.click(triage_step, [chat_state, msg], [chatbot, chat_state])` looping until guidance, then `trace.log(...)`.

### Model Integration Approach
- Singleton llama.cpp model, **pre-warmed** at import.
- Router picks Nemotron (EN/HI) vs tiny-aya (other langs).
- Retrieve KB → inject as grounded context → stream.
- GBNF grammar only in Agent Mode for structured question/answer steps.
- Graceful degradation: if model load fails → serve verified KB cards + a banner ("Running in safe Field-Guide mode"). **The app must never show a blank error to someone in an emergency.**

### Performance Considerations
- `demo.queue(max_size=…, default_concurrency_limit=1)` — single model, serialize inference.
- Streaming generators; cap tokens; KB cards for instant response.
- Lazy import voice libs; cache TTS for repeated phrases.
- Show a friendly spinner / "Thinking… (offline model, a few seconds)" for >2s.

### Deployment on HF Spaces
1. Repo with `app.py`, `requirements.txt`, `packages.txt` (if native deps), `README.md` (frontmatter from spec).
2. `sdk: gradio`, `sdk_version: "5.x"`, `app_file: app.py`.
3. First boot downloads GGUF + assets (progress shown), cache to `/data`.
4. Start on **CPU Basic**; if latency unacceptable, bump tier — keep zero runtime network calls.
5. Add HF metadata tags (already in spec) for sponsor/badge discoverability.
6. Provide an **air-gapped local run** path in README for judges who literally disconnect.

---

## 6. Development Roadmap (phased, hackathon-paced)

> Effort uses **focused-hours** (one builder + AI agents). Total ≈ 28–40 hrs → a long weekend.

### Phase 1 — Foundation (≈4–6h)
- **Tasks:** scaffold repo & dirs; `requirements.txt`; verify exact GGUF repo IDs on HF; get llama-cpp-python loading a tiny GGUF locally; basic `ask()` returns text; stub Gradio app launches.
- **Dependencies:** llama-cpp-python build wheels available for the platform.
- **Risks:** **model ID/GGUF may not exist** (spec flags this) → resolve a real repo *first*; llama-cpp-python build issues on Windows → use prebuilt wheels.
- **Success:** `python app.py` opens UI; typing a question returns a model answer locally.

### Phase 2 — Core Features (≈6–8h)
- **Tasks:** 10 quick-scenario buttons wired; curated KB cards (verified content) for those 10; instant-card rendering; streaming answers; disclaimer footer; offline guarantee audit (no runtime `requests`).
- **Dependencies:** Phase 1 model loading.
- **Risks:** content accuracy (use reviewed, attributed paraphrases of authoritative sources); CPU latency → streaming + cards.
- **Success:** all 10 scenarios produce correct verified steps instantly + LLM enrichment streams; works with network off locally.

### Phase 3 — AI Integration (≈8–10h)
- **Tasks:** build emergency RAG (KB JSON + retriever, BM25 fallback); grounded prompting; run the LoRA fine-tune on Modal and publish to HF (**Well-Tuned**); multilingual detect + tiny-aya routing (**Cohere sponsor**); Agent Mode triage with GBNF; trace logging + publish dataset (**Sharing is Caring**).
- **Dependencies:** dataset authored; Modal account, GPU credits, and a Hugging Face write-token secret.
- **Risks:** LoRA training time/quality; tiny-aya GGUF availability; grammar bugs. Mitigate by making each feature independently shippable (router/agent/voice are additive).
- **Success:** non-English query routes correctly; triage asks then answers; LoRA + traces live on HF; grounded answers cite verified KB.

### Phase 4 — UI/UX Enhancement (≈4–6h)
- **Tasks:** custom dark-emergency theme + CSS (**Off-Brand**); status chip; tabs; mobile CSS; voice in (whisper.cpp) + voice out (Piper); low-power toggle; flashlight mode.
- **Dependencies:** Phase 2 UI skeleton.
- **Risks:** native deps for whisper/piper on Spaces → keep voice optional & lazy; CSS regressions.
- **Success:** polished, branded, responsive UI; voice round-trip works; looks nothing like default Gradio.

### Phase 5 — Optimization (≈3–4h)
- **Tasks:** pre-warm; KV/context tuning; quant ladder; concurrency limit; graceful-degradation safe mode; CPU-only latency test; battery/low-power path.
- **Dependencies:** features complete.
- **Risks:** memory limits on free tier → pick right quant; cold-start time → cache to /data.
- **Success:** first answer < a few seconds on target CPU; never crashes to blank; runs in RAM budget.

### Phase 6 — Submission Prep (≈3–4h)
- **Tasks:** README (frontmatter + full doc + screenshots); **Field Notes** blog post; demo video (the unplug moment + voice); deploy to Space; verify all 6 badges + Tiny Titan; final offline test by pulling the cable; submission form.
- **Dependencies:** everything above.
- **Risks:** last-minute Space build failures → deploy early, iterate; video time.
- **Success:** live Space, video, blog, dataset, LoRA all linked; checklist (§9) 100% green.

---

## 7. Hackathon Strategy

### Tracks / Badges to Target
Primary: **Backyard AI / Off-the-Grid.** Stack (from spec, with my reinforcement):
| Badge | How we secure it |
|---|---|
| **Off the Grid** | Zero runtime network calls; verified by unplugging. The whole architecture. |
| **Well-Tuned** | LoRA on 200–300 emergency Q&A, published to HF. |
| **Llama Champion** | All inference via llama.cpp (llama-cpp-python). |
| **Off-Brand** | Custom dark "night-vision emergency" theme, not default Gradio. |
| **Open Trace** | Publish triage agent trace dataset to HF. |
| **Field Notes** | Blog: "Why offline AI for emergencies matters." |
| **Tiny Titan** | Nemotron Nano **4B** qualifies (≤ threshold). |
> ⚠️ Verify exact badge names/criteria against the official 2026 rules — treat the spec's list as a strong prior, confirm before claiming.

### How to Maximize Scoring
- **Lead with mission + demo**, not architecture. Judges feel the unplug moment before they read code.
- **Show, don't tell, offline:** live network-off demo = irrefutable proof.
- **Safety credibility:** grounded RAG on Red Cross/WHO + visible source attribution signals seriousness.
- **Breadth of badges** without diluting the core — each badge maps to a real feature, not a hack.
- **Polish:** branded UI + voice = "this is a product," not a notebook.

### What Judges Typically Look For
Originality · technical depth · correct/clever use of small + sponsor models · real-world usefulness · execution quality · demo clarity · reproducibility (published artifacts). Blackout Buddy scores on all.

### Demo Strategy (90–120s)
1. **Hook (10s):** "When disaster hits, the internet dies first. Watch." — pull the ethernet/disable Wi-Fi on camera.
2. **Quick card (15s):** tap 🩹 Bleeding → instant verified steps.
3. **Voice (20s):** *speak* "someone is choking" → hear spoken steps back.
4. **Triage agent (25s):** describe an ambiguous emergency → Buddy asks 3 questions → tailored plan.
5. **Multilingual (15s):** ask in Spanish/Hindi → answers in-language (tiny-aya).
6. **Payoff (15s):** "Still no internet. Runs on a 4B model on a CPU — even a Raspberry Pi. This is the AI that works when nothing else does."

### Presentation Strategy
- Slide 1: the problem, visceral (a photo of a blackout/disaster).
- Slide 2: the thesis line (§0).
- Slide 3: architecture diagram (§3) — emphasize "network used once, then never."
- Slide 4: badges + published artifacts (LoRA, dataset, blog) as proof of depth.
- Slide 5: roadmap → "this is a movement, not a weekend hack."

### Storytelling Approach
Open with a real scenario: *"It's 2am. The earthquake cut the power and the towers. Your neighbor is bleeding. Your phone has no signal — but it has Blackout Buddy."* Make the judge the protagonist who needs it.

### Key Differentiators
1. Offline is the **product category**, not a feature.
2. **Grounded** small-model RAG = safe + technically deep.
3. **Voice** round-trip, fully on-device.
4. Genuinely **runs on edge hardware** (Pi/phone) — show it.
5. **6 badges + Tiny Titan**, each backed by a real capability.

---

## 8. README Plan (structure for the HF Space README.md)

1. **Frontmatter** — exactly as in the spec (title, emoji 🔦, colorFrom orange → colorTo red, sdk gradio, tags). Keep tags for sponsors + badges.
2. **Title + tagline** — "🔦 Blackout Buddy — the AI that works when the grid goes down."
3. **Elevator pitch** — 2–3 sentences (the thesis §0).
4. **The unplug demo GIF** — top of README, the hero moment.
5. **Features** — offline-first, grounded RAG, multilingual, voice in/out, triage agent, edge-ready, 10 instant scenario cards.
6. **How it works / Architecture** — diagram (§3) + "network used only at first download."
7. **Models & sponsors** — Nemotron Nano 4B (+LoRA), Cohere tiny-aya, llama.cpp; links to published LoRA + trace dataset.
8. **Badges** — table mapping each badge to the feature proving it.
9. **Installation** — HF Space usage + local/air-gapped run (`pip install -r requirements.txt`, model download, `python app.py`).
10. **Usage** — quick help, voice, agent mode, language switching, low-power mode.
11. **Reproduce the fine-tune** — link to `finetune/README.md` + command.
12. **Screenshots** — Quick Help, Agent, Field Guide, mobile.
13. **Demo video** — link.
14. **Performance / hardware** — RAM/quant table, "runs on Raspberry Pi."
15. **Safety & disclaimer** — AI guidance, grounded on public-domain Red Cross/WHO, not a substitute for professionals when reachable.
16. **Future work** — roadmap (§2).
17. **License** — e.g., Apache-2.0 (confirm model/content licenses are compatible; cite content sources).
18. **Contributors / acknowledgements** — you + sponsor models + content sources.

---

## 9. Submission Checklist

**Code readiness**
- [ ] App runs locally and on the Space without errors
- [ ] **Zero runtime network calls** (audited; verified by disconnecting)
- [ ] Graceful degradation to KB-only safe mode if model fails
- [ ] Streaming works; concurrency limited; pre-warm on boot
- [ ] All 10 scenarios + agent + multilingual paths tested

**Documentation**
- [ ] README complete with frontmatter, features, architecture, install, usage, screenshots, demo, license, contributors
- [ ] `finetune/README.md` reproduces the LoRA
- [ ] Inline code comments where non-obvious; sources cited for KB content

**UI polish**
- [ ] Custom dark-emergency theme (Off-Brand) — not default Gradio
- [ ] Mobile responsive; large targets; AA+ contrast
- [ ] Status chip, low-power, flashlight, disclaimer all present

**Performance testing**
- [ ] CPU-only latency acceptable (first token < a few seconds)
- [ ] Fits free-tier RAM at chosen quant
- [ ] Voice round-trip works (or cleanly optional if unavailable)

**Model optimization**
- [ ] Correct GGUF repos verified & pinned
- [ ] Q4_K_M default + quant ladder
- [ ] LoRA trained, evaluated for quality, merged/attached

**Hugging Face deployment**
- [ ] Space live, correct SDK/version, tags set
- [ ] LoRA repo published (Well-Tuned)
- [ ] Trace dataset published (Open Trace)
- [ ] All sponsor model links correct

**Demo video**
- [ ] 90–120s, includes the unplug moment + voice + agent + multilingual
- [ ] Captions; clear audio; shows it on CPU/edge

**Presentation assets**
- [ ] Slide deck (problem → thesis → architecture → badges → roadmap)
- [ ] Field Notes blog post published & linked

**Final validation**
- [ ] **Pull the network cable test passes**
- [ ] All 6 badges + Tiny Titan claims are backed by live artifacts
- [ ] Submission form completed before deadline
- [ ] Spare: screenshots, GIFs, repo link, dataset link, LoRA link all collected

---

## 10. Risks & Verification Notes (read first)
1. **Model availability is the top risk.** Confirm these exist on HF *before building*: `nvidia/Nemotron-3-Nano-4B-Instruct-GGUF`, `CohereLabs/tiny-aya-global-GGUF`. If a GGUF is missing, use a community conversion or convert with `llama.cpp/convert`. Fallback primary: tiny-aya Q4_K_M (spec already allows this).
2. **Verify hackathon rules/badge names/Tiny Titan threshold** against the official 2026 page — the spec's badge list is a strong prior, not gospel.
3. **Medical-safety liability:** ground every answer on public-domain Red Cross/WHO content, attribute sources, keep the disclaimer prominent, and prefer "stabilize + reach help if possible" framing.
4. **CPU latency:** mitigate with streaming + instant KB cards + pre-warm; the demo should never feel slow.
5. **Voice native deps on Spaces:** keep whisper.cpp/Piper optional and lazy so a build issue never blocks the core app.
6. **Scope control:** ship in this order — core advisor → KB grounding → multilingual → LoRA → voice → agent/traces. Each step is independently demo-able, so you always have something to submit.
```
The build agent prompt is in AGENT_BUILD_PROMPT.md.
```
