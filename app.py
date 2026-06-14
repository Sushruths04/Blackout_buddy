from __future__ import annotations

import html
from typing import Iterator

import gradio as gr

from agent import triage_step
from emergency_kb import SCENARIOS
from kb import card, field_guide, format_card, format_context, search
from llm import ask_stream, model_status
from router import LANGUAGE_CODES, route
from trace import log_trace

CSS = """
:root {
  --bb-bg: #09090d;
  --bb-panel: #151219;
  --bb-panel-2: #1d171d;
  --bb-ink: #f5e9d8;
  --bb-muted: #b3a593;
  --bb-signal: #ff8a00;
  --bb-critical: #ff3b30;
  --bb-safe: #28c76f;
}
body, .gradio-container {
  background:
    radial-gradient(circle at 80% 0%, rgba(255, 59, 48, .13), transparent 32rem),
    radial-gradient(circle at 10% 20%, rgba(255, 138, 0, .08), transparent 28rem),
    var(--bb-bg) !important;
  color: var(--bb-ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}
.gradio-container { max-width: 1380px !important; }
.bb-header {
  border: 1px solid rgba(255, 138, 0, .35);
  border-radius: 18px;
  padding: 20px 24px;
  margin-bottom: 14px;
  background: linear-gradient(135deg, rgba(255,138,0,.09), rgba(255,59,48,.05));
}
.bb-brand { font-size: clamp(2rem, 6vw, 4.8rem); font-weight: 900; letter-spacing: -.05em; }
.bb-tagline { color: var(--bb-muted); font-size: 1.05rem; margin-top: 4px; }
.bb-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 14px;
  padding: 8px 12px;
  border: 1px solid rgba(40,199,111,.5);
  border-radius: 999px;
  color: #7ff0aa;
  background: rgba(40,199,111,.09);
  font-weight: 800;
  letter-spacing: .05em;
}
.bb-panel {
  background: rgba(20,17,24,.93) !important;
  border: 1px solid rgba(255,138,0,.22) !important;
  border-radius: 16px !important;
}
.bb-answer {
  min-height: 440px;
  padding: 12px;
  background: #0d0c10 !important;
  border: 2px solid rgba(255,138,0,.35) !important;
  border-radius: 16px !important;
  font-family: "Cascadia Code", "JetBrains Mono", ui-monospace, monospace !important;
  font-size: 17px !important;
  line-height: 1.72 !important;
}
.bb-answer h2, .bb-answer h3 { color: var(--bb-signal) !important; }
.bb-answer blockquote {
  border-left: 4px solid var(--bb-critical) !important;
  color: #ffb0a9 !important;
}
.scenario button, button.scenario {
  min-height: 68px !important;
  text-align: left !important;
  border: 1px solid rgba(255,138,0,.4) !important;
  background: linear-gradient(145deg, #20161a, #151217) !important;
  color: var(--bb-ink) !important;
  font-weight: 800 !important;
}
.scenario button:hover, button.scenario:hover {
  border-color: var(--bb-signal) !important;
  transform: translateY(-1px);
}
#get-help button {
  min-height: 74px !important;
  font-size: 1.25rem !important;
  font-weight: 950 !important;
  letter-spacing: .04em !important;
  background: linear-gradient(135deg, #d9271e, #ff5a20) !important;
  border: none !important;
  box-shadow: 0 12px 34px rgba(255,59,48,.22) !important;
}
.bb-disclaimer {
  border: 1px solid rgba(255,59,48,.35);
  background: rgba(255,59,48,.08);
  color: #ffc6c1;
  padding: 12px 16px;
  border-radius: 12px;
  font-size: .92rem;
}
.bb-footer { color: var(--bb-muted); text-align: center; font-size: .88rem; }
footer { display: none !important; }
@media (max-width: 760px) {
  .gradio-container { padding: 8px !important; }
  .bb-header { padding: 16px; }
  .bb-answer { min-height: 360px; font-size: 16px !important; }
  .scenario button, button.scenario { min-height: 62px !important; }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill="#09090d",
    body_text_color="#f5e9d8",
    block_background_fill="#151219",
    block_border_color="#3b2b27",
    input_background_fill="#0d0c10",
    button_primary_background_fill="#d9271e",
    button_primary_background_fill_hover="#ff3b30",
)

HEADER = """
<div class="bb-header">
  <div class="bb-brand">🔦 Blackout Buddy</div>
  <div class="bb-tagline">Verified emergency guidance that still works when the network does not.</div>
  <div class="bb-chip">● OFFLINE-READY · SAFE FIELD GUIDE ACTIVE</div>
</div>
"""

DISCLAIMER = """
<div class="bb-disclaimer">
  <strong>Safety boundary:</strong> This tool provides general first-aid and preparedness
  information, not a diagnosis. Reach trained emergency or poison-control help whenever
  any route is available. Do not delay urgent care to use this app.
</div>
"""


def _safe_question(question: str) -> str:
    return html.escape((question or "").strip())


def advise(
    question: str, language: str, low_power: bool
) -> Iterator[tuple[str, str]]:
    cleaned = (question or "").strip()
    if not cleaned:
        yield "Describe the emergency or choose a quick scenario.", "Waiting for details"
        return

    route_info = route(cleaned, language)
    matches = search(cleaned, k=3)
    if not matches:
        yield (
            "## I could not match this safely\n\n"
            "1. Make the scene safe.\n"
            "2. Check responsiveness, normal breathing, and severe bleeding.\n"
            "3. Use the Field Guide or describe the most visible symptom or hazard.\n"
            "4. Reach trained help by any available route.",
            "No verified card match",
        )
        return

    verified = format_card(matches[0])
    status = model_status(route_info.model, low_power=low_power)
    yield verified, f"Verified card ready · {status}"

    generated = ""
    for token in ask_stream(
        cleaned,
        format_context(matches),
        model=route_info.model,
        language=route_info.language,
        low_power=low_power,
    ):
        generated += token
        yield (
            verified + "\n\n---\n\n## Grounded offline explanation\n\n" + generated,
            f"Local {route_info.model} model · streaming",
        )

    log_trace(
        [
            {"role": "user", "content": cleaned},
            {
                "role": "assistant",
                "content": generated or f"Served verified card: {matches[0].id}",
            },
        ],
        route={
            "language": route_info.language,
            "model": route_info.model,
            "intent": route_info.intent,
            "cards": [item.id for item in matches],
        },
        outcome="model-grounded" if generated else "field-guide fallback",
    )


def choose_scenario(card_id: str) -> tuple[str, str, str]:
    snippet = card(card_id)
    prompt = f"I need immediate guidance for: {snippet.title}."
    return prompt, format_card(snippet), "Verified card ready instantly"


def filter_guide(query: str) -> str:
    return field_guide(query or "")


def triage_chat(
    message: str, history: list[dict] | None, state: dict | None
) -> tuple[list[dict], dict, str]:
    history = list(history or [])
    cleaned = (message or "").strip()
    if not cleaned:
        return history, state or {}, ""
    reply, next_state = triage_step(cleaned, state)
    history.extend(
        [
            {"role": "user", "content": cleaned},
            {"role": "assistant", "content": reply},
        ]
    )
    return history, next_state, ""


def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Blackout Buddy",
        theme=THEME,
        css=CSS,
        fill_height=True,
    ) as demo:
        gr.HTML(HEADER)
        with gr.Row():
            low_power = gr.Checkbox(
                value=False,
                label="Low-Power Mode",
                info="Shorter context and answers for battery-critical use.",
            )
            gr.Markdown(
                "**No implicit downloads.** Local GGUFs are used only when already cached or configured.",
                elem_classes=["bb-panel"],
            )

        with gr.Tabs():
            with gr.Tab("Quick Help"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=280):
                        gr.Markdown("## Quick scenarios")
                        scenario_buttons: list[tuple[gr.Button, str]] = []
                        for label, card_id in SCENARIOS.items():
                            button = gr.Button(
                                label,
                                elem_classes=["scenario"],
                                size="lg",
                            )
                            scenario_buttons.append((button, card_id))

                    with gr.Column(scale=7, min_width=360):
                        language = gr.Dropdown(
                            choices=list(LANGUAGE_CODES),
                            value="Auto",
                            label="Response language",
                        )
                        question = gr.Textbox(
                            label="Describe the emergency",
                            placeholder=(
                                "Example: An adult is choking and cannot speak or cough."
                            ),
                            lines=3,
                            max_lines=5,
                            autofocus=True,
                        )
                        ask_button = gr.Button(
                            "⚡ GET VERIFIED HELP",
                            variant="primary",
                            elem_id="get-help",
                        )
                        status = gr.Markdown("Ready · no network required")
                        answer = gr.Markdown(
                            "Choose a scenario or describe what is happening.",
                            elem_classes=["bb-answer"],
                        )

                        ask_event = ask_button.click(
                            advise,
                            inputs=[question, language, low_power],
                            outputs=[answer, status],
                            show_progress="minimal",
                            concurrency_limit=1,
                        )
                        question.submit(
                            advise,
                            inputs=[question, language, low_power],
                            outputs=[answer, status],
                            show_progress="minimal",
                            concurrency_limit=1,
                        )
                        for button, card_id in scenario_buttons:
                            button.click(
                                lambda selected=card_id: choose_scenario(selected),
                                outputs=[question, answer, status],
                                show_progress="hidden",
                                queue=False,
                            )

            with gr.Tab("Guided Triage"):
                gr.Markdown(
                    "## Calm, structured triage\n"
                    "Describe the event. Buddy asks four short questions and returns only reviewed field-guide cards."
                )
                triage_state = gr.State({})
                triage_history = gr.Chatbot(
                    type="messages",
                    height=540,
                    placeholder="Start with one sentence about what happened.",
                    elem_classes=["bb-panel"],
                )
                triage_input = gr.Textbox(
                    label="Your answer",
                    placeholder="Type a short answer...",
                )
                with gr.Row():
                    triage_send = gr.Button("Continue", variant="primary")
                    yes_button = gr.Button("Yes")
                    no_button = gr.Button("No")
                    unsure_button = gr.Button("Not sure")
                    clear_button = gr.ClearButton(
                        [triage_history, triage_input, triage_state]
                    )
                triage_send.click(
                    triage_chat,
                    inputs=[triage_input, triage_history, triage_state],
                    outputs=[triage_history, triage_state, triage_input],
                )
                triage_input.submit(
                    triage_chat,
                    inputs=[triage_input, triage_history, triage_state],
                    outputs=[triage_history, triage_state, triage_input],
                )
                for choice_button, choice in (
                    (yes_button, "Yes"),
                    (no_button, "No"),
                    (unsure_button, "Not sure"),
                ):
                    choice_button.click(
                        lambda selected=choice: selected,
                        outputs=triage_input,
                        queue=False,
                    ).then(
                        triage_chat,
                        inputs=[triage_input, triage_history, triage_state],
                        outputs=[triage_history, triage_state, triage_input],
                    )

            with gr.Tab("Offline Field Guide"):
                gr.Markdown(
                    "## Reviewed cards\nSearch by symptom, hazard, or action. Every card includes its source."
                )
                guide_query = gr.Textbox(
                    label="Search the guide",
                    placeholder="bleeding, unsafe water, fire, cold, lost...",
                )
                guide_output = gr.Markdown(
                    value=field_guide(),
                    elem_classes=["bb-answer"],
                )
                guide_query.input(
                    filter_guide,
                    inputs=guide_query,
                    outputs=guide_output,
                    show_progress="hidden",
                )

        gr.HTML(DISCLAIMER)
        gr.Markdown(
            "Blackout Buddy stores optional, PII-scrubbed traces locally. "
            "It makes no inference-time HTTP requests.",
            elem_classes=["bb-footer"],
        )

    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name="0.0.0.0",
        show_error=True,
    )
