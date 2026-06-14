from __future__ import annotations

from dataclasses import dataclass, field

from kb import format_card, search
from trace import log_trace

QUESTIONS = [
    "Is the person awake and responding?",
    "Are they breathing normally?",
    "Is there severe bleeding?",
    "What happened, and what is the most dangerous thing you can see right now?",
]


@dataclass
class TriageState:
    answers: list[str] = field(default_factory=list)
    turns: list[dict[str, str]] = field(default_factory=list)
    complete: bool = False

    def as_dict(self) -> dict:
        return {
            "answers": self.answers,
            "turns": self.turns,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: dict | None) -> "TriageState":
        value = value or {}
        return cls(
            answers=list(value.get("answers", [])),
            turns=list(value.get("turns", [])),
            complete=bool(value.get("complete", False)),
        )


def triage_step(message: str, state_value: dict | None) -> tuple[str, dict]:
    state = TriageState.from_dict(state_value)
    cleaned = (message or "").strip()
    if not cleaned:
        return "Describe the emergency in one sentence.", state.as_dict()

    if state.complete:
        state = TriageState()

    state.answers.append(cleaned)
    state.turns.append({"role": "user", "content": cleaned})

    question_index = len(state.answers) - 1
    if question_index < len(QUESTIONS):
        reply = QUESTIONS[question_index]
        state.turns.append({"role": "assistant", "content": reply})
        return reply, state.as_dict()

    combined = " ".join(state.answers)
    matches = search(combined, k=2)
    if matches:
        reply = "## ACTION PLAN\n\n" + "\n\n".join(
            format_card(item) for item in matches
        )
    else:
        reply = (
            "## ACTION PLAN\n\n"
            "1. Make the scene safe.\n"
            "2. Check responsiveness, normal breathing, and severe bleeding.\n"
            "3. Treat the most immediate threat within your training.\n"
            "4. Reach trained help by any available route.\n\n"
            "> I could not match this to a specific offline card."
        )
    state.complete = True
    state.turns.append({"role": "assistant", "content": reply})
    log_trace(state.turns, outcome="field-guide triage")
    return reply, state.as_dict()
