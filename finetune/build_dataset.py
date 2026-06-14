from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "emergency_kb.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset.jsonl"

PROMPT_TEMPLATES = [
    "What should I do right now for {title}?",
    "We have no internet and may not reach help quickly. Give safe first steps for {title}.",
    "Give a calm, short action plan for {title}. Include what not to do.",
    "Emergency field guide: {title}. What are the immediate priorities?",
    "Someone nearby may have {title}. Give only reviewed immediate actions and warnings.",
    "Explain the first five safe actions for {title} in plain language.",
    "I am panicking about {title}. Tell me what to do first, then what to avoid.",
]


def format_response(card: dict) -> str:
    steps = "\n".join(
        f"{index}. {step}" for index, step in enumerate(card["steps"], start=1)
    )
    avoid = "\n".join(f"- {item}" for item in card.get("avoid", []))
    return (
        f"ACT NOW\n{steps}\n\nAVOID\n{avoid}\n\n"
        f"Source: {card['source_name']}\n"
        "Reach trained help whenever any route is available."
    )


def build_rows() -> list[dict[str, str]]:
    cards = json.loads(KB_PATH.read_text(encoding="utf-8"))
    rows = []
    for card in cards:
        for template in PROMPT_TEMPLATES:
            rows.append(
                {
                    "instruction": template.format(title=card["title"]),
                    "response": format_response(card),
                    "source_card": card["id"],
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
