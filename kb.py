from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from emergency_kb import cards_by_id, load_cards

TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Snippet:
    id: str
    title: str
    category: str
    urgency: str
    steps: tuple[str, ...]
    avoid: tuple[str, ...]
    source_name: str
    source_url: str
    score: float = 0.0

    @classmethod
    def from_card(cls, card: dict, score: float = 0.0) -> "Snippet":
        return cls(
            id=card["id"],
            title=card["title"],
            category=card["category"],
            urgency=card["urgency"],
            steps=tuple(card["steps"]),
            avoid=tuple(card.get("avoid", [])),
            source_name=card["source_name"],
            source_url=card["source_url"],
            score=score,
        )


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _document(card: dict) -> str:
    return " ".join(
        [
            card["id"].replace("_", " "),
            card["title"],
            card["category"],
            " ".join(card.get("keywords", [])),
            " ".join(card["steps"]),
            " ".join(card.get("avoid", [])),
        ]
    )


class OfflineRetriever:
    def __init__(self, cards: Iterable[dict]):
        self.cards = list(cards)
        self.documents = [_tokens(_document(card)) for card in self.cards]
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))
        self.average_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
        )

    def _score(self, query_tokens: list[str], index: int) -> float:
        document = self.documents[index]
        counts = Counter(document)
        score = 0.0
        k1, b = 1.5, 0.75
        for token in query_tokens:
            frequency = counts[token]
            if not frequency:
                continue
            containing = self.document_frequency[token]
            inverse_frequency = math.log(
                1 + (len(self.documents) - containing + 0.5) / (containing + 0.5)
            )
            length_norm = 1 - b + b * len(document) / self.average_length
            score += inverse_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_norm)
            )
        return score

    def search(self, query: str, k: int = 3) -> list[Snippet]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        ranked = sorted(
            (
                (self._score(query_tokens, index), card)
                for index, card in enumerate(self.cards)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        matches = [item for item in ranked if item[0] > 0][:k]
        return [Snippet.from_card(card, score) for score, card in matches]


@lru_cache(maxsize=1)
def retriever() -> OfflineRetriever:
    return OfflineRetriever(load_cards())


def search(query: str, k: int = 3) -> list[Snippet]:
    return retriever().search(query, k=k)


def card(card_id: str) -> Snippet:
    try:
        return Snippet.from_card(cards_by_id()[card_id])
    except KeyError as exc:
        raise KeyError(f"Unknown emergency card: {card_id}") from exc


def format_card(snippet: Snippet, *, include_source: bool = True) -> str:
    urgency = {
        "critical": "🔴 ACT NOW",
        "high": "🟠 URGENT",
        "medium": "🟡 TAKE CARE",
    }.get(snippet.urgency, "GUIDANCE")
    lines = [f"## {urgency}: {snippet.title}", "", "### ACT"]
    lines.extend(
        f"{number}. {step}" for number, step in enumerate(snippet.steps, start=1)
    )
    if snippet.avoid:
        lines.extend(["", "### AVOID"])
        lines.extend(f"- {warning}" for warning in snippet.avoid)
    if include_source:
        lines.extend(
            [
                "",
                f"**Source:** [{snippet.source_name}]({snippet.source_url})",
                "",
                "> This offline field guide is not a diagnosis. Reach trained help whenever any route is available.",
            ]
        )
    return "\n".join(lines)


def format_context(snippets: list[Snippet]) -> str:
    sections = []
    for snippet in snippets:
        sections.append(
            "\n".join(
                [
                    f"TITLE: {snippet.title}",
                    "VERIFIED STEPS:",
                    *[f"- {step}" for step in snippet.steps],
                    "DO NOT:",
                    *[f"- {warning}" for warning in snippet.avoid],
                    f"SOURCE: {snippet.source_name} ({snippet.source_url})",
                ]
            )
        )
    return "\n\n".join(sections)


def field_guide(query: str = "") -> str:
    snippets = (
        search(query, k=8)
        if query.strip()
        else [Snippet.from_card(card) for card in load_cards()]
    )
    if not snippets:
        return "No matching field-guide card. Try a symptom, hazard, or action."
    return "\n\n---\n\n".join(format_card(item) for item in snippets)
