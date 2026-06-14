from __future__ import annotations

from dataclasses import dataclass

LANGUAGE_CODES = {
    "Auto": None,
    "English": "en",
    "Hindi": "hi",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Ukrainian": "uk",
}


@dataclass(frozen=True)
class Route:
    language: str
    model: str
    intent: str


def detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "en"


def detect_intent(text: str) -> str:
    lowered = text.lower()
    intent_terms = {
        "bleeding": ("blood", "bleed", "wound"),
        "breathing": ("breath", "chok", "unconscious", "cpr"),
        "water": ("water", "drink", "boil", "purif"),
        "fire": ("fire", "smoke", "burn"),
        "weather": ("flood", "storm", "tornado", "hurricane", "lightning"),
        "temperature": ("heat", "cold", "hypother", "frost"),
    }
    for intent, terms in intent_terms.items():
        if any(term in lowered for term in terms):
            return intent
    return "general"


def route(text: str, manual_language: str = "Auto") -> Route:
    language = LANGUAGE_CODES.get(manual_language) or detect_language(text)
    model = "primary" if language in {"en", "hi"} else "multilingual"
    return Route(language=language, model=model, intent=detect_intent(text))
