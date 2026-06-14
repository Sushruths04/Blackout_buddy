from router import route


def test_manual_language_routes_to_multilingual_model():
    selected = route("Someone is bleeding", "German")
    assert selected.language == "de"
    assert selected.model == "multilingual"


def test_english_routes_to_primary_model():
    selected = route("Someone is choking and cannot speak", "English")
    assert selected.language == "en"
    assert selected.model == "primary"
    assert selected.intent == "breathing"
