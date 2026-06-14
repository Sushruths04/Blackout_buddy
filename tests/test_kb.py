from emergency_kb import SCENARIOS, load_cards
from kb import card, format_card, search


def test_knowledge_base_has_reviewed_cards():
    cards = load_cards()
    assert len(cards) >= 25
    assert len({item["id"] for item in cards}) == len(cards)
    assert all(item["source_url"].startswith("https://") for item in cards)
    assert all(item["steps"] for item in cards)


def test_all_quick_scenarios_resolve():
    for card_id in SCENARIOS.values():
        assert card(card_id).id == card_id


def test_bleeding_retrieval_and_formatting():
    result = search("There is blood everywhere from a deep cut", k=1)
    assert result
    assert result[0].id == "severe_bleeding"
    rendered = format_card(result[0])
    assert "ACT" in rendered
    assert "Source:" in rendered
