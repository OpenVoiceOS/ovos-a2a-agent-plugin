"""AgentCard parsing for A2A 1.0 and 0.3 cards."""
from ovos_a2a_solver.client import AgentCard, AgentSkill

CARD_1_0 = {
    "name": "FullAgent",
    "description": "Full-featured agent",
    "supportedInterfaces": [
        {"url": "https://full.example.com/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
        {"url": "https://full.example.com/rest", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
    ],
    "version": "2.1",
    "capabilities": {"streaming": True},
    "skills": [{"id": "search", "name": "Web Search", "description": "Search the web",
                "tags": ["web", "search"], "examples": ["What is the capital of France?"]}],
}

CARD_0_3 = {"name": "OldAgent", "description": "0.3 card", "url": "https://old.example.com/",
            "protocolVersion": "0.3", "version": "1.0"}


def test_1_0_card_takes_the_url_of_the_first_interface():
    card = AgentCard.from_dict(CARD_1_0)
    assert card.url == "https://full.example.com/a2a"
    assert card.version == "2.1"
    assert card.streaming is True
    assert card.raw == CARD_1_0


def test_1_0_card_skills():
    (skill,) = AgentCard.from_dict(CARD_1_0).skills
    assert skill == AgentSkill(id="search", name="Web Search", description="Search the web",
                               tags=["web", "search"], examples=["What is the capital of France?"])


def test_0_3_card_takes_the_top_level_url():
    card = AgentCard.from_dict(CARD_0_3)
    assert card.url == "https://old.example.com/"
    assert card.streaming is False
    assert card.skills == []


def test_missing_fields_default_to_empty():
    card = AgentCard.from_dict({})
    assert (card.name, card.description, card.url, card.version) == ("", "", "", "1.0")


def test_skill_defaults():
    skill = AgentSkill.from_dict({"id": "x", "name": "X"})
    assert (skill.description, skill.tags, skill.examples) == ("", [], [])
