"""Tests for agent-card parsing."""

import pytest
from ovos_a2a_solver.client import AgentCard, AgentSkill


MINIMAL_CARD = {
    "name": "TestAgent",
    "description": "A test A2A agent",
    "url": "https://agent.example.com",
    "version": "1.0",
}

FULL_CARD = {
    "name": "FullAgent",
    "description": "Full-featured agent",
    "url": "https://full.example.com",
    "version": "2.1",
    "capabilities": {"streaming": True},
    "skills": [
        {
            "id": "search",
            "name": "Web Search",
            "description": "Search the web",
            "tags": ["web", "search"],
            "examples": ["What is the capital of France?"],
        }
    ],
}


def test_minimal_card_parse():
    card = AgentCard.from_dict(MINIMAL_CARD)
    assert card.name == "TestAgent"
    assert card.description == "A test A2A agent"
    assert card.url == "https://agent.example.com"
    assert card.version == "1.0"
    assert card.skills == []
    assert card.streaming is False


def test_full_card_parse():
    card = AgentCard.from_dict(FULL_CARD)
    assert card.name == "FullAgent"
    assert card.streaming is True
    assert len(card.skills) == 1
    skill = card.skills[0]
    assert isinstance(skill, AgentSkill)
    assert skill.id == "search"
    assert skill.name == "Web Search"
    assert "web" in skill.tags
    assert len(skill.examples) == 1


def test_card_stores_raw():
    card = AgentCard.from_dict(FULL_CARD)
    assert card.raw == FULL_CARD


def test_skill_from_dict_defaults():
    skill = AgentSkill.from_dict({"id": "x", "name": "X"})
    assert skill.description == ""
    assert skill.tags == []
    assert skill.examples == []


def test_card_missing_capabilities_defaults_no_streaming():
    card = AgentCard.from_dict({**MINIMAL_CARD, "capabilities": {}})
    assert card.streaming is False


def test_card_version_default():
    d = dict(MINIMAL_CARD)
    del d["version"]
    card = AgentCard.from_dict(d)
    assert card.version == "1.0"
