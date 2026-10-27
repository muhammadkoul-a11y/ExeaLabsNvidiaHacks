import os
import json
from pathlib import Path
import pytest

from arrival.models.tavily import TavilyNotConfigured, search as tavily_search
from arrival.tasks.tavily_research import TavilyResearchAgent


def test_tavily_not_configured_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyNotConfigured):
        tavily_search("test query")


def test_tavily_research_agent_cache_hit(tmp_path):
    cache_file = tmp_path / "test_tavily_cache.json"
    cache_file.write_text(json.dumps({
        "test appliance query": {
            "source": "tavily_api",
            "answer": "Test answer for appliance latch and clearance.",
            "results": [{"title": "Test Guide", "url": "https://example.com/test", "content": "..."}]
        }
    }))

    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=cache_file)
    res = agent.search("test appliance query")
    assert agent.last_source == "cache"
    assert "Test answer" in res["answer"]
    assert len(res["results"]) == 1


def test_tavily_research_agent_fallback_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    empty_cache = tmp_path / "empty.json"

    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=empty_cache)
    info = agent.research_appliance("dishwasher")

    assert info["fixture_type"] == "dishwasher"
    assert info["canonical_type"] == "dishwasher"
    assert len(info["safety_guidelines"]) >= 1
    assert len(info["operating_steps"]) >= 1
    assert "counterbalanced" in info["summary"] or "hinge" in info["summary"]


def test_tavily_research_agent_mock_live_search(monkeypatch, tmp_path):
    cache_file = tmp_path / "save_cache.json"

    def mock_tavily_search(query, search_depth="basic", include_answer=True):
        return {
            "answer": "Mock live Tavily answer for microwave button.",
            "results": [{"title": "Microwave Manual", "url": "https://example.com/mw", "content": "specs"}]
        }

    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search", mock_tavily_search)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-dummy-key")

    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=cache_file)
    info = agent.research_appliance("microwave")

    assert agent.last_source == "tavily_api"
    assert "Mock live Tavily answer" in info["summary"]
    assert "https://example.com/mw" in info["sources"]
    assert cache_file.is_file()

    # Second query should hit cache
    agent2 = TavilyResearchAgent(log=lambda *_: None, cache_path=cache_file)
    info2 = agent2.research_appliance("microwave")
    assert agent2.last_source == "cache"
    assert info2["summary"] == info["summary"]


def test_enrich_twin_with_tavily_research(tmp_path):
    agent = TavilyResearchAgent(log=lambda *_: None)
    twin = {
        "room_id": "test_room",
        "fixtures": [
            {"id": "c1", "type": "counter"},
            {"id": "dw1", "type": "dishwasher"},
            {"id": "st1", "type": "gas_range"},
        ],
        "objects": []
    }

    enriched = agent.enrich_twin(twin)
    assert "tavily_research" in enriched
    research = enriched["tavily_research"]
    assert "dishwasher" in research
    assert "gas_range" in research
    assert len(research["dishwasher"]["safety_guidelines"]) >= 1


def test_cli_research_command(tmp_path):
    import argparse
    from arrival.cli import cmd_research

    args = argparse.Namespace(
        room="hero_kitchen",
        query=None,
        out=str(tmp_path),
        scans="scans",
    )
    code = cmd_research(args)
    assert code == 0
    report_file = tmp_path / "research" / "hero_kitchen_research.json"
    assert report_file.is_file()
    data = json.loads(report_file.read_text())
    assert "sink" in data or "counter" in data or "stove" in data


def test_enrich_twin_skips_non_appliances_and_negative_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = TavilyResearchAgent(log=lambda *_: None, cache_path=tmp_path / "c.json")
    twin = {"fixtures": [{"id": "b", "type": "box"}, {"id": "c", "type": "counter"},
                         {"id": "h", "type": "hinge_cabinet"}, {"id": "s", "type": "sink"}]}
    research = agent.enrich_twin(twin)["tavily_research"]
    assert set(research) == {"sink"}

    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise RuntimeError("network down")

    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search", boom)
    agent.research_appliance("fridge")
    agent.research_appliance("fridge")
    assert len(calls) == 1


def test_load_dotenv_reads_every_file(monkeypatch, tmp_path):
    from arrival.models import tavily, token_factory

    (tmp_path / "a.env").write_text("UNRELATED=1\n")
    (tmp_path / "b.env").write_text("TAVILY_KEY_TEST=real\n")
    monkeypatch.delenv("TAVILY_KEY_TEST", raising=False)
    for mod in (tavily, token_factory):
        monkeypatch.delenv("TAVILY_KEY_TEST", raising=False)
        mod.load_dotenv([tmp_path / "a.env", tmp_path / "b.env"])
        assert os.environ["TAVILY_KEY_TEST"] == "real"
    monkeypatch.delenv("TAVILY_KEY_TEST", raising=False)
    os.environ.pop("UNRELATED", None)


def test_default_cache_is_under_out_not_in_the_package(tmp_path, monkeypatch):
    # a cache in the package answered the standard appliance queries, so a run with a TAVILY_API_KEY never
    # called Tavily for them; the default now lives in the run's out/ folder
    import arrival

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("arrival.tasks.tavily_research.tavily_search",
                        lambda q, **k: calls.append(q) or {"answer": "live", "results": []})
    agent = TavilyResearchAgent(log=lambda *_: None)
    for fixture in ("dishwasher", "stove", "fridge", "microwave", "sink"):
        assert agent.research_appliance(fixture)["source"] == "tavily_api"
    assert len(calls) == 5
    assert (tmp_path / "out" / "cache" / "tavily_cache.json").is_file()
    assert not list((Path(arrival.__file__).parent / "models").glob("*cache*.json"))
