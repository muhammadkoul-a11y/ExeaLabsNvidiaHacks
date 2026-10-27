import json
import os
from pathlib import Path
import pytest

from arrival.sim.policy import VLMPlanner, rule_plan
from arrival.tasks.instructions import generate_instructions


def test_rule_plan_parsing():
    plan = rule_plan("Pick up the mug and put it in the sink", ["mug_1", "bowl_1"], ["sink_1", "counter_a"])
    assert plan["family"] == "counter_to_sink"
    assert plan["object"] == "mug_1"
    assert plan["target"] == "sink_1"

    drawer_plan = rule_plan("Slide open the drawer", ["mug_1"], ["counter_a"])
    assert drawer_plan["family"] == "open_drawer"

    door_plan = rule_plan("Open the cabinet door", [], ["counter_a"])
    assert door_plan["family"] == "open_door"


def test_vlm_planner_cache_hit(tmp_path):
    cache_file = tmp_path / "test_cache.json"
    cache_file.write_text(json.dumps({
        "grab the mug and drop in sink::['mug_1']::['sink_1']": {
            "family": "counter_to_sink",
            "object": "mug_1",
            "target": "sink_1",
        }
    }))

    planner = VLMPlanner(log=lambda *_: None, cache_path=cache_file)
    plan = planner.plan("grab the mug and drop in sink", ["mug_1"], ["sink_1"])
    assert planner.source == "cache"
    assert plan == {"family": "counter_to_sink", "object": "mug_1", "target": "sink_1"}


def test_vlm_planner_falls_back_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)
    cache_file = tmp_path / "empty_cache.json"

    planner = VLMPlanner(log=lambda *_: None, cache_path=cache_file)
    plan = planner.plan("Pull out the top drawer.", ["mug_1"], ["counter_a"])
    assert planner.source == "rule_based"
    assert plan["family"] == "open_drawer"


def test_vlm_planner_caches_successful_call(monkeypatch, tmp_path):
    cache_file = tmp_path / "save_cache.json"

    def mock_complete(messages, role="super", **params):
        return '{"family": "open_door", "object": null, "target": null}'

    monkeypatch.setattr("arrival.models.token_factory.complete", mock_complete)
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", "test-key-fake")

    planner = VLMPlanner(log=lambda *_: None, cache_path=cache_file)
    plan = planner.plan("Open the pantry door please", ["mug_1"], ["counter_a"])

    assert planner.source == "token_factory"
    assert plan["family"] == "open_door"
    assert cache_file.is_file()

    # Second call should be a cache hit
    plan2 = planner.plan("Open the pantry door please", ["mug_1"], ["counter_a"])
    assert planner.source == "cache"
    assert plan2 == plan


def test_generate_instructions_train_eval_split(tmp_path):
    cache_file = tmp_path / "instr_cache.json"
    res = generate_instructions(
        family="counter_to_sink",
        object_name="mug",
        eval_ratio=0.25,
        cache_path=cache_file,
    )

    assert "train" in res and "eval" in res
    assert len(res["train"]) >= 1
    assert len(res["eval"]) >= 1
    # Train and eval sets must be disjoint
    assert set(res["train"]).isdisjoint(set(res["eval"]))


def test_sim_collect_with_vlm_planner(tmp_path, monkeypatch):
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)  # no network; fallback is not cached
    mujoco = pytest.importorskip("mujoco")
    from arrival.scan.bundle import load_room
    from arrival.sim.collect import collect
    from arrival.twin.compile import compile_twin

    twin = compile_twin(load_room("hero_kitchen", "scans"))
    # Run 1 episode with planner="vlm"
    res = collect(
        twin=twin,
        families=["counter_to_sink"],
        episodes=1,
        out_dir=tmp_path,
        people=0,
        res=32,
        planner_name="vlm",
        use_generated_instructions=True,
        log=lambda *_: None,
    )

    report = res["report_data"]
    assert report["planner"] == "vlm"
    assert report["attempts"] == 1


def test_instruction_fallback_not_cached_and_train_nonempty(tmp_path, monkeypatch):
    import arrival.models.token_factory as tf

    cache_file = tmp_path / "c.json"

    def down(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(tf, "complete", down)
    res = generate_instructions("counter_to_sink", "mug", cache_path=cache_file)
    assert res["source"] == "template_fallback" and not cache_file.exists()

    monkeypatch.setattr(tf, "complete", lambda **k: '["Put the mug in the sink."]')
    res = generate_instructions("counter_to_sink", "mug", cache_path=cache_file)
    assert res["source"] == "token_factory"
    assert res["train"] == ["Put the mug in the sink."] and res["eval"] == []

    monkeypatch.setattr(tf, "complete", lambda **k: "no list here")
    res = generate_instructions("counter_to_sink", "mug", n=3, cache_path=tmp_path / "d.json")
    assert res["source"] == "template_fallback" and len(res["train"]) + len(res["eval"]) == 3 and res["train"]


def _planner_with_reply(monkeypatch, tmp_path, reply):
    monkeypatch.setattr("arrival.models.token_factory.complete", lambda *a, **k: reply)
    return VLMPlanner(log=lambda *_: None, cache_path=tmp_path / "c.json")


def test_planner_rejects_unknown_family_and_does_not_cache(monkeypatch, tmp_path):
    planner = _planner_with_reply(monkeypatch, tmp_path, 'Sure {"family": "fold_laundry", "object": null, "target": null}')
    plan = planner.plan("Open the cabinet door", ["mug_1"], ["counter_a"])
    assert planner.source == "rule_based" and plan["family"] == "open_door"
    assert not (tmp_path / "c.json").exists()


def test_planner_rejects_unknown_object_and_non_dict(monkeypatch, tmp_path):
    planner = _planner_with_reply(monkeypatch, tmp_path, '{"family": "counter_to_sink", "object": "ghost", "target": null}')
    planner.plan("Put the mug in the sink", ["mug_1"], ["sink_1"])
    assert planner.source == "rule_based"
    planner = _planner_with_reply(monkeypatch, tmp_path, "[1, 2]")
    planner.plan("Put the mug in the sink", ["mug_1"], ["sink_1"])
    assert planner.source == "rule_based"


def test_planner_moves_a_fixture_named_as_object_to_target(monkeypatch, tmp_path):
    # Nemotron answers "open the fridge" with the fridge as its object; that was refused and fell back to the parser
    planner = _planner_with_reply(monkeypatch, tmp_path, '{"family": "open_fridge", "object": "fridge_1", "target": null}')
    plan = planner.plan("Let me check if we're out of milk.", ["mug_1"], ["fridge_1", "sink_1"])
    assert planner.source == "token_factory"
    assert plan == {"family": "open_fridge", "object": None, "target": "fridge_1"}


def test_planner_calls_at_temperature_zero(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("arrival.models.token_factory.complete",
                        lambda *a, **k: seen.update(k) or '{"family": "open_door", "object": null, "target": null}')
    VLMPlanner(log=lambda *_: None, cache_path=tmp_path / "c.json").plan("Open the door", [], [])
    assert seen["temperature"] == 0


def test_planner_takes_first_json_object_of_a_chatty_reply(monkeypatch, tmp_path):
    reply = 'Plan: {"family": "open_door", "object": null, "target": null} (note {not json})'
    planner = _planner_with_reply(monkeypatch, tmp_path, reply)
    assert planner.plan("Open the door", [], [])["family"] == "open_door"
    assert planner.source == "token_factory"


def test_planner_stops_calling_a_dead_endpoint(monkeypatch, tmp_path):
    calls = []

    def dead(*a, **k):
        calls.append(1)
        raise TimeoutError("timed out")

    monkeypatch.setattr("arrival.models.token_factory.complete", dead)
    planner = VLMPlanner(log=lambda *_: None, cache_path=tmp_path / "c.json")
    planner.plan("Open the door", [], [])
    plan = planner.plan("Close the drawer", [], [])
    assert len(calls) == 1 and plan["family"] == "close_drawer" and planner.source == "rule_based"


def test_research_cache_key_is_stable_across_processes():
    import subprocess
    import sys

    code = ("from arrival.sim.policy import VLMPlanner;import tempfile;"
            "print(VLMPlanner(cache_path=tempfile.mkdtemp()+'/c.json')._cache_key('a',[],[],'ctx'))")
    keys = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(i)}).stdout for i in (1, 2)}
    assert len(keys) == 1


def test_default_cache_is_under_out_not_in_the_package(tmp_path, monkeypatch):
    # a planner cache in the package answered the hero kitchen's template wordings, so `sim --planner vlm`
    # with a TOKEN_FACTORY_API_KEY never called Token Factory for them; the default now lives in out/
    from arrival.tasks.instructions import FALLBACK_TEMPLATES

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("arrival.models.token_factory.complete",
                        lambda *a, **k: calls.append(1) or '{"family": "counter_to_sink", "object": "mug_1", "target": "sink_1"}')
    planner = VLMPlanner(log=lambda *_: None)
    objects, fixtures = ["bowl_1", "mug_1"], ["counter_a", "counter_b", "fridge_1", "sink_1", "stove_1"]
    wordings = [t.format(name="mug") for t in FALLBACK_TEMPLATES["counter_to_sink"]]
    for w in wordings:
        planner.plan(w, objects, fixtures)
        assert planner.source == "token_factory"
    assert len(calls) == len(wordings)
    assert planner.cache_path.resolve() == (tmp_path / "out" / "cache" / "nemotron_cache.json").resolve()
