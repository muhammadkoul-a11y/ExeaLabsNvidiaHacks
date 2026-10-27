import argparse

import pytest

from arrival import cli


def _ns(tmp_path, **kw):
    d = dict(room="nope_room", scans="scans", out=str(tmp_path), families="counter_to_sink", episodes=1, people=0,
             policy="scripted", res=32, seed=0, save_preview=False, planner="rule", split="train",
             instructions="template", tavily=False, holdout=False, query=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_sim_missing_room_is_usage_error(tmp_path, capsys):
    pytest.importorskip("mujoco")
    assert cli.cmd_sim(_ns(tmp_path)) == cli.EXIT_USAGE
    assert "no scan for room" in capsys.readouterr().err


def test_sim_missing_room_lists_the_rooms_that_exist(tmp_path, capsys, monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    assert cli.main(["sim", "--room", "hero_kitchn", "--episodes", "1", "--out", str(tmp_path)]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "no scan for room 'hero_kitchn'" in err and "hero_kitchen" in err.split("'hero_kitchn'", 1)[1]


def test_demo_names_the_fixture_folder_for_a_test_room(tmp_path, monkeypatch):
    """rotated_kitchen and u_kitchen_42 live in tests/fixtures/scans; the error only said 'not found under scans'."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.run_demo("rotated_kitchen", out_dir=tmp_path)
    msg = str(e.value)
    assert "room 'rotated_kitchen' not found under" in msg
    assert "hero_kitchen" in msg and "--scans tests/fixtures/scans" in msg


@pytest.mark.parametrize("flag", ["--people", "--seed"])
def test_sim_negative_people_or_seed_is_a_usage_error_and_keeps_the_old_dataset(tmp_path, capsys, monkeypatch, flag):
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    old = tmp_path / "datasets" / "hero_kitchen_sim" / "meta" / "info.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}")
    assert cli.main(["sim", flag, "-1", "--episodes", "1", "--families", "open_drawer",
                     "--out", str(tmp_path)]) == cli.EXIT_USAGE
    assert f"{flag} must be >= 0" in capsys.readouterr().err
    assert old.is_file()


def test_sim_tavily_without_the_vlm_planner_is_a_usage_error(tmp_path, capsys, monkeypatch):
    """Only the VLM planner reads the research: with --planner rule, --tavily made its lookups and changed nothing."""
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/

    def no_lookup(*a, **k):
        raise AssertionError("research ran")

    monkeypatch.setattr("arrival.tasks.tavily_research.TavilyResearchAgent", no_lookup)
    assert cli.main(["sim", "--tavily", "--episodes", "1", "--families", "open_drawer",
                     "--out", str(tmp_path)]) == cli.EXIT_USAGE
    assert "--tavily needs --planner vlm" in capsys.readouterr().err
    assert not (tmp_path / "datasets").exists()


def test_research_missing_room_is_usage_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("arrival.tasks.tavily_research.TavilyResearchAgent", lambda *a, **k: object())
    assert cli.cmd_research(_ns(tmp_path)) == cli.EXIT_USAGE


@pytest.mark.parametrize("cmd", ["sim", "work", "summary"])
def test_empty_families_is_a_usage_error(tmp_path, capsys, monkeypatch, cmd):
    """`--families ,` parsed to no families: sim replaced the old dataset with an empty, invalid one and exited 0."""
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    argv = [cmd, "--families", " , ", "--out", str(tmp_path)]
    argv += {"sim": ["--room", "hero_kitchen"], "work": ["--once"]}.get(cmd, [])
    assert cli.main(argv) == cli.EXIT_USAGE
    assert "--families is empty" in capsys.readouterr().err
    assert not (tmp_path / "datasets").exists()


def test_sim_vla_not_configured_is_usage_error(tmp_path, capsys, monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.delenv("GR00T_POLICY_URL", raising=False)
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    assert cli.cmd_sim(_ns(tmp_path, room="hero_kitchen", policy="vla")) == cli.EXIT_USAGE
    assert "policy vla" in capsys.readouterr().err


def test_fanout_reports_invalid_merged_dataset(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("arrival.jobs.fanout.run_fanout",
                        lambda rec, cfg, out: {"dataset": str(tmp_path / "ds"), "ran": [0], "skipped": []})
    monkeypatch.setattr("arrival.export.validate.validate", lambda d: ["meta/info.json: missing"])
    a = _ns(tmp_path, room="hero_kitchen", backend="local", run_id="r", shards=1)
    assert cli.cmd_fanout(a) == cli.EXIT_INVALID
    assert "INVALID: meta/info.json: missing" in capsys.readouterr().err


def _room(tmp_path):
    import json
    import os
    import shutil

    from conftest import REPO

    scans = tmp_path / "scans"
    shutil.copytree(REPO / "scans" / "hero_kitchen", scans / "hero_kitchen")
    twin = tmp_path / "out" / "twins" / "hero_kitchen" / "twin.json"
    twin.parent.mkdir(parents=True)
    twin.write_text(json.dumps({"cached": True}))
    return scans, twin, scans / "hero_kitchen" / "scan.json", os


def _twin(tmp_path, scans):
    pytest.importorskip("mujoco")
    fams, twin = cli._sim_args(_ns(tmp_path / "out", base="fixed", arm="procedural", room="hero_kitchen", scans=str(scans)))
    return twin


def test_cached_twin_is_used_while_it_is_newer_than_the_scan(tmp_path):
    scans, twin, scan, os = _room(tmp_path)
    os.utime(scan, (1_000_000, 1_000_000))
    assert _twin(tmp_path, scans) == {"cached": True}


def test_cached_twin_is_recompiled_after_the_scan_changes(tmp_path):
    scans, twin, scan, os = _room(tmp_path)
    os.utime(twin, (1_000_000, 1_000_000))
    assert "cached" not in _twin(tmp_path, scans)


def test_sim_exits_1_when_every_task_errors(tmp_path, capsys, monkeypatch):
    """A renderer that cannot start fails every task at build; that run must not exit 0 with an empty dataset."""
    pytest.importorskip("mujoco")
    import arrival.sim.collect as collect

    def broken(*a, **k):
        raise RuntimeError("gladLoadGL error")

    monkeypatch.setattr(collect, "SimEnv", broken)
    monkeypatch.chdir(tmp_path)  # resolve_scans falls back to the repo's scans/
    assert cli.cmd_sim(_ns(tmp_path, room="hero_kitchen")) == cli.EXIT_INVALID
    assert "every task errored" in capsys.readouterr().err

def test_cost_cli_runs_and_reports(tmp_path, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["cost", "--out", str(tmp_path)])
    rc = cli.cmd_cost(args)
    assert rc == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "| source | room | family |" in out
    assert "| dollars | usd | measured | note |" in out

@pytest.mark.parametrize("cmd", ["cmd_sim", "cmd_eval"])
@pytest.mark.parametrize("kw,expected_err", [
    ({"people": -1}, "--people must be >= 0"),
    ({"seed": -1}, "--seed must be >= 0"),
    ({"res": 50}, "--res must be a positive multiple of 16"),
    ({"res": 0}, "--res must be a positive multiple of 16"),
])
def test_sim_eval_numerical_bounds(tmp_path, capsys, cmd, kw, expected_err):
    pytest.importorskip("mujoco")
    ns = _ns(tmp_path, **kw)
    assert getattr(cli, cmd)(ns) == cli.EXIT_USAGE
    assert expected_err in capsys.readouterr().err


@pytest.mark.parametrize("threshold", [-0.5, float("nan"), float("inf"), -1.0])
def test_subset_drawer_bounds(tmp_path, capsys, threshold):
    ns = argparse.Namespace(src=str(tmp_path), out=str(tmp_path / "o"), families="", min_drawer_opening=threshold)
    assert cli.cmd_subset(ns) == cli.EXIT_USAGE
    assert "min_drawer_opening must be a finite, nonnegative distance in metres" in capsys.readouterr().err


@pytest.mark.parametrize("bad_family", ["bad_family", "counter_to_sink,unknown_family"])
def test_subset_unknown_families(tmp_path, capsys, bad_family):
    ns = argparse.Namespace(src=str(tmp_path), out=str(tmp_path / "o"), families=bad_family, min_drawer_opening=None)
    assert cli.cmd_subset(ns) == cli.EXIT_USAGE
    assert "unknown families" in capsys.readouterr().err


def test_merge_shards_refuses_out_matching_input_shard(tmp_path, capsys):
    shard_dir = tmp_path / "shard0"
    shard_dir.mkdir()
    ns = argparse.Namespace(shards=[str(shard_dir)], out=str(shard_dir))
    assert cli.cmd_merge_shards(ns) == cli.EXIT_USAGE
    assert "--out cannot be an input shard directory" in capsys.readouterr().err


@pytest.mark.parametrize("kw,expected", [({"shards": 0}, "shards must be"), ({"run_id": "../x"}, "run_id must be")])
def test_fanout_bad_config_is_usage_error(tmp_path, capsys, kw, expected):
    """FanoutConfig checks its settings in __post_init__; that error is a usage message, not a traceback."""
    ns = _ns(tmp_path, backend="local", **{"run_id": "r", "shards": 1, **kw})
    assert cli.cmd_fanout(ns) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("fanout: ") and expected in err
    assert not (tmp_path / "fanout_store").exists()


@pytest.mark.parametrize("bad_family", ["bogus", "counter_to_sink,open_drwaer"])
def test_fanout_unknown_family_is_usage_error_before_any_shard(tmp_path, capsys, monkeypatch, bad_family):
    """Every shard's sim would reject the name, so fanout used to launch and retry each shard (Nebius jobs too)."""
    pytest.importorskip("mujoco")
    monkeypatch.setattr("arrival.jobs.fanout.run_fanout", lambda *a, **k: pytest.fail("a shard was launched"))
    ns = _ns(tmp_path, backend="local", run_id="r", shards=2, families=bad_family)
    assert cli.cmd_fanout(ns) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("fanout: unknown families") and "open_drawer" in err
    assert not (tmp_path / "fanout_store").exists()


def test_fanout_accepts_all_families(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    seen = []
    def fake_run(rec, cfg, out):
        seen.append(cfg.families)
        return {"dataset": tmp_path, "ran": [0], "skipped": []}
    monkeypatch.setattr("arrival.jobs.fanout.run_fanout", fake_run)
    monkeypatch.setattr("arrival.export.validate.validate", lambda d: [])
    ns = _ns(tmp_path, room="hero_kitchen", backend="local", run_id="r", shards=1, families="all")
    assert cli.cmd_fanout(ns) == cli.EXIT_OK
    assert seen == ["all"]


@pytest.mark.parametrize("command", ["sim", "eval", "work", "summary"])
def test_families_help_lists_every_sim_family(command, capsys):
    from arrival.sim.tasks import SIM_FAMILIES

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--help"])
    out = capsys.readouterr().out
    assert [f for f in SIM_FAMILIES if f not in out] == []
    assert [f.strip() for f in cli.FAMILY_NAMES.split(",")] == list(SIM_FAMILIES)
