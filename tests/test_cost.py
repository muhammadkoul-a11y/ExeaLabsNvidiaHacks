"""OPS-3a: the cost report from synthetic reports, eval files and job statuses. No sim, no network."""

import json
from pathlib import Path

import pytest

from arrival.eval import cost

PRICING = Path(__file__).resolve().parent.parent / "configs" / "pricing.example.yaml"
BASELINE = {"task": "counter_to_sink", "room": "hero_kitchen", "n_success": 10, "operator_seconds_total": 1500,
            "setup_assumptions": ["one operator", "VR rig owned"], "cost_basis_usd": 50}


def _w(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def out(tmp_path):
    o = tmp_path / "out"
    _w(o / "reports" / "hero_kitchen_sim.json",
       {"room_id": "hero_kitchen", "attempts": 12, "successes": 10, "seconds": 120.0, "human_contacts": 1,
        "families": {"counter_to_sink": {"attempts": 8, "successes": 7, "human_contacts": 1},
                     "open_drawer": {"attempts": 4, "successes": 3, "human_contacts": 0}}})
    _w(o / "eval" / "hero_kitchen_scripted.json",
       {"room_id": "hero_kitchen", "families": {"open_drawer": {"episodes": 4, "successes": 3, "mean_wall_seconds": 5.0,
                                                                  "human_contacts": 0}},
        "totals": {"episodes": 4, "successes": 3, "wall_seconds": 20.0, "human_contacts": 0}})
    _w(o / "jobs" / "abc" / "status.json",
       {"room_id": "hero_kitchen_abc", "status": "done", "started_at": 1000, "finished_at": 1060, "attempts": 6,
        "successes": 6})
    _w(o / "jobs" / "queued" / "status.json", {"status": "queued", "started_at": None, "finished_at": None,
                                                  "attempts": None, "successes": None})
    return o


def _row(rep, source, family):
    return next(r for r in rep["rows"] if r["source"] == source and r["family"] == family)


def test_throughput_per_room_family_and_job(out):
    rep = cost.build(out)
    r = _row(rep, "report", None)
    assert r["demos_per_hour"] == 300.0 and r["seconds_per_demo"] == 12.0
    assert r["attempts_per_demo"] == 1.2 and r["drop_rate"] == round(2 / 12, 4)
    f = _row(rep, "report", "counter_to_sink")
    assert f["drop_rate"] == round(1 / 8, 4) and f["seconds"] is None and f["demos_per_hour"] is None
    e = _row(rep, "eval", "open_drawer")
    assert e["seconds"] == 20.0 and e["seconds_per_demo"] == round(20 / 3, 3)
    j = _row(rep, "job", None)
    assert j["seconds"] == 60 and j["demos_per_hour"] == 360.0
    assert len([r for r in rep["rows"] if r["source"] == "job"]) == 1  # the queued job has no time to report


def test_no_baseline_says_not_measured_and_no_ratio(out):
    rep = cost.build(out)
    text = cost.markdown(rep)
    assert "comparison not measured" in text
    assert "teleop over generated" not in text and "saving" not in text.lower() and "x." not in text
    assert rep["teleop_baseline"] is None and rep["comparison"]["measured"] is False
    assert "seconds_ratio_teleop_over_generated" not in rep["comparison"]


def test_default_baseline_is_the_shipped_p3_run_unless_the_out_dir_has_one(out):
    """The demo export and `arrival cost` both use this, so the site compares against P3 like the CLI does."""
    p3 = cost.default_baseline(out)
    assert p3 is not None and p3.parts[-3:] == ("results", "p3-teleop-2026-10-21", "teleop_baseline.json")
    rep = cost.build(out, baseline_path=p3)
    assert rep["comparison"]["measured"] is True and rep["comparison"]["teleop"]["n_success"] == 20
    _w(out / "teleop_baseline.json", BASELINE)
    assert cost.default_baseline(out) is None and cost.build(out)["teleop_baseline"]["n_success"] == 10


def test_unmeasured_products_stay_null(out):
    d = cost.build(out)["dollars"]
    assert d["local_cpu"] == {"usd": 0, "measured": True}
    for k in ("token_factory", "serverless_jobs", "cosmos", "object_storage", "endpoint"):
        assert d[k] == {"usd": None, "measured": False}


def test_usage_with_assumed_rates_is_an_estimate_never_measured(out, tmp_path):
    usage = _w(tmp_path / "usage.json", {"gpu_hour": 2, "token_factory_per_mtoken": 4})
    d = cost.build(out, usage, PRICING)["dollars"]
    assert d["serverless_jobs"]["usd"] == 5.0 and d["token_factory"]["usd"] == 2.0
    assert d["serverless_jobs"]["measured"] is False and d["serverless_jobs"]["estimate"] is True
    assert d["object_storage"] == {"usd": None, "measured": False}


def test_invoiced_rate_with_usage_is_measured(out, tmp_path):
    pricing = tmp_path / "p.yaml"
    pricing.write_text("products:\n  gpu_hour: {unit: GPU-hour, usd_per_unit: 3.0, basis: invoiced}\n")
    usage = _w(tmp_path / "usage.json", {"gpu_hour": 2})
    d = cost.build(out, usage, pricing)["dollars"]["serverless_jobs"]
    assert d["usd"] == 6.0 and d["measured"] is True and "estimate" not in d


def test_baseline_and_usage_give_comparison_with_both_sample_sizes(out, tmp_path, capsys):
    base = _w(tmp_path / "teleop_baseline.json", BASELINE)
    usage = _w(tmp_path / "usage.json", {"gpu_hour": 2})
    rep = cost.build(out, usage, PRICING, base)
    c = rep["comparison"]
    assert c["measured"] and c["generated"]["n_success"] == 7 and c["teleop"]["n_success"] == 10
    assert rep["teleop_baseline"]["seconds_per_success"] == 150.0 and rep["teleop_baseline"]["usd_per_success"] == 5.0
    assert c["generated"]["scope"] == "task" and c["generated"]["seconds_per_success"] is None  # no per-family seconds
    assert c["generated"]["usd_per_success"] is None and c["generated"]["usd_measured"] is False
    text = cost.markdown(rep)
    assert "| generated (task) | 7 |" in text and "| teleop | 10 | 150.0 | 5.0 |" in text
    assert cost.main(["--out", str(out), "--usage", str(usage), "--pricing", str(PRICING), "--baseline", str(base),
                      "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["comparison"]["teleop"]["n_success"] == 10


def test_room_level_comparison_has_ratio(out, tmp_path):
    base = _w(tmp_path / "b.json", {**BASELINE, "task": "all_tasks"})
    c = cost.build(out, baseline_path=base)["comparison"]
    assert c["generated"]["scope"] == "room" and c["generated"]["n_success"] == 10
    assert c["generated"]["seconds_per_success"] == 12.0 and c["seconds_ratio_teleop_over_generated"] == 12.5
    assert c["generated"]["usd_per_success"] is None


def test_scoped_usage_prices_only_matching_task(out, tmp_path):
    base = _w(tmp_path / "b.json", BASELINE)
    usage = _w(tmp_path / "usage.json", {"scope": {"room": "hero_kitchen", "task": "counter_to_sink"},
                                        "gpu_hour": 2})
    c = cost.build(out, usage, PRICING, base)["comparison"]
    assert c["generated"]["n_success"] == 7
    assert c["generated"]["usd_per_success"] == round(5 / 7, 4)
    usage = _w(tmp_path / "usage.json", {"scope": {"room": "garage", "task": "counter_to_sink"},
                                        "gpu_hour": 2})
    assert cost.build(out, usage, PRICING, base)["comparison"]["generated"]["usd_per_success"] is None


def test_usage_scope_requires_room_and_task(tmp_path):
    with pytest.raises(cost.CostError, match="scope"):
        cost.load_usage(_w(tmp_path / "usage.json", {"scope": {"room": "hero_kitchen"}, "gpu_hour": 2}))


def test_baseline_for_another_room_is_not_measured(out, tmp_path):
    base = _w(tmp_path / "b.json", {**BASELINE, "room": "garage"})
    c = cost.build(out, baseline_path=base)["comparison"]
    assert c["measured"] is False and "garage" in c["reason"]


def test_baseline_for_unreported_task_does_not_use_room_totals(out, tmp_path):
    base = _w(tmp_path / "b.json", {**BASELINE, "task": "pick_and_place"})
    c = cost.build(out, baseline_path=base)["comparison"]
    assert c["measured"] is False
    assert "pick_and_place" in c["reason"]


@pytest.mark.parametrize("patch,msg", [
    ({"n_success": 0}, "n_success"), ({"n_success": 2.5}, "n_success"), ({"operator_seconds_total": -1}, "operator_seconds_total"),
    ({"cost_basis_usd": "50"}, "cost_basis_usd"), ({"setup_assumptions": []}, "setup_assumptions"), ({"room": ""}, "room"),
])
def test_malformed_baseline_raises_clear_error(tmp_path, patch, msg):
    p = _w(tmp_path / "b.json", {**BASELINE, **patch})
    with pytest.raises(cost.CostError, match=msg):
        cost.load_baseline(p)


def test_baseline_missing_field_and_bad_json(tmp_path, out, capsys):
    b = dict(BASELINE)
    del b["room"]
    with pytest.raises(cost.CostError, match="missing room"):
        cost.load_baseline(_w(tmp_path / "b.json", b))
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(cost.CostError, match="cannot read"):
        cost.load_baseline(bad)
    assert cost.main(["--out", str(out), "--baseline", str(bad)]) == 2
    assert "error: teleop baseline" in capsys.readouterr().out


def test_unknown_usage_product_raises(tmp_path):
    with pytest.raises(cost.CostError, match="unknown product"):
        cost.load_usage(_w(tmp_path / "u.json", {"tpu_hour": 1}))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_cost_inputs_cannot_be_reported(tmp_path, bad):
    with pytest.raises(cost.CostError, match="operator_seconds_total"):
        cost.load_baseline(_w(tmp_path / "baseline.json", {**BASELINE, "operator_seconds_total": bad}))
    with pytest.raises(cost.CostError, match="cost_basis_usd"):
        cost.load_baseline(_w(tmp_path / "baseline.json", {**BASELINE, "cost_basis_usd": bad}))
    with pytest.raises(cost.CostError, match="gpu_hour"):
        cost.load_usage(_w(tmp_path / "usage.json", {"gpu_hour": bad}))
    pricing = _w(tmp_path / "pricing.json", {"products": {"gpu_hour": {"usd_per_unit": bad}}})
    with pytest.raises(cost.CostError, match="gpu_hour"):
        cost.load_pricing(pricing)


def test_empty_out_dir(tmp_path):
    text = cost.markdown(cost.build(tmp_path / "nothing"))
    assert "no reports" in text and "comparison not measured" in text


@pytest.mark.parametrize("usage,metric", [
    ({"wall_seconds": -1}, "wall_seconds"),
    ({"frames": 2.5}, "frames"),
    ({"upload_bytes": True}, "upload_bytes"),
    ({"episodes_attempted": 1, "episodes_succeeded": 2}, "episodes_succeeded"),
])
def test_invalid_job_metering_cannot_change_totals(tmp_path, usage, metric):
    (tmp_path / "usage").mkdir()
    (tmp_path / "usage" / "jobs.jsonl").write_text(json.dumps({"id": "job1", **usage}) + "\n")
    with pytest.raises(cost.CostError, match=metric):
        cost.metering_totals(tmp_path)


def test_unreadable_report_is_a_cost_error_not_a_traceback(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "bad.json").write_text("{not json")
    with pytest.raises(cost.CostError):
        cost.collect_rows(tmp_path)


@pytest.mark.parametrize("body", ['{"id":"first"}\n{bad}\n{"id":"last"}\n',
                                  '{"id":"first"}\n{bad}\n'])
def test_corrupt_committed_usage_line_is_not_silently_dropped(tmp_path, body):
    ledger = tmp_path / "usage" / "jobs.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(body)
    with pytest.raises(cost.CostError, match="invalid JSON on line 2"):
        cost.job_usage(tmp_path)


@pytest.mark.parametrize("attempts,successes", [(2, 3), (-1, 0), (True, 1), (3, False)])
def test_inconsistent_report_counts_cannot_produce_throughput(tmp_path, attempts, successes):
    _w(tmp_path / "reports" / "bad.json", {"room_id": "room", "attempts": attempts,
                                          "successes": successes, "seconds": 10})
    row, = cost.collect_rows(tmp_path)
    assert row["attempts"] is row["successes"] is None
    assert row["drop_rate"] is row["demos_per_hour"] is None
