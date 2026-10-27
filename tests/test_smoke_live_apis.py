import importlib.util
import json

from conftest import REPO

_spec = importlib.util.spec_from_file_location("smoke_live_apis", REPO / "scripts" / "smoke_live_apis.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

KEYS = ("TOKEN_FACTORY_API_KEY", "TAVILY_API_KEY", "NEMOTRON_OMNI_MODEL")


def _env(monkeypatch, tmp_path, **set_):
    monkeypatch.setenv("ARRIVAL_DOTENV", "0")  # a developer's .env must not leak in
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in set_.items():
        monkeypatch.setenv(k, v)


def test_no_key_is_exit_2_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _env(monkeypatch, tmp_path)
    out = tmp_path / "r.json"
    assert smoke.main(["--out", str(out)]) == 2
    assert not out.exists() and "no key set" in capsys.readouterr().err


def test_report_records_each_call_and_never_the_key(tmp_path, monkeypatch):
    secret = "tf-secret-value-123"
    _env(monkeypatch, tmp_path, TOKEN_FACTORY_API_KEY=secret, NEMOTRON_SUPER_MODEL="nvidia/super",
         NEMOTRON_LIGHTNING_MODEL="nvidia/lightning")
    replies = {"super": '{"family": "counter_to_sink", "object": "mug_1", "target": "sink_1"}',
               "lightning": '["Put the mug in the sink.", "Move the mug to the sink.", "Sink the mug.", "Mug to sink."]'}
    monkeypatch.setattr("arrival.models.token_factory.complete", lambda messages, role="super", **k: replies[role])
    out = tmp_path / "r.json"
    assert smoke.main(["--out", str(out)]) == 0
    text = out.read_text()
    assert secret not in text
    checks = json.loads(text)["checks"]
    assert checks["token_factory_planner"]["source"] == "token_factory"
    assert checks["token_factory_planner"]["model"] == "nvidia/super"
    assert checks["token_factory_wordings"]["ok"] and "tavily_research" not in checks


def test_a_failed_call_is_exit_1(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path, TOKEN_FACTORY_API_KEY="k", NEMOTRON_SUPER_MODEL="m", NEMOTRON_LIGHTNING_MODEL="m")

    def down(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("arrival.models.token_factory.complete", down)
    out = tmp_path / "r.json"
    assert smoke.main(["--out", str(out)]) == 1
    assert json.loads(out.read_text())["checks"]["token_factory_planner"]["source"] == "rule_based"


def test_planner_benchmark_without_a_key_is_exit_2(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("planner_benchmark", REPO / "scripts" / "planner_benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    monkeypatch.setenv("ARRIVAL_DOTENV", "0")
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)
    assert bench.main([]) == 2
    assert "TOKEN_FACTORY_API_KEY" in capsys.readouterr().err
    assert bench.right({"family": "open_fridge", "object": None}, "open_fridge", None)
    assert not bench.right({"family": "counter_to_sink", "object": "bowl_1"}, "counter_to_sink", "mug_1")


def test_a_failed_wordings_call_says_why(tmp_path, monkeypatch, capsys):
    # the planner check keeps its log; the wordings check printed FAIL with no reason at all
    import arrival.models.token_factory as tf

    _env(monkeypatch, tmp_path, TOKEN_FACTORY_API_KEY="k", NEMOTRON_SUPER_MODEL="m")
    monkeypatch.delenv("NEMOTRON_LIGHTNING_MODEL", raising=False)
    real = tf.complete
    plan = '{"family": "counter_to_sink", "object": "mug_1", "target": "sink_1"}'
    monkeypatch.setattr(tf, "complete", lambda messages, role="super", **k: plan if role == "super"
                        else real(messages, role=role, **k))
    out = tmp_path / "r.json"
    assert smoke.main(["--out", str(out)]) == 1
    wordings = json.loads(out.read_text())["checks"]["token_factory_wordings"]
    assert wordings["source"] == "template_fallback"
    assert "NEMOTRON_LIGHTNING_MODEL" in wordings["error"]
    assert "NEMOTRON_LIGHTNING_MODEL" in capsys.readouterr().out


def test_detect_check_places_the_fixture_mug_and_only_runs_what_is_asked(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path, TOKEN_FACTORY_API_KEY="k", NEMOTRON_SUPER_MODEL="m", NEMOTRON_LIGHTNING_MODEL="m",
         NEMOTRON_OMNI_MODEL="nvidia/vl")
    calls = []

    def reply(messages, role="super", **k):  # the same box in every frame: the empty room must still place nothing
        calls.append(role)
        return '{"objects": [{"label": "mug", "box": [57, 38, 69, 54], "confidence": 0.9}]}'

    monkeypatch.setattr("arrival.models.token_factory.complete", reply)
    out = tmp_path / "r.json"
    assert smoke.main(["--only", "token_factory_detect", "--out", str(out)]) == 0
    checks = json.loads(out.read_text())["checks"]
    assert list(checks) == ["token_factory_detect"] and calls == ["omni"] * 6
    c = checks["token_factory_detect"]
    assert c["model"] == "nvidia/vl" and c["reply_sources"] == {"token_factory": 6}
    assert c["mug"]["position_error_m"] < 0.03 and c["empty"]["objects"] == []
