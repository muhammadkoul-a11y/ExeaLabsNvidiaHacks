"""scripts/planner_benchmark.py offline: complete() is stubbed, so the scoring, the rejected-reply accounting and the
v1 report keys are checked with no network and no key."""
import importlib.util
import json
import urllib.error

import pytest

from conftest import REPO

from arrival.sim.tasks import SIM_FAMILIES

_spec = importlib.util.spec_from_file_location("planner_benchmark", REPO / "scripts" / "planner_benchmark.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

V1 = json.loads((REPO / "docs" / "results" / "planner-benchmark-2026-10-26.json").read_text(encoding="utf-8"))
SECRET = "tf-secret-value-123"
# a family's verb, fixture or object; an out_of_scope request that names one would blur the false-positive count
FAMILY_WORDS = ("fridge", "freezer", "wipe", "scrub", "clean", "drawer", "door", "sink", "close", "shut", "push",
                "pull", "open", "oven", "stove", "dish", "faucet", "tap", "water", "table", "counter", "cabinet",
                "cupboard", "move", "put", "place", "load", "slide", "shift", "carry", "lower", "mug", "bowl", "sponge",
                "plate", "cup", "apple", "laptop", "book", "chair")
REJECT = {"There are crumbs everywhere, clean up the counter.":  # the rule parser gets this one right
          '{"family": "wipe_counter", "object": null, "target": "counter_z"}',
          "Let me check if we're out of milk.": "I would open the fridge."}
BLIND = [{"text": "Could you get the fridge open?", "inventory": "hero_kitchen", "family": "open_fridge",
          "object": None, "author": "A"},
         {"text": "Nudge the book along the table.", "inventory": "office", "family": "move_on_table",
          "object": "book_1", "author": "B"},
         {"text": "Slide the laptop over on the table.", "inventory": "office", "family": "move_on_table",
          "object": "laptop_1", "author": "B"}]


def _env(monkeypatch):
    monkeypatch.setenv("ARRIVAL_DOTENV", "0")
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", SECRET)
    monkeypatch.setenv("NEMOTRON_SUPER_MODEL", "nvidia/super")
    monkeypatch.setenv("NEMOTRON_LIGHTNING_MODEL", "nvidia/lightning")


def _stub(monkeypatch, seen):
    answers = {w: (fam, obj) for w, fam, obj in bench.INDIRECT}
    answers.update({r["text"]: (r["family"], r["object"]) for r in BLIND})

    def complete(messages, role="super", **params):
        seen.append((role, params))
        if role == "lightning":  # two wordings that name the family, so the stub planner can answer them
            fam = messages[1]["content"].split("Task family: ")[1].split("\n")[0]
            return json.dumps([f"{fam} please.", f"{fam} now."])
        text = json.loads(messages[1]["content"])["instruction"]
        if text in REJECT:
            return REJECT[text]
        fam = text.split(" ")[0]
        if fam in bench.FAMILIES:
            return json.dumps({"family": fam, "object": bench.FAMILIES[fam][0], "target": None})
        fam, obj = answers.get(text, ("open_fridge", None))  # out_of_scope and unlisted: always the fridge
        return json.dumps({"family": fam, "object": obj, "target": None})

    monkeypatch.setattr("arrival.models.token_factory.complete", complete)


def test_scores_rejections_and_keeps_v1_keys(tmp_path, monkeypatch):
    _env(monkeypatch)
    seen = []
    _stub(monkeypatch, seen)
    blind = tmp_path / "blind.json"
    blind.write_text(json.dumps({"requests": BLIND}))
    out = tmp_path / "r.json"
    assert bench.main(["--per-family", "2", "--blind", str(blind), "--out", str(out)]) == 0
    text = out.read_text()
    rep = json.loads(text)
    # no key, no header: of the call only the model id, the temperature and the base URL
    assert SECRET not in text and "Bearer" not in text and "Authorization" not in text
    assert rep["planner_params"] == {"model": "nvidia/super", "temperature": 0,
                                     "base_url": "https://api.tokenfactory.nebius.com/v1"}

    assert rep["version"] == 2 and set(V1) <= set(rep)
    assert rep["objects"] == V1["objects"] and rep["fixtures"] == V1["fixtures"]
    for kind in ("generated", "indirect"):
        assert set(V1["summary"][kind]) <= set(rep["summary"][kind])
    assert all(set(V1["rows"][0]) <= set(r) for r in rep["rows"])
    assert {"kitchen_real_01", "office"} <= set(rep["inventories"])

    # one live call per case at temperature 0, plus one wordings call per family
    assert sum(r == "super" for r, _ in seen) == len(rep["rows"]) == 18 + 15 + 3 + 20 + 20
    assert sum(r == "lightning" for r, _ in seen) == 9 == rep["live_calls"]["wordings"]
    assert all(p == {"temperature": 0} for r, p in seen if r == "super")
    prompt = rep["planner_system_prompt"]  # LISTED is what the prompt names, and only that
    assert all(f in prompt for f in bench.LISTED) and not any(fam in prompt for fam in bench.UNLISTED_ROOM)
    assert all(r["live_call"] and r["calls"] == 1 for r in rep["rows"])

    g, ind = rep["summary"]["generated"], rep["summary"]["indirect"]
    assert g["family"]["nemotron"]["k"] == g["family"]["nemotron"]["n"] == 18 and g["rejected"]["k"] == 0
    assert ind["rejected"] == {"k": 2, "n": 15, "rate": round(2 / 15, 4), "wilson_95": bench.rate(2, 15)["wilson_95"]}
    assert ind["reject_reasons"] == {"object_or_target_not_in_inventory": 1, "no_json_object": 1}
    assert ind["family"]["nemotron"]["k"] == 13 and ind["object"]["nemotron"]["n"] == 6
    assert ind["nemotron_live_calls"] == 13  # v1 meaning: plans the planner accepted
    assert ind["confusion"]["nemotron"]["wipe_counter"] == {"wipe_counter": 1, "rejected": 1}

    crumbs = next(r for r in rep["rows"] if r["instruction"].startswith("There are crumbs"))
    assert crumbs["raw_reply"] == REJECT[crumbs["instruction"]] and not crumbs["reply_accepted"]
    assert "counter_z" in crumbs["reject_reason"] and crumbs["nemotron_source"] == "rule_based"
    assert crumbs["nemotron_ok"] is True and crumbs["nemotron_plan_ok"] is False  # v1 credits the parser fallback
    assert ind["nemotron_right"] == ind["right"]["nemotron"]["k"] + 1

    oos = rep["summary"]["out_of_scope"]
    assert oos["family"] is None and oos["nemotron_right"] is None
    assert oos["confusion"]["nemotron"] == {"out_of_scope": {"open_fridge": 20}}
    assert rep["summary"]["unlisted_families"]["family"]["nemotron"]["k"] == 0

    blind_s = rep["summary"]["blind"]  # all rows, then only the row whose family the prompt lists
    assert blind_s["right"]["nemotron"]["k"] == 3 and blind_s["right"]["rule"] == bench.rate(1, 3)
    assert all(r["sim_executable"] is None for r in rep["rows"] if r["set"] == "out_of_scope")
    if importlib.util.find_spec("mujoco"):  # the reach check behind sim_executable needs it; null without it
        assert blind_s["sim_not_executable"] == 1  # the laptop row is kept and counted, not dropped
        laptop = next(r for r in rep["rows"] if r["object"] == "laptop_1")
        assert laptop["sim_executable"] is False and laptop["nemotron_plan_ok"] is True
        assert all(r["sim_executable"] for r in rep["rows"] if r["set"] == "unlisted_families")
    listed = blind_s["listed_families"]
    assert listed["n"] == 1 and listed["family"]["nemotron"]["k"] == 1 and listed["family"]["rule"]["k"] == 1
    assert next(r for r in rep["rows"] if r["set"] == "blind" and r["inventory"] == "office")["nemotron_object_ok"]


def test_a_timeout_is_a_call_error_not_a_rejection_and_spares_later_cases(tmp_path, monkeypatch):
    # VLMPlanner goes offline after an OSError; a fresh planner per case keeps one timeout from turning every later
    # case into a rule-parser answer
    _env(monkeypatch)
    seen = []
    _stub(monkeypatch, seen)
    import arrival.models.token_factory as tf

    stub = tf.complete
    first = bench.INDIRECT[0][0]

    def complete(messages, role="super", **params):
        if role == "super" and json.loads(messages[1]["content"])["instruction"] == first:
            raise TimeoutError("timed out")
        return stub(messages, role, **params)

    monkeypatch.setattr("arrival.models.token_factory.complete", complete)
    out = tmp_path / "r.json"
    assert bench.main(["--sets", "indirect", "--out", str(out)]) == 1  # a case got no reply
    rep = json.loads(out.read_text())
    ind = rep["summary"]["indirect"]
    assert ind["call_errors"] == {"timeout": 1} and ind["live_calls"] == 14
    assert ind["reject_reasons"] == {"object_or_target_not_in_inventory": 1, "no_json_object": 1}
    assert ind["confusion"]["nemotron"]["counter_to_sink"] == {"no_reply": 1, "counter_to_sink": 1}
    assert [r["call_error_kind"] for r in rep["rows"]].count("timeout") == 1
    assert sum(r["nemotron_source"] == "token_factory" for r in rep["rows"]) == 12  # 14 replies, 2 rejected


def test_null_content_is_a_call_error(tmp_path, monkeypatch):
    _env(monkeypatch)
    seen = []
    _stub(monkeypatch, seen)
    import arrival.models.token_factory as tf

    stub, first = tf.complete, bench.INDIRECT[0][0]

    def complete(messages, role="super", **params):  # the real complete returns message.content, here null
        if role == "super" and json.loads(messages[1]["content"])["instruction"] == first:
            return None
        return stub(messages, role, **params)

    monkeypatch.setattr("arrival.models.token_factory.complete", complete)
    out = tmp_path / "r.json"
    assert bench.main(["--sets", "indirect", "--out", str(out)]) == 1
    rep = json.loads(out.read_text())
    row = next(r for r in rep["rows"] if r["instruction"] == first)
    assert row["call_error_kind"] == "no_content" and not row["live_call"] and row["raw_reply"] is None
    assert rep["summary"]["indirect"]["call_errors"] == {"no_content": 1}


def test_429_and_5xx_are_retried_three_times_and_nothing_else(tmp_path, monkeypatch, capsys):
    _env(monkeypatch)
    seen = []
    _stub(monkeypatch, seen)
    import arrival.models.token_factory as tf

    stub = tf.complete
    busy, down, bad = (w for w, _, _ in bench.INDIRECT[:3])
    tries = {busy: 0, down: 0, bad: 0}

    def complete(messages, role="super", **params):
        text = json.loads(messages[1]["content"])["instruction"] if role == "super" else None
        if text in tries:
            tries[text] += 1
            if text == busy and tries[text] <= 2:
                raise urllib.error.HTTPError("u", 529 if tries[text] == 1 else 429, "Busy", {}, None)
            if text == down:
                raise urllib.error.HTTPError("u", 503, "Unavailable", {}, None)
            if text == bad:
                raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
        return stub(messages, role, **params)

    waits = []
    monkeypatch.setattr("arrival.models.token_factory.complete", complete)
    monkeypatch.setattr(bench.time, "sleep", waits.append)
    out = tmp_path / "r.json"
    assert bench.main(["--sets", "indirect", "--out", str(out)]) == 1  # two cases got no reply
    assert tries == {busy: 3, down: 4, bad: 1} and waits == [2, 4, 2, 4, 8]
    rows = {r["instruction"]: r for r in json.loads(out.read_text())["rows"]}
    assert rows[busy]["retries"] == 2 and rows[busy]["live_call"] and rows[busy]["call_error"] is None
    assert rows[down]["retries"] == 3 and rows[down]["call_error_kind"] == "http_503"
    assert rows[bad]["retries"] == 0 and rows[bad]["call_error_kind"] == "http_400"
    assert all(r["retries"] == 0 for t, r in rows.items() if t not in tries)
    err = capsys.readouterr().err
    assert f"retry 2/3 in 4 s after http_429: indirect {busy!r}" in err
    assert f"retry 3/3 in 8 s after http_503: indirect {down!r}" in err and repr(bad) not in err
    assert SECRET not in out.read_text()


@pytest.mark.parametrize("err, kind", [(TimeoutError("timed out"), "timeout"),
                                       (urllib.error.URLError(TimeoutError("timed out")),
                                        "timeout"),
                                       (urllib.error.HTTPError("u", 403, "Forbidden", {}, None),
                                        "http_403"),
                                       (TypeError("x"), "other"),
                                       (ValueError("x"), "other")])
def test_call_error_kind(err, kind):
    assert bench.call_error_kind(err) == kind


def test_placeholder_blind_rows_stop_before_any_call(tmp_path, monkeypatch, capsys):
    _env(monkeypatch)

    def no_network(*a, **k):
        raise AssertionError("called Token Factory")

    monkeypatch.setattr("arrival.models.token_factory.complete", no_network)
    out = tmp_path / "r.json"
    placeholder = tmp_path / "placeholder.json"
    placeholder.write_text(json.dumps({"requests": [dict(BLIND[0], placeholder=True)]}))
    assert bench.main(["--blind", str(placeholder), "--out", str(out)]) == 2
    assert not out.exists() and "placeholder" in capsys.readouterr().err
    assert bench.main(["--dry-run"]) == 0  # the shipped file loads
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"requests": [{"text": "x", "inventory": "office", "family": "fold_laundry"}]}))
    assert bench.main(["--blind", str(bad), "--out", str(out)]) == 2


def test_out_of_scope_requests_name_no_family_word():
    assert len(bench.OUT_OF_SCOPE) == 20 and len(bench.UNLISTED) == 20
    for text, _ in bench.OUT_OF_SCOPE:
        assert not [w for w in FAMILY_WORDS if w in text.lower()], text
    assert {fam for _, fam, _ in bench.UNLISTED} == set(SIM_FAMILIES) - set(bench.LISTED) == set(bench.UNLISTED_ROOM)


def test_inventories_are_the_ids_the_cli_sends():
    # the benchmark builds each twin as `arrival demo --from-roomplan` does; the tests compile, then place objects
    from arrival.scan import roomplan
    from arrival.twin.compile import compile_twin, place_default_objects

    inv = bench.inventories()
    for room in bench.ROOMPLAN_ROOMS:
        twin = compile_twin(roomplan.load(REPO / "tests" / "fixtures" / "roomplan" / f"{room}.json", room))
        place_default_objects(twin, seed=0)
        assert inv[room]["objects"] == [o["id"] for o in twin["objects"]]
        assert inv[room]["fixtures"] == [f["id"] for f in twin["fixtures"]]


def test_each_unlisted_request_is_a_task_the_sim_makes_in_its_room():
    # a miss in unlisted_families should be about the family the prompt leaves out, not a room without the fixture
    pytest.importorskip("mujoco")
    twins = bench.room_twins()
    assert [f["id"] for f in twins["hero_kitchen"]["fixtures"]] == bench.FIXTURES
    assert [o["id"] for o in twins["hero_kitchen"]["objects"]] == bench.OBJECTS
    made = {}
    for text, fam, obj in bench.UNLISTED:
        assert bench.sim_executable(twins[bench.UNLISTED_ROOM[fam]], fam, obj, made), text
    # the same check flags what the sim cannot run: the laptop is not TABLE_MOVABLE, the office has no dishwasher
    assert not bench.sim_executable(twins["office"], "move_on_table", "laptop_1")
    assert not bench.sim_executable(twins["office"], "open_dishwasher", None)
