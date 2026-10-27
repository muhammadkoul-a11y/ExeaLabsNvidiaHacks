"""`--instructions nemotron` with no Token Factory key falls back to the template wordings for every sim family.

close_drawer, close_door, put_in_drawer, wipe_counter, open_fridge and move_on_table had no fallback templates, so
the offline path wrote "Do task <name>" as the episode's instruction and had no held-out wording for them:
`sim --instructions nemotron --split eval` stopped with "empty range for randrange()".
"""

import pytest

pytest.importorskip("mujoco")

from arrival.sim.tasks import SIM_FAMILIES, SIM_WORDINGS, make_tasks  # noqa: E402
from arrival.tasks.instructions import FALLBACK_TEMPLATES, generate_instructions  # noqa: E402

ADDED = ("close_drawer", "close_door", "put_in_drawer", "wipe_counter", "open_fridge", "move_on_table")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.delenv("TOKEN_FACTORY_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # the default cache is out/cache/ under the cwd: never read a developer's real one


@pytest.mark.parametrize("family", SIM_FAMILIES)
def test_every_sim_family_falls_back_to_real_wordings(family, offline):
    res = generate_instructions(family, object_name="mug")
    assert res["source"] == "template_fallback"
    assert [w for w in res["train"] + res["eval"] if w.startswith("Do task")] == []
    assert res["train"] and res["eval"]  # --split eval needs a held-out wording
    assert set(res["train"]).isdisjoint(res["eval"])


def test_added_fallbacks_are_the_sim_wordings():
    for family in ADDED:
        assert FALLBACK_TEMPLATES[family] == SIM_WORDINGS[family]


def test_eval_split_with_generated_wordings_runs_offline(hero_twin, offline):
    tasks = make_tasks(hero_twin, ["close_drawer", "open_fridge"], 2, split="eval", use_generated_instructions=True,
                       log=lambda *_: None)
    assert [t["instruction"] for t in tasks] == ["Push the drawer on the counter b back in.", "Get the fridge open."]
    assert {t["instruction_source"] for t in tasks} == {"template_fallback"}
