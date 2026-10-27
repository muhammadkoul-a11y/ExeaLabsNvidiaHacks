"""Make one real call to each configured hosted API and keep the evidence: Nemotron on Nebius Token Factory (the
sim planner, the instruction wordings and the C4 object detector) and Tavily (appliance research).

    python scripts/smoke_live_apis.py                       # writes docs/results/live-apis-<UTC date>.json
    python scripts/smoke_live_apis.py --out out/live.json
    python scripts/smoke_live_apis.py --only token_factory_detect --out docs/results/live-detect.json

Keys come from the environment or an untracked .env, as everywhere else (TOKEN_FACTORY_API_KEY with
NEMOTRON_SUPER_MODEL and NEMOTRON_LIGHTNING_MODEL, NEMOTRON_OMNI_MODEL for detection, TAVILY_API_KEY). Every call goes through the same code the sim
uses, with an empty throwaway cache, so a shipped cache cannot answer for the API. The report names the models,
the latency, the source each call reported and a short excerpt of the reply; it never contains a key.

Exits 0 when every configured API answered, 1 when one failed, 2 when none is configured.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OBJECTS, FIXTURES = ["bowl_1", "mug_1"], ["counter_a", "counter_b", "fridge_1", "sink_1", "stove_1"]
INSTRUCTION = "Pick up the mug from the counter and place it in the sink."


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t, 3)


def planner(tmp: Path) -> dict:
    from arrival.models.token_factory import config
    from arrival.sim.policy import VLMPlanner

    logs: list[str] = []
    p = VLMPlanner(log=logs.append, cache_path=tmp / "nemotron.json")
    plan, s = _timed(lambda: p.plan(INSTRUCTION, OBJECTS, FIXTURES))
    return {"ok": p.source == "token_factory", "model": config()["models"]["super"], "seconds": s,
            "source": p.source, "plan": plan, "log": logs}


def wordings(tmp: Path) -> dict:
    from arrival.models.token_factory import config
    from arrival.tasks.instructions import generate_instructions

    res, s = _timed(lambda: generate_instructions("counter_to_sink", "mug", n=4, cache_path=tmp / "instr.json"))
    return {"ok": res.get("source") == "token_factory", "model": config()["models"]["lightning"], "seconds": s,
            "source": res.get("source"), "train": res.get("train"), "eval": res.get("eval"),
            **({"error": res["error"]} if res.get("error") else {})}


DETECT_FIXTURES = REPO / "tests" / "fixtures" / "detection"
MUG = [3.2, 0.45, 0.97]  # where scripts/make_detection_fixture.py rendered the mug, in the twin's frame


def detection(tmp: Path) -> dict:
    """The C4 detector on the rendered fixture frames: one mug on counter_b in `mug/`, nothing in `empty/`."""
    from arrival.scan.bundle import load_room
    from arrival.twin import detect
    from arrival.twin.compile import compile_twin

    bundle = load_room("hero_kitchen", REPO / "scans")
    twin = compile_twin(bundle)
    twin["objects"] = []
    logs: list[str] = []
    clients = {name: detect.DetectionClient(tmp / f"detect-{name}.json", log=logs.append) for name in ("mug", "empty")}
    runs, s = _timed(lambda: {name: detect.detect_objects(DETECT_FIXTURES / name, twin, bundle, client=c,
                                                          log=logs.append) for name, c in clients.items()})
    replies = {f"{name}/{e['frame']}": e["reply"]["objects"] for name, c in clients.items() for e in c.entries.values()}
    mug, empty = runs["mug"], runs["empty"]
    placed = [o for o in mug["objects"] if o["catalog_id"] == "mug"]
    err = round(sum((a - b) ** 2 for a, b in zip(placed[0]["position"], MUG)) ** 0.5, 4) if placed else None
    sources = {k: mug["reply_sources"].get(k, 0) + empty["reply_sources"].get(k, 0)
               for k in {*mug["reply_sources"], *empty["reply_sources"]}}
    return {"ok": sources == {"token_factory": 6} and len(placed) == 1 and not empty["objects"],
            "model": clients["mug"].model, "seconds": s, "source": "token_factory" if "token_factory" in sources else None,
            "reply_sources": sources, "replies": replies,
            "mug": {"state": mug["state"], "objects": mug["objects"], "rejected": mug["rejected"],
                    "truth": MUG, "position_error_m": err},
            "empty": {"state": empty["state"], "objects": empty["objects"], "rejected": empty["rejected"]},
            "log": logs}


def tavily(tmp: Path) -> dict:
    from arrival.tasks.tavily_research import TavilyResearchAgent

    logs: list[str] = []
    agent = TavilyResearchAgent(log=logs.append, cache_path=tmp / "tavily.json")
    res, s = _timed(lambda: agent.research_appliance("fridge"))
    return {"ok": res.get("source") == "tavily_api", "seconds": s, "source": res.get("source"),
            "summary": (res.get("summary") or "")[:300], "sources": res.get("sources", [])[:5], "log": logs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="report path (default docs/results/live-apis-<UTC date>.json)")
    ap.add_argument("--only", action="append", help="run only this check (repeatable)")
    a = ap.parse_args(argv)

    from arrival import config as settings

    checks = []
    if settings.get("TOKEN_FACTORY_API_KEY"):
        checks += [("token_factory_planner", planner), ("token_factory_wordings", wordings)]
        if settings.get("NEMOTRON_OMNI_MODEL"):
            checks.append(("token_factory_detect", detection))
    if settings.get("TAVILY_API_KEY"):
        checks.append(("tavily_research", tavily))
    if a.only:
        checks = [c for c in checks if c[0] in a.only]
    if not checks:
        print("no key set: put TOKEN_FACTORY_API_KEY (and the NEMOTRON_*_MODEL ids) or TAVILY_API_KEY in .env",
              file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    report = {"run_at": now.isoformat(timespec="seconds"), "checks": {}}
    with tempfile.TemporaryDirectory() as d:
        for name, fn in checks:
            try:
                report["checks"][name] = fn(Path(d))
            except Exception as e:  # a missing model id raises before any request; record it like a failed call
                report["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            r = report["checks"][name]
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {name}: source={r.get('source')} {r.get('seconds', '-')} s"
                  + (f" ({r['error']})" if r.get("error") else ""))

    out = a.out or REPO / "docs" / "results" / f"live-apis-{now:%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if all(r["ok"] for r in report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
