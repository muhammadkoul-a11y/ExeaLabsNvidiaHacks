"""docs/SCANNING.md is the field protocol people follow to commit a scan; its notes.md template has to pass
validate_scan_dir, and it has to cover the RGB-D frames that `arrival realism` sends people to it for."""

import json
import re

from arrival.scan.intake import FIELDS, validate_scan_dir

from conftest import REPO

DOC = (REPO / "docs" / "SCANNING.md").read_text(encoding="utf-8")


def test_notes_template_passes_validate_scan_dir(tmp_path):
    template = re.search(r"```markdown\n(.*?)```", DOC, re.S)
    assert template, "SCANNING.md has no ```markdown notes.md template"
    (tmp_path / "notes.md").write_text(template.group(1), encoding="utf-8")
    values = {k: m.group(1) for k in FIELDS if k.endswith("_m")
              and (m := re.search(rf"- {re.escape(FIELDS[k])}: ([0-9.]+)", template.group(1)))}
    (tmp_path / "measurements.json").write_text(json.dumps({k: float(v) for k, v in values.items()}), encoding="utf-8")
    problems = [p for p in validate_scan_dir(tmp_path) if "notes.md" in p or "measurements.json" in p]
    assert problems == []


def test_names_the_intake_script_and_the_rgbd_frames():
    assert "scripts/intake_scan.py" in DOC
    assert "RGB-D frames" in DOC and "arrival realism" in DOC
    assert "still planned" not in DOC
