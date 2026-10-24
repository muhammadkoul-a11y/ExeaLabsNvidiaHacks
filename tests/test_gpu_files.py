import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_finetune.sh"
FILES = [SCRIPT, ROOT / "Dockerfile.gpu", ROOT / "Makefile"]


def test_files_and_targets_exist():
    assert SCRIPT.is_file() and (ROOT / "Dockerfile.gpu").is_file()
    targets = set(re.findall(r"^([A-Za-z_][\w-]*):", (ROOT / "Makefile").read_text(), re.M))
    assert {"train", "eval-policy", "serve-policy"} <= targets


def test_script_parses():
    if shutil.which("bash") is None:
        import pytest
        pytest.skip("bash is not installed on this runner")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_make_dry_run_parses():
    for t in ("train", "eval-policy", "serve-policy"):
        r = subprocess.run(["make", "-n", t], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_script_contract_and_no_secrets():
    s = SCRIPT.read_text()
    for needle in ("check_dataset_loader.py", "nvidia-smi", "HF_TOKEN", "--max-steps", "UNVERIFIED"):
        assert needle in s
    # the token is never echoed
    assert not [ln for ln in s.splitlines() if "echo" in ln and "HF_TOKEN" in ln and "is not set" not in ln]
    for f in FILES:
        assert not re.search(r"hf_[A-Za-z0-9]{20,}|nvapi-|sk-[A-Za-z0-9]{20,}", f.read_text()), f
