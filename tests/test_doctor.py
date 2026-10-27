"""arrival doctor: offline environment checks before a run."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from arrival import cli, doctor
from arrival.doctor import Check, check_core, check_disk, check_env, check_ffmpeg, check_mujoco, check_python, exit_code, render, render_json, run_checks


def test_check_python():
    # Current runtime
    chk = check_python()
    assert chk.name == "python"
    assert chk.status == "ok"
    assert "on" in chk.detail

    # Mock older Python version (e.g. 3.10)
    old = check_python(version_info=(3, 10, 8))
    assert old.status == "fail"
    assert "needs 3.11 or newer" in old.detail

    # Supported versions
    assert check_python(version_info=(3, 11, 0)).status == "ok"
    assert check_python(version_info=(3, 13, 1)).status == "ok"


def test_check_core():
    checks = check_core()
    assert len(checks) == len(doctor.CORE)
    for c in checks:
        assert c.status == "ok"
        assert c.detail != "unknown version"

    # Missing core package
    fake_deps = {"nonexistent_module_xyz_123": "fake-pkg"}
    fake_chk = check_core(fake_deps)
    assert len(fake_chk) == 1
    assert fake_chk[0].status == "fail"
    assert "not installed" in fake_chk[0].detail


def test_check_mujoco():
    chk = check_mujoco()
    assert chk.name == "mujoco"
    assert chk.status in ("ok", "warn")

    with patch.dict("sys.modules", {"mujoco": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module named 'mujoco'")):
            res = check_mujoco()
            assert res.status == "warn"
            assert "pip install" in res.detail

    with patch("builtins.__import__", side_effect=RuntimeError("DLL failed to load")):
        res = check_mujoco()
        assert res.status == "fail"
        assert "RuntimeError" in res.detail


def test_check_ffmpeg():
    chk = check_ffmpeg()
    assert chk.name == "ffmpeg"
    assert chk.status in ("ok", "fail")

    # Missing binary
    missing = check_ffmpeg(exe="nonexistent_ffmpeg_binary_xyz_123")
    assert missing.status == "fail"
    assert "no ffmpeg found" in missing.detail

    # Failing binary
    with patch("subprocess.run", side_effect=OSError("Exec format error")):
        failed = check_ffmpeg(exe=str(Path(__file__).resolve()))
        assert failed.status == "fail"
        assert "binary does not run" in failed.detail


def test_env_parsing_and_privacy(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        "EMPTY=\n"
        "TOKEN_FACTORY_API_KEY=super_secret_token_value_999\n"
        "export TAVILY_API_KEY=\"tavily_secret_abc\"\n"
        "'GR00T_POLICY_URL'='http://secret-gr00t.local:8000'\n"
        "NO_EQUALS_HERE\n",
        encoding="utf-8",
    )

    names = doctor._env_file_names([env_file])
    assert "TOKEN_FACTORY_API_KEY" in names
    assert "TAVILY_API_KEY" in names
    assert "GR00T_POLICY_URL" in names
    assert "EMPTY" not in names
    assert "NO_EQUALS_HERE" not in names

    # Crucial security guarantee: secrets are never in the returned Checks
    checks = check_env(environ={}, env_files=[env_file])
    for c in checks:
        assert "super_secret" not in c.detail
        assert "tavily_secret" not in c.detail
        assert "http://secret" not in c.detail

    tf_check = next(c for c in checks if c.name == "TOKEN_FACTORY_API_KEY")
    assert tf_check.status == "ok"
    assert "set in .env" in tf_check.detail

    cosmos_check = next(c for c in checks if c.name == "COSMOS_URL")
    assert cosmos_check.status == "info"
    assert "not set" in cosmos_check.detail

    # When set in os.environ
    env_checks = check_env(environ={"COSMOS_URL": "http://my-cosmos:5000"}, env_files=[])
    c_chk = next(c for c in env_checks if c.name == "COSMOS_URL")
    assert c_chk.status == "ok"
    assert "set in the environment" in c_chk.detail
    assert "http://my-cosmos" not in c_chk.detail


def test_check_disk(tmp_path):
    # Existing valid directory
    chk = check_disk(tmp_path)
    assert chk.name == "disk"
    assert chk.status in ("ok", "warn")

    # Non-existent subfolder under existing parent
    sub_chk = check_disk(tmp_path / "sub" / "folder")
    assert sub_chk.name == "disk"
    assert sub_chk.status in ("ok", "warn")

    # Existing file
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("hello", encoding="utf-8")
    file_chk = check_disk(dummy_file)
    assert file_chk.status == "fail"
    assert "existing file, not a directory" in file_chk.detail

    # Mock low disk space (warn)
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value.free = int(1.5 * doctor.GIB)
        warn_chk = check_disk(tmp_path)
        assert warn_chk.status == "warn"
        assert "low (under 2 GiB)" in warn_chk.detail

    # Mock critical disk space (fail)
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value.free = int(0.1 * doctor.GIB)
        fail_chk = check_disk(tmp_path)
        assert fail_chk.status == "fail"
        assert "need at least 0.25 GiB" in fail_chk.detail

    # Mock unwritable directory
    with patch("os.access", return_value=False):
        unwritable_chk = check_disk(tmp_path)
        assert unwritable_chk.status == "fail"
        assert "not writable" in unwritable_chk.detail


def test_exit_code_and_render():
    ok_checks = [
        Check("python", "ok", "3.13.0"),
        Check("numpy", "ok", "2.0.0"),
        Check("COSMOS_URL", "info", "optional"),
        Check("disk", "warn", "low space"),
    ]
    assert exit_code(ok_checks) == 0

    fail_checks = ok_checks + [Check("ffmpeg", "fail", "not found")]
    assert exit_code(fail_checks) == 1

    rendered = render(ok_checks)
    assert "[ok  ] python: 3.13.0" in rendered
    assert "[info] COSMOS_URL: optional" in rendered
    assert "[WARN] disk: low space" in rendered
    assert "0 problem(s), 1 warning(s)" in rendered

    rendered_fail = render(fail_checks)
    assert "[FAIL] ffmpeg: not found" in rendered_fail
    assert "1 problem(s), 1 warning(s)" in rendered_fail

    # JSON render
    j_ok = json.loads(render_json(ok_checks))
    assert j_ok["ok"] is True
    assert len(j_ok["checks"]) == 4

    j_fail = json.loads(render_json(fail_checks))
    assert j_fail["ok"] is False
    assert len(j_fail["checks"]) == 5


def test_run_checks_integration(tmp_path):
    checks = run_checks(out_dir=tmp_path)
    assert isinstance(checks, list)
    assert any(c.name == "python" for c in checks)
    assert any(c.name == "disk" for c in checks)
    assert any(c.name == "ffmpeg" for c in checks)


def test_cli_doctor(tmp_path, capsys):
    rc = cli.main(["doctor", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "[ok  ] python:" in out
    assert rc in (cli.EXIT_OK, cli.EXIT_INVALID)

    # CLI --json
    rc_json = cli.main(["doctor", "--out", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert "checks" in data
    assert rc_json in (cli.EXIT_OK, cli.EXIT_INVALID)

    # CLI with file destination fails cleanly
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("x", encoding="utf-8")
    rc_fail = cli.main(["doctor", "--out", str(dummy)])
    out_fail = capsys.readouterr().out
    assert rc_fail == cli.EXIT_INVALID
    assert "[FAIL] disk:" in out_fail

def test_doctor_nebius_cloud_env_keys(tmp_path):
    from arrival.jobs.nebius import JOB_SETTINGS, STORE_SETTINGS
    env = {
        "NEBIUS_IAM_TOKEN": "secret_iam_token_xyz",
        "NEBIUS_PROJECT_ID": "project-12345",
        "NEBIUS_JOB_IMAGE": "cr.nemotron/arrival:v1",
        "NEBIUS_JOB_S3_SECRET_ID": "secret-s3-id-999",
        "NEBIUS_BUCKET": "my-bucket",
        "NEBIUS_ACCESS_KEY_ID": "access_key_abc",
        "NEBIUS_SECRET_ACCESS_KEY": "secret_access_key_def",
    }
    assert set(env.keys()) == set(JOB_SETTINGS + STORE_SETTINGS)
    checks = check_env(environ=env, env_files=[])
    for k in env:
        chk = next(c for c in checks if c.name == k)
        assert chk.status == "ok"
        assert "set in the environment" in chk.detail
        assert env[k] not in chk.detail  # value never leaked


def test_token_factory_key_without_model_ids_warns(tmp_path):
    # With the key set but no model ids, the planner and the wordings fall back offline; doctor must say so.
    checks = check_env(environ={"TOKEN_FACTORY_API_KEY": "k"}, env_files=[])
    warned = {c.name for c in checks if c.status == "warn"}
    assert warned == {"NEMOTRON_SUPER_MODEL", "NEMOTRON_LIGHTNING_MODEL", "NEMOTRON_OMNI_MODEL"}
    assert "warning(s)" in render(checks)
    assert exit_code(checks) == 0

    env_file = tmp_path / ".env"
    env_file.write_text("TOKEN_FACTORY_API_KEY=k\nNEMOTRON_SUPER_MODEL=nvidia/nemotron-3-super-120b-a12b\n", encoding="utf-8")
    checks = check_env(environ={"NEMOTRON_OMNI_MODEL": "m"}, env_files=[env_file])
    assert [c.name for c in checks if c.status == "warn"] == ["NEMOTRON_LIGHTNING_MODEL"]
    assert not any("nemotron-3-super" in c.detail for c in checks)

    # No key: the model ids do not matter, so nothing is flagged.
    assert not any(c.status == "warn" for c in check_env(environ={}, env_files=[]))
