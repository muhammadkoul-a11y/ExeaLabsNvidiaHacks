"""Test that delete_job handles the race condition where the job folder disappears between
reading status and computing paths, instead of crashing on the unpack."""

from pathlib import Path

import arrival.jobs.retention as retention


def test_delete_job_returns_none_when_job_folder_disappears(tmp_path: Path, monkeypatch):
    """delete_job should return None when job_paths returns an empty list (job folder deleted
    between reading status and filtering paths by existence), not crash with ValueError."""
    job_id = "job_abc123def456789012345678abcdef00"
    out_dir = tmp_path / "out"
    job_dir = out_dir / "jobs" / job_id
    job_dir.mkdir(parents=True)

    status = {"id": job_id, "status": "done", "room_id": "test_room"}
    (job_dir / "status.json").write_text('{"id": "job_abc123def456789012345678abcdef00", "status": "done"}')

    def mock_job_paths(out_dir_arg, job_id_arg, status_arg):
        return []

    monkeypatch.setattr(retention, "job_paths", mock_job_paths)

    result = retention.delete_job(out_dir, job_id)
    assert result is None, "delete_job should return None when paths list is empty"


def test_delete_job_works_normally_when_paths_exist(tmp_path: Path):
    """Verify normal operation still works when job folder exists."""
    job_id = "job_abc123def456789012345678abcdef00"
    out_dir = tmp_path / "out"
    job_dir = out_dir / "jobs" / job_id
    job_dir.mkdir(parents=True)

    (job_dir / "status.json").write_text('{"id": "job_abc123def456789012345678abcdef00", "status": "done"}')
    (job_dir / "upload.bin").write_bytes(b"test data")

    result = retention.delete_job(out_dir, job_id, dry_run=True)
    assert result is not None
    assert result["id"] == job_id
    assert any("status.json" in p or job_id in p for p in result["paths"])
