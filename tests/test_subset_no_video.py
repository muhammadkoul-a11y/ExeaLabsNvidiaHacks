import json
import pytest
from pathlib import Path

from arrival.export.subset import SubsetError, subset_dataset
from arrival.jobs.shards import ShardError, merge_shards


def test_subset_rejects_dataset_without_video_features(tmp_path):
    """subset requires video features to determine resolution."""
    root = tmp_path / "datasets" / "test_sim"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "arrival_room_id": "test",
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
        },
        "total_episodes": 1,
    }))
    (meta / "episodes.jsonl").write_text('{"episode_index": 0, "length": 10, "arrival_family": "counter_to_sink"}\n')
    (meta / "embodiment.json").write_text('{}')

    with pytest.raises(SubsetError, match="no video features"):
        subset_dataset(root, tmp_path / "out")


def test_merge_shards_rejects_shards_without_video_features(tmp_path):
    """merge-shards also takes the resolution from the first camera, so a shard without cameras is refused."""
    root = tmp_path / "shards" / "0of1" / "datasets" / "test_sim"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "arrival_room_id": "test",
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
        },
        "total_episodes": 1,
        "arrival_shard": {"index": 0, "count": 1, "task_range": [0, 1], "total_tasks": 1,
                          "families": ["counter_to_sink"]},
    }))
    (meta / "episodes.jsonl").write_text(
        '{"episode_index": 0, "length": 10, "arrival_task_id": "sim_counter_to_sink_0000"}\n')
    (meta / "modality.json").write_text('{}')
    (meta / "embodiment.json").write_text('{}')

    with pytest.raises(ShardError, match="no video features"):
        merge_shards([tmp_path / "shards"], tmp_path / "out")
    assert not (tmp_path / "out").exists()
