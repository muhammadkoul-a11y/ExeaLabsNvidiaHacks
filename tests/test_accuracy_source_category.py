import pytest

from arrival.twin.accuracy import _pieces, counter_runs


def test_fixture_without_source_category_does_not_crash():
    """A twin loaded from disk may lack source_category on its fixtures."""
    twin = {
        "room_id": "test",
        "fixtures": [
            {
                "id": "counter_1",
                "type": "counter",
                "size": [2.0, 0.65, 0.92],
                "position": [0.0, 0.0, 0.46],
                "yaw": 0.0,
            }
        ],
        "objects": [],
        "walls": [],
    }
    pieces = _pieces(twin)
    assert len(pieces) == 1
    assert pieces[0]["id"] == "counter_1"


def test_counter_runs_with_no_source_category():
    twin = {
        "room_id": "test",
        "fixtures": [
            {
                "id": "counter_1",
                "type": "counter",
                "size": [2.0, 0.65, 0.92],
                "position": [0.0, 0.0, 0.46],
                "yaw": 0.0,
            }
        ],
        "objects": [],
        "walls": [],
    }
    runs = counter_runs(twin)
    assert len(runs) == 1
    assert runs[0]["length"] == pytest.approx(2.0)
