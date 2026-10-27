"""A scan.json `floors` entry that is not {id, corners: [[x, y], ...]} is refused with a ScanBundleError, like every
other malformed list in a scan. It used to escape as an AttributeError or TypeError, which the upload worker treats
as a machine fault and retries."""
import json

import pytest

from arrival.jobs.worker import load_upload, retryable
from arrival.scan import bundle as scan_bundle
from conftest import SCANS

MALFORMED = {
    "corner list instead of a floor object": [[0, 0], [4, 0], [4, 3], [0, 3]],
    "floor as a string": ["floor"],
    "corners null": [{"id": "floor", "corners": None}],
    "corners as flat numbers": [{"id": "floor", "corners": [0, 0, 4, 0, 4, 3]}],
}


def _hero(floors):
    data = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text(encoding="utf-8"))
    data["floors"] = floors
    return data


@pytest.mark.parametrize("floors", MALFORMED.values(), ids=MALFORMED.keys())
def test_malformed_floor_entry_is_a_scan_bundle_error(floors):
    with pytest.raises(scan_bundle.ScanBundleError, match="floor"):
        scan_bundle.validate(_hero(floors))


@pytest.mark.parametrize("floors", MALFORMED.values(), ids=MALFORMED.keys())
def test_upload_with_malformed_floors_is_not_retried(floors, tmp_path):
    upload = tmp_path / "upload.bin"
    upload.write_text(json.dumps(_hero(floors)), encoding="utf-8")
    with pytest.raises(scan_bundle.ScanBundleError) as e:
        load_upload(upload, "room_job")
    assert not retryable(e.value)


def test_well_formed_floor_still_validates():
    b = scan_bundle.validate(_hero([{"id": "floor", "corners": [[0, 0], [4, 0], [4, 3]]}]))
    assert b["floors"][0]["corners"] == [[0, 0], [4, 0], [4, 3]]
