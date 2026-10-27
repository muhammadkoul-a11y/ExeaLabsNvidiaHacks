"""Compile the app's Foundation-only saved-scan metadata on macOS; no LiDAR required."""
import platform
import shutil
import subprocess

import pytest

from conftest import REPO

pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or shutil.which("swiftc") is None,
                                reason="needs swiftc on macOS")


@pytest.fixture(scope="module")
def scan_check(tmp_path_factory):
    folder = tmp_path_factory.mktemp("scan_swift")
    main = folder / "main.swift"
    main.write_text('''import Foundation
var scan = Scan(id: "sample", roomName: "kitchen", created: Date(timeIntervalSince1970: 0))
scan.jobStatus = CommandLine.arguments[1] == "nil" ? nil : CommandLine.arguments[1]
let data = try JSONEncoder().encode(scan)
let restored = try JSONDecoder().decode(Scan.self, from: data)
precondition(restored == scan)
print(restored.isJobFinished ? "finished" : "active")
''')
    exe = folder / "scan_check"
    subprocess.run(["swiftc", str(main), str(REPO / "ios/ArrivalCapture/ArrivalCapture/Scan.swift"),
                    "-o", str(exe)], check=True, capture_output=True, timeout=120)
    return exe


@pytest.mark.parametrize(("status", "expected"), [
    ("done", "finished"), ("failed", "finished"), ("cancelled", "finished"),
    ("queued", "active"), ("running", "active"), ("nil", "active"), ("unknown", "active"),
])
def test_saved_scan_terminal_status_survives_round_trip(scan_check, status, expected):
    result = subprocess.run([str(scan_check), status], check=True,
                            capture_output=True, text=True, timeout=10)
    assert result.stdout.strip() == expected
