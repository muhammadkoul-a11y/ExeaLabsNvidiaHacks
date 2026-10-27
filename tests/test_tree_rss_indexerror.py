"""Test that tree_rss handles truncated /proc/[pid]/stat gracefully."""

import sys
from unittest.mock import patch, mock_open
from arrival.jobs.worker import tree_rss


def test_truncated_proc_stat_returns_none():
    """tree_rss should return None (not raise IndexError) when /proc/[pid]/stat is truncated."""
    if sys.platform != "linux":
        return  # the /proc logic only runs on Linux

    # A truncated /proc/[pid]/stat file: only has a few fields after the comm name
    # This is missing the ppid (field 1 after ")") and rss (field 21 after ")")
    truncated_stat = "1 (init) S"  # only the state field, missing ppid and rss

    with patch("os.listdir", return_value=["1"]):
        with patch("pathlib.Path.read_text", return_value=truncated_stat):
            with patch("os.sysconf", return_value=4096):
                # Before the fix, this would raise IndexError on f[1] or f[21]
                # After the fix, it should return None gracefully
                result = tree_rss(1)
                assert result is None, f"Expected None for truncated stat, got {result}"


def test_malformed_ps_output_returns_none():
    """tree_rss should return None (not raise IndexError) when ps output is malformed."""
    if sys.platform == "win32" or sys.platform == "linux":
        return  # the ps fallback only runs on non-linux, non-windows

    # Malformed ps output: only has 2 columns instead of 3
    import subprocess
    malformed_output = "123 456\n"  # missing the rss column

    mock_result = type("CompletedProcess", (), {"stdout": malformed_output})()
    with patch("subprocess.run", return_value=mock_result):
        # Before the fix, this would raise ValueError on unpacking in line.split()
        # Actually ValueError is already caught, so this tests the boundary
        result = tree_rss(123)
        assert result is None, f"Expected None for malformed ps output, got {result}"
