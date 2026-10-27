"""scripts/check_site.py against a --base with no server: one line and exit 1, not one refused connection per path."""
import importlib.util
import urllib.error

import pytest

from conftest import REPO

_spec = importlib.util.spec_from_file_location("check_site", REPO / "scripts" / "check_site.py")
check_site = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_site)


def test_no_server_is_one_line_and_one_request(monkeypatch, capsys):
    calls = []

    def refuse(url, timeout=None):
        calls.append(url)
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(check_site.urllib.request, "urlopen", refuse)
    monkeypatch.setattr("sys.argv", ["check_site.py", "--base", "http://127.0.0.1:9"])
    with pytest.raises(SystemExit) as e:
        check_site.main()
    assert e.value.code == 1
    assert calls == ["http://127.0.0.1:9/"]
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("no site at http://127.0.0.1:9/") and "--offline" in out[0]


def test_offline_makes_no_requests(monkeypatch, capsys):
    monkeypatch.setattr(check_site.urllib.request, "urlopen", lambda *a, **k: pytest.fail("offline fetched a URL"))
    monkeypatch.setattr("sys.argv", ["check_site.py", "--offline"])
    with pytest.raises(SystemExit) as e:
        check_site.main()
    assert e.value.code == 0
    assert "checked" in capsys.readouterr().out
