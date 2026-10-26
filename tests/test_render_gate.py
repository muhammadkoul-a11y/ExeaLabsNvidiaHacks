from conftest import REPO

pytest_plugins = ["pytester"]


def test_no_render_covers_shared_fixtures(pytester, monkeypatch):
    # Nested pytest runs must not inherit the parent CI shard selection.
    monkeypatch.delenv("ARRIVAL_TEST_SHARD", raising=False)
    pytester.makeconftest((REPO / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makeini("[pytest]\nmarkers = render: needs an OpenGL context\n")
    pytester.makepyfile(mujoco="""
        from pathlib import Path

        def Renderer():
            Path("renderer-called").touch()
            return "frame"
    """, test_fixtures="""
        from pathlib import Path
        import mujoco
        from mujoco import Renderer
        import pytest
        import subprocess
        import sys

        @pytest.fixture(scope="session")
        def session_frame():
            return mujoco.Renderer()

        @pytest.fixture(scope="module")
        def module_frame():
            return Renderer()

        @pytest.fixture
        def function_frame():
            return mujoco.Renderer()

        @pytest.fixture(scope="module")
        def child_frame():
            subprocess.run([sys.executable, "-c",
                            "from pathlib import Path; import mujoco; "
                            "mujoco.Renderer(); Path('child-process-called').touch()"], check=True)

        def test_session_frame(session_frame):
            assert session_frame == "frame"

        def test_module_frame(module_frame):
            assert module_frame == "frame"

        def test_function_frame(function_frame):
            assert function_frame == "frame"

        @pytest.mark.render
        def test_child_process_render(child_frame):
            Path("marked-test-called").touch()

        def test_without_graphics():
            assert callable(mujoco.Renderer)
    """)

    portable = pytester.runpytest_subprocess("--no-render", "-q", "-p", "no:cacheprovider")
    portable.assert_outcomes(passed=1, skipped=4)
    assert not (pytester.path / "renderer-called").exists()
    assert not (pytester.path / "marked-test-called").exists()
    assert not (pytester.path / "child-process-called").exists()

    full = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
    full.assert_outcomes(passed=5)
    assert (pytester.path / "renderer-called").exists()
    assert (pytester.path / "marked-test-called").exists()
    assert (pytester.path / "child-process-called").exists()


def test_no_render_precedes_plugin_session_fixture(pytester, monkeypatch):
    # Nested pytest runs must not inherit the parent CI shard selection.
    monkeypatch.delenv("ARRIVAL_TEST_SHARD", raising=False)
    pytester.makeconftest((REPO / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(mujoco="""
        from pathlib import Path

        def Renderer():
            Path("renderer-called").touch()
            return "frame"
    """, early_render="""
        import mujoco
        import pytest

        @pytest.fixture(scope="session", autouse=True)
        def plugin_frame():
            return mujoco.Renderer()
    """, test_plugin="""
        def test_frame(plugin_frame):
            assert plugin_frame == "frame"
    """)
    portable = pytester.runpytest_subprocess("--no-render", "-p", "early_render", "-q")
    portable.assert_outcomes(skipped=1)
    assert not (pytester.path / "renderer-called").exists()

    full = pytester.runpytest_subprocess("-p", "early_render", "-q")
    full.assert_outcomes(passed=1)
    assert (pytester.path / "renderer-called").exists()
