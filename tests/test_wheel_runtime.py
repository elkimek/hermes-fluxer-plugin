"""Build the distributable away from the checkout and smoke its pure imports."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_wheel_ships_voice_runtime_without_bundling_host_namespaces(tmp_path):
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "adapter.py",
        "agent_tools.py",
        "kokoro_tts.py",
        "livekit_bridge.py",
        "xai_realtime.py",
    ):
        shutil.copyfile(root / name, source / name)
    shutil.copytree(
        root / "scripts",
        source / "scripts",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    (wheel_path,) = (source / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert "kokoro_tts.py" in names
        assert "agent_tools.py" in names
        assert "scripts/fluxer_voice_auto_join.py" in names
        assert "scripts/fluxer_stt_voice_loop.py" in names
        assert "scripts/fluxer_xai_room_loop.py" in names
        assert not any(
            name.startswith(("tools/", "gateway/", "hermes_cli/")) for name in names
        )
    # -I removes the checkout and PYTHONPATH: these must be actual wheel imports.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import agent_tools, kokoro_tts, livekit_bridge, xai_realtime; "
        "assert '.whl/' in agent_tools.__file__; "
        "assert '.whl/' in kokoro_tts.__file__; "
        "assert 'livekit' not in sys.modules"
    )
    smoked = subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheel_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert smoked.returncode == 0, smoked.stdout + smoked.stderr
