"""Public registrations must be declared for plugin discovery and audit."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_declares_the_thread_title_hook():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert manifest.get("provides_hooks") == ["pre_command"]


def test_manifest_declares_all_read_only_agent_tools():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert set(manifest.get("provides_tools", [])) == {
        "flux_recall", "flux_search", "flux_members", "flux_audit",
    }
