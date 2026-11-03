"""Structured home channels must never become dictionary-shaped chat IDs."""
import os

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


@pytest.fixture(autouse=True)
def isolated_bridge(monkeypatch):
    # The bridge writes through os.environ directly, not monkeypatch.setenv.
    # Restore the whole mapping after each test, including newly seeded keys.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in tuple(os.environ):
        if key.startswith("FLUXER_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(fluxer_adapter, "_profile_scoped_config_load", lambda: False)


@pytest.mark.parametrize("channel", ["123456789012345678", 123456789012345678])
def test_structured_home_channel_seeds_only_chat_id(monkeypatch, channel):
    home = {"platform": "fluxer", "chat_id": channel, "name": "Notifications"}
    seed = fluxer_adapter._apply_yaml_config(
        {}, {"bot_token": "test-only", "home_channel": home}
    )

    assert seed is not None
    assert seed["home_channel"] == home
    assert os.environ["FLUXER_HOME_CHANNEL"] == str(channel)
    assert os.environ["FLUXER_HOME_CHANNEL_NAME"] == "Notifications"
    env_seed = fluxer_adapter._env_enablement()
    assert env_seed is not None
    assert env_seed["home_channel"] == {
        "chat_id": str(channel), "name": "Notifications"
    }
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=seed))
    assert adapter._home_channel_ids == {str(channel)}
    assert adapter._known_channel_ids == {str(channel)}


def test_structured_home_channel_preserves_environment_precedence(monkeypatch):
    monkeypatch.setenv("FLUXER_HOME_CHANNEL", "223456789012345678")
    monkeypatch.setenv("FLUXER_HOME_CHANNEL_NAME", "Environment home")
    fluxer_adapter._apply_yaml_config(
        {}, {"bot_token": "test-only", "home_channel": {
            "platform": "fluxer", "chat_id": "123456789012345678", "name": "YAML home"
        }}
    )
    env_seed = fluxer_adapter._env_enablement()
    assert env_seed is not None
    assert env_seed["home_channel"] == {
        "chat_id": "223456789012345678", "name": "Environment home"
    }


def test_structured_home_channel_remains_profile_scoped(monkeypatch):
    monkeypatch.setattr(fluxer_adapter, "_profile_scoped_config_load", lambda: True)
    home = {"platform": "fluxer", "chat_id": "123456789012345678", "name": "Home"}
    seed = fluxer_adapter._apply_yaml_config({}, {"home_channel": home})
    assert seed is not None
    assert seed["home_channel"] == home
    assert os.environ.get("FLUXER_HOME_CHANNEL") is None
    assert os.environ.get("FLUXER_HOME_CHANNEL_NAME") is None


@pytest.mark.parametrize("home", [{}, {"name": "Home"}, {"chat_id": None},
                                  {"chat_id": {"invalid": "nested"}}])
def test_incomplete_structured_home_does_not_poison_environment(home):
    fluxer_adapter._apply_yaml_config({}, {"home_channel": home})
    assert os.environ.get("FLUXER_HOME_CHANNEL") is None


def test_legacy_scalar_home_and_list_allowlists_still_bridge():
    seed = fluxer_adapter._apply_yaml_config({}, {
        "home_channel": "123456789012345678", "home_channel_name": "Home",
        "allowed_users": ["123456789012345679", "123456789012345680"],
        "allowed_channels": ["123456789012345678"],
    })
    assert seed is not None
    assert seed["home_channel"] == {"chat_id": "123456789012345678", "name": "Home"}
    assert os.environ["FLUXER_HOME_CHANNEL"] == "123456789012345678"
    assert os.environ["FLUXER_ALLOWED_USERS"] == "123456789012345679,123456789012345680"
    assert os.environ["FLUXER_ALLOWED_CHANNELS"] == "123456789012345678"
