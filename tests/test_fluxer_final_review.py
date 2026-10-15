from types import SimpleNamespace

import adapter
from gateway.config import PlatformConfig


def test_source_map_preserves_real_mention_sanitizer():
    ad = adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={"bot_token": "fake", "allow_all_users": True}))
    for text in ["@everyone @here", "<@123> <@&456>", "literal\u200b @everyone"]:
        sanitized, boundaries = adapter._sanitize_mentions_with_source_map(ad, text)
        assert sanitized == ad._sanitize_outbound_mentions(text)
        assert len(boundaries) == len(sanitized) + 1
        assert boundaries[-1] == len(text)


def test_prospective_turn_registered_before_thread_creation():
    ad = adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={"bot_token": "fake", "allow_all_users": True}))
    ad._threads_pending["anchor"] = "parent"
    source = SimpleNamespace(chat_id="anchor", chat_type="thread", parent_chat_id="parent", user_id="user")
    assert ad._register_active_auto_thread(source) == ("parent", "user", "anchor")
    assert ad._active_auto_thread_for_user("parent", "user") == "anchor"
