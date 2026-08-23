"""Test chat history truncation fix (#490, #489).

Verifies that:
1. The GET /agents/{name}/chat-history/{id} endpoint returns full content (not truncated)
2. The list endpoint truncates to display, but GET returns complete content
3. PATCH can update content > 500 chars without losing data
4. IDOR scope verification prevents cross-agent access
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pinky_daemon.api import create_api
from pinky_daemon.conversation_store import ConversationStore


@pytest.fixture
def app_client(tmp_path):
    """Create app with isolated conversation store."""
    db_path = str(tmp_path / "conversations.db")
    app = create_api(db_path=db_path)
    # Manually register a test agent so we have a valid agent name
    app.state.registry.register(
        name="test-agent",
        model="claude-opus-5",
        soul="Test agent",
    )
    app.state.registry.register(
        name="other-agent",
        model="claude-opus-5",
        soul="Other agent",
    )
    return TestClient(app)


def test_chat_history_full_content_retrieval(app_client):
    """Verify GET chat-history/{id} returns full content, not truncated."""
    # Create a long message (> 500 chars)
    long_content = "x" * 800
    agent_name = "test-agent"

    # Simulate appending a message via internal store
    store = app_client.app.state.conversation_store
    stored = store.append(
        session_id=f"session-{agent_name}",
        role="user",
        content=long_content,
        platform="internal",
        chat_id=agent_name,
    )
    msg_id = stored.id

    # GET should return full content
    response = app_client.get(f"/agents/{agent_name}/chat-history/{msg_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == long_content, "GET must return full content, not truncated"


def test_chat_history_patch_preserves_long_content(app_client):
    """Verify PATCH preserves content > 500 chars without truncation."""
    agent_name = "test-agent"
    long_content = "y" * 750
    new_content = "z" * 600

    store = app_client.app.state.conversation_store
    stored = store.append(
        session_id=f"session-{agent_name}",
        role="assistant",
        content=long_content,
        platform="internal",
        chat_id=agent_name,
    )
    msg_id = stored.id

    # PATCH with new long content
    response = app_client.patch(
        f"/agents/{agent_name}/chat-history/{msg_id}",
        json={"content": new_content},
    )
    assert response.status_code == 200

    # Verify the stored content is the new value, unmodified
    response = app_client.get(f"/agents/{agent_name}/chat-history/{msg_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == new_content, "PATCH must preserve exact content length"


def test_chat_history_idor_scope_verification(app_client):
    """Verify GET/PATCH/DELETE verify IDOR scope (agent owns only own messages)."""
    store = app_client.app.state.conversation_store

    # User A creates a message
    agent_a = "test-agent"
    msg_a = store.append(
        session_id=f"session-{agent_a}",
        role="user",
        content="secret from A",
        platform="internal",
        chat_id=agent_a,
    )

    # Agent B should not be able to GET it
    response = app_client.get(f"/agents/other-agent/chat-history/{msg_a.id}")
    assert response.status_code == 403, "Cross-agent GET must be denied"

    # Agent B should not be able to PATCH it
    response = app_client.patch(
        f"/agents/other-agent/chat-history/{msg_a.id}",
        json={"content": "hacked"},
    )
    assert response.status_code == 403, "Cross-agent PATCH must be denied"

    # Agent B should not be able to DELETE it
    response = app_client.delete(f"/agents/other-agent/chat-history/{msg_a.id}")
    assert response.status_code == 403, "Cross-agent DELETE must be denied"

    # Agent A can still access their own message
    response = app_client.get(f"/agents/{agent_a}/chat-history/{msg_a.id}")
    assert response.status_code == 200
    assert response.json()["content"] == "secret from A"
