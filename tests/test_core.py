import json
from unittest.mock import MagicMock, patch

from chatbot.core import ChatSession, list_models


def test_add_user_message_appends_to_history():
    chat = ChatSession()
    chat.add_user_message("hello")
    assert len(chat.history) == 1
    assert chat.history[0].role == "user"
    assert chat.history[0].content == "hello"


def test_payload_messages_puts_system_prompt_first():
    chat = ChatSession(system_prompt="be terse")
    chat.add_user_message("hi")
    payload = chat._payload_messages()
    assert payload[0] == {"role": "system", "content": "be terse"}
    assert payload[1] == {"role": "user", "content": "hi"}


def test_history_is_trimmed_to_max_turns():
    chat = ChatSession(max_turns=2)
    for i in range(10):
        chat.add_user_message(f"message {i}")
    payload = chat._payload_messages()
    # system prompt + (max_turns * 2) most recent messages
    assert len(payload) == 1 + 4
    assert payload[1]["content"] == "message 6"
    assert payload[-1]["content"] == "message 9"


def _fake_stream_response(chunks):
    response = MagicMock()
    response.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]
    response.raise_for_status.return_value = None
    return response


def test_stream_reply_yields_tokens_and_updates_history():
    chunks = [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo!"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]
    with patch("chatbot.core.requests.post", return_value=_fake_stream_response(chunks)) as mock_post:
        chat = ChatSession()
        chat.add_user_message("hi")
        tokens = list(chat.stream_reply())

    assert tokens == ["Hel", "lo!"]
    assert chat.history[-1].role == "assistant"
    assert chat.history[-1].content == "Hello!"
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["stream"] is True


def test_reset_clears_history():
    chat = ChatSession()
    chat.add_user_message("hi")
    chat.reset()
    assert chat.history == []


def test_list_models_parses_tags_response():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"models": [{"name": "llama3.2"}, {"name": "phi3"}]}
    with patch("chatbot.core.requests.get", return_value=response):
        assert list_models() == ["llama3.2", "phi3"]
