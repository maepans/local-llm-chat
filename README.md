# Local LLM Chat

A small chatbot that runs entirely on your machine using [Ollama](https://ollama.com)
and an open-source language model — no API keys, no usage cost, no data leaving
your computer.

The point of this project isn't to wrap a framework, it's to show how a chat
app actually works underneath one: how a conversation is represented as a list
of role/content messages, how the context window is kept bounded as the chat
grows, and how a streaming response is parsed token by token. `chatbot/core.py`
talks to Ollama's REST API directly with `requests` — no LangChain, no SDK.

## Features

- Chat UI built with [Streamlit](https://streamlit.io)
- Streaming responses (tokens appear as the model generates them)
- Configurable system prompt and model, picked from your locally installed models
- Bounded conversation history (`max_turns`) so context never grows unbounded
- Core chat logic is fully unit tested with the network mocked out — no Ollama
  required to run the test suite

## How it works

```
chatbot/core.py
├── Message         # one {role, content} entry: "system" | "user" | "assistant"
└── ChatSession
    ├── add_user_message()   # append a user turn to history
    ├── _trimmed_history()   # keep only the last N exchanges
    ├── _payload_messages()  # system prompt + trimmed history, as dicts
    └── stream_reply()       # POST to /api/chat, yield tokens as they arrive,
                              # then append the full reply to history
```

Ollama's `/api/chat` endpoint streams newline-delimited JSON — one small JSON
object per token/chunk, with `"done": true` on the last one. `stream_reply()`
parses that stream directly and yields plain text tokens, which the Streamlit
UI renders live with `st.write_stream`.

## Setup

1. Install [Ollama](https://ollama.com/download) and pull a small model:

   ```bash
   ollama pull llama3.2
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

Ollama needs to be running in the background (`ollama serve`, or it's already
running if you installed the desktop app). The sidebar lets you switch models,
edit the system prompt, and control how many exchanges the bot remembers.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests mock the HTTP calls to Ollama, so they run without any model installed.

## Why a local model

Everything here runs for free, indefinitely, on your own hardware — no API
key, no rate limit, no bill. Small instruct models like `llama3.2` (3B) or
`phi3` run comfortably on a laptop without a GPU. Swap `DEFAULT_MODEL` in
[`chatbot/core.py`](chatbot/core.py) or pick a different one from the sidebar
once you've pulled it with `ollama pull <model>`.

## License

MIT — see [LICENSE](LICENSE).
