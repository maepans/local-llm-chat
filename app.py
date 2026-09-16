import streamlit as st

from chatbot import ChatSession
from chatbot.core import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, list_models

st.set_page_config(page_title="Local LLM Chat", page_icon="💬")
st.title("💬 Local LLM Chat")
st.caption(
    "A small chatbot that runs entirely on your machine via Ollama — "
    "no API keys, no usage cost."
)

with st.sidebar:
    st.header("Settings")

    try:
        available_models = list_models()
    except Exception:
        available_models = []

    if available_models:
        model = st.selectbox("Model", available_models)
    else:
        model = st.text_input("Model", value=DEFAULT_MODEL)
        st.warning("Couldn't reach Ollama — make sure it's running (`ollama serve`).")

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=100)
    max_turns = st.slider("Exchanges to remember", min_value=1, max_value=30, value=12)

    if st.button("Clear conversation"):
        st.session_state.pop("chat_session", None)
        st.rerun()

needs_new_session = "chat_session" not in st.session_state or (
    st.session_state.chat_session.model != model
    or st.session_state.chat_session.system_prompt != system_prompt
    or st.session_state.chat_session.max_turns != max_turns
)
if needs_new_session:
    st.session_state.chat_session = ChatSession(
        model=model, system_prompt=system_prompt, max_turns=max_turns
    )

chat: ChatSession = st.session_state.chat_session

for message in chat.history:
    with st.chat_message(message.role):
        st.markdown(message.content)

if prompt := st.chat_input("Ask me anything..."):
    chat.add_user_message(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            st.write_stream(chat.stream_reply())
        except Exception as exc:
            st.error(f"Couldn't reach Ollama at {chat.host}: {exc}")
