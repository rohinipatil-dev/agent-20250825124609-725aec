import os
from typing import List, Dict

import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()


def get_system_prompt() -> str:
    return (
        "You are a knowledgeable, friendly Python programming assistant. "
        "Provide clear, accurate explanations and code examples following PEP 8 style. "
        "When appropriate, include short runnable examples and explain edge cases. "
        "If a question is ambiguous, ask for clarification. "
        "Prefer standard library solutions where possible."
    )


def build_api_messages(history: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}] + history


def generate_response(
    history: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    messages = build_api_messages(history, get_system_prompt())
    response = client.chat.completions.create(
        model=model,  # "gpt-4" or "gpt-3.5-turbo"
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}


def sidebar_controls():
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Use gpt-4 for best reasoning quality; gpt-3.5-turbo for speed."
    )
    temperature = st.sidebar.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values are more focused and deterministic."
    )
    max_tokens = st.sidebar.slider(
        "Max tokens in response",
        min_value=256,
        max_value=2048,
        value=800,
        step=64,
        help="Upper bound on the response length."
    )
    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("This app uses your OPENAI_API_KEY from the environment.")
    return model, temperature, max_tokens


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main():
    st.set_page_config(page_title="Python Programming Chatbot", page_icon="🐍", layout="wide")
    st.title("🐍 Python Programming Chatbot")
    st.caption("Ask questions about Python. Get concise explanations and runnable examples.")

    init_session_state()
    model, temperature, max_tokens = sidebar_controls()

    render_chat_history()

    user_input = st.chat_input("Type your Python question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            try:
                answer = generate_response(
                    history=st.session_state.messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response_container.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Error generating response. Ensure OPENAI_API_KEY is set in the environment.")
                st.exception(e)


if __name__ == "__main__":
    main()