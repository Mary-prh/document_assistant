from backend.core2 import run_llm
import streamlit as st
from streamlit_chat import message
from typing import Set

# streamlit run c:/Users/Mary/Documents/document_assistant/main.py

st.set_page_config(layout="wide")
st.header("Baby Care Assistant Bot")

# Initialize session state variables
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to create sources string
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# Add custom CSS to make the chat history scrollable and fix the input at the bottom
st.markdown("""
    <style>
    .chat-container {
        height: calc(50vh - 50px); /* Adjust this value to fit your needs */
        overflow-y: auto;
        padding-bottom: 50px; /* To ensure the input box does not overlap the chat */
        margin-top: 1px; /* Adjust this value to change the space between the header and the chat */
    }
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 10px 0;
        z-index: 1000; /* Ensure the input container stays on top */
    }
    </style>
""", unsafe_allow_html=True)

# Display chat history in a scrollable container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history first
if st.session_state["chat_answers_history"]:
    for i, (generated_response, user_query) in enumerate(zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    )):
        message(user_query, is_user=True, key=f"user_{i}")
        message(generated_response, key=f"bot_{i}")
st.markdown('</div>', unsafe_allow_html=True)

# Fixed input container at the bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)

with st.form(key='input_form', clear_on_submit=True):
    prompt = st.text_input("Prompt", placeholder="How can I assist you today?", key="prompt")
    submit_button = st.form_submit_button(label='Send')

if submit_button and prompt:
        with st.spinner("Generating response.."):
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )
            sources = set(
                [doc.metadata["source"] for doc in generated_response["source_documents"]]
            )

            formatted_response = (
                f"{generated_response['result']}"
            )

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", generated_response["result"]))

            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
