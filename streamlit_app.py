import streamlit as st
from chatbot import ChatbotAgent
from dotenv import load_dotenv
import os
from message_history import extract_conversation_by_session  # Use our custom function

# Load environment variables from .env
load_dotenv()

os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("FN (Finance + News) Chat")

# Create a temporary ChatbotAgent to fetch existing session IDs.
temp_agent = ChatbotAgent()
existing_sessions = temp_agent.agent_storage.get_all_session_ids()  # returns a list of session IDs
session_options = ["New Session"] + existing_sessions

# Sidebar: let the user select an existing session or start a new one.
selected_session = st.sidebar.selectbox("Select a session", session_options, key="selected_session")

# Use a separate key to track previous selection.
if "prev_selected_session" not in st.session_state or st.session_state.prev_selected_session != selected_session:
    st.session_state.prev_selected_session = selected_session

    chatbot = ChatbotAgent()
    # Override get_session_id so that it uses the selected session from the sidebar.
    chatbot.get_session_id = lambda: None if selected_session == "New Session" else selected_session
    chatbot.create_agent()
    st.session_state.chatbot_agent = chatbot.agent

    # Load chat history using our custom function.
    if selected_session == "New Session":
        st.session_state.chat_history = []
    else:
        st.session_state.chat_history = extract_conversation_by_session(
            db_path="tmp/chat.db", session_id=selected_session
        )

# Display chat history using st.chat_message.
for chat in st.session_state.chat_history:
    if chat["message"] is not None:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

# Use st.chat_input for user messages.
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append and display the user's message.
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get the chatbot response (assuming agent.run returns an object with a .content attribute).
    response = st.session_state.chatbot_agent.run(user_input)
    bot_message = response.content

    # Append and display the assistant's message.
    st.session_state.chat_history.append({"role": "assistant", "message": bot_message})
    with st.chat_message("assistant"):
        st.markdown(bot_message)
