import streamlit as st
import httpx

# Streamlit App Config
st.set_page_config(page_title="AI ChatBot Rumah Sakit", layout="centered")
st.title("AI Customer Service Rumah Sakit")

# Backend API URL
API_URL = "http://localhost:8000/chat"

# Session State for Context and Messages
# if "context" not in st.session_state:
#     st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []  # To store chat history

# Chat UI
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "bot":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Masukkan pertanyaan Anda di sini..."):
    # Append the user's message to the chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send request to FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    API_URL,
                    # json={"context": st.session_state.context, "query": user_input}
                    json={"query": user_input},
                    timeout=200
                )
                response_data = response.json()
                bot_response = response_data["response"]
                # st.session_state.context = response_data["context"]
                st.session_state.messages.append({"role": "bot", "content": bot_response})
                st.markdown(bot_response)
            except Exception as e:
                st.error(f"Error: {e}")
