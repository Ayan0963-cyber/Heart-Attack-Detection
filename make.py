import streamlit as st

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Ayan Chatbot", page_icon="🤖", layout="wide")

# ---- SESSION STATE INITIALIZATION ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ---- TITLE ----
st.title("💬 Ayan Chatbot")
st.write("Ask me anything, and I'll try to help!")

# ---- CHAT DISPLAY ----
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 You:** {msg['content']}")
    else:
        st.markdown(f"🤖 Bot:** {msg['content']}")

# ---- INPUT BOX ----
user_input = st.text_input(
    "Your message:",
    key="user_input",
    placeholder="Type your message here..."
)

# ---- SEND BUTTON ----
if st.button("Send", type="primary"):
    if user_input.strip():
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # --- BOT RESPONSE LOGIC ---
        bot_reply = "This is where your AI logic will go."
        # Example: You can replace with your ML model prediction or API call

        st.session_state.messages.append({"role": "bot", "content": bot_reply})

        # Clear input field after sending
        st.session_state.user_input = ""

# ---- RESET CHAT ----
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.user_input = ""
