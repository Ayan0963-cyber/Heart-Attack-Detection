# app.py
import streamlit as st
import os
import time

# Try to import OpenAI and transformers (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, set_seed
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---- UI setup ----
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("💬 General AI Chatbot")
st.write("Ask anything. Choose model: *OpenAI* (better) or *Local (distilgpt2)* (offline).")

# Sidebar controls
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Model", ("auto (OpenAI if key)", "OpenAI (requires key)", "Local (distilgpt2)"))
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
max_tokens = st.sidebar.slider("Max tokens (local only)", 50, 512, 150, 10)
clear_chat = st.sidebar.button("Clear chat history")

# Provide place for API key (optional)
st.sidebar.markdown("---")
st.sidebar.markdown("*OpenAI API Key* (optional, for better replies)")
openai_key = st.sidebar.text_input("Paste OpenAI API Key (sk-...)", type="password")

# Persist chat
if "history" not in st.session_state:
    # history is list of {"role": "user"/"assistant", "content": "..."}
    st.session_state.history = []

if clear_chat:
    st.session_state.history = []
    st.experimental_rerun()

# Initialize local generator if needed
def get_local_generator():
    if "local_gen" not in st.session_state:
        if not TRANSFORMERS_AVAILABLE:
            st.error("Transformers not installed. Add transformers and torch to requirements.")
            return None
        # create text-generation pipeline
        st.session_state.local_gen = pipeline("text-generation", model="distilgpt2")
        set_seed(42)
    return st.session_state.local_gen

# Function: call OpenAI chat completion
def call_openai_chat(history, api_key, temperature):
    if not OPENAI_AVAILABLE:
        st.error("openai library not installed. Add openai to requirements.")
        return "OpenAI library missing."
    if not api_key:
        return "No OpenAI API key provided."
    openai.api_key = api_key
    # transform history to OpenAI message format
    msgs = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        msgs.append({"role": role, "content": content})
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msgs,
            temperature=temperature,
            max_tokens=600,
            n=1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI request failed: {e}"

# Function: call local model
def call_local_model(history, gen, temperature, max_tokens):
    # build a prompt from recent history
    # We'll keep last 6 turns to keep prompt length small
    turns = history[-6:]
    prompt = ""
    for t in turns:
        role = t["role"]
        content = t["content"]
        if role == "user":
            prompt += f"User: {content}\n"
        else:
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant:"

    # generate
    try:
        out = gen(prompt, max_length=len(prompt.split()) + max_tokens, do_sample=True, temperature=temperature, num_return_sequences=1)
        text = out[0]["generated_text"]
        # After generation, extract assistant answer by removing prompt prefix
        if text.startswith(prompt):
            answer = text[len(prompt):].strip()
        else:
            # fallback: return whole text
            answer = text.strip()
        # Trim to first obvious end (if there are multiple lines where user might speak next)
        # Stop at "User:" if appears
        if "User:" in answer:
            answer = answer.split("User:")[0].strip()
        return answer
    except Exception as e:
        return f"Local model generation failed: {e}"

# Chat input
with st.container():
    user_input = st.text_input("You:", placeholder="Type a question and press Enter", key="user_input")

if user_input:
    # Save user message
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        # Decide model
        use_openai = False
        if model_choice == "OpenAI (requires key)":
            use_openai = True
        elif model_choice == "Local (distilgpt2)":
            use_openai = False
        else:  # auto
            use_openai = bool(openai_key)

        if use_openai:
            # Call OpenAI
            answer = call_openai_chat(st.session_state.history, openai_key, temperature)
        else:
            # Ensure local model available
            gen = get_local_generator()
            if gen is None:
                answer = "Local model not available. Install transformers and torch, or provide OpenAI key."
            else:
                answer = call_local_model(st.session_state.history, gen, temperature, max_tokens)

        st.session_state.history.append({"role": "assistant", "content": answer})
    # Clear the input box
    st.session_state.user_input = ""

# Display chat history
st.markdown("---")
chat_area = st.container()
with chat_area:
    for turn in st.session_state.history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            st.markdown(f"*You:* {content}")
        else:
            st.markdown(f"*Bot:* {content}")

# Footer
st.markdown("---")
st.write("Hints: 1) For best results use OpenAI (paste your key in the sidebar). 2) Local distilgpt2 is offline but lower quality.")
