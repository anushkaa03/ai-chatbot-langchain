import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("🤖 AI Chatbot using LangChain")

# ===============================
# PROMPT TEMPLATES (US-03)
# ===============================

general_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.
Answer the question clearly and simply:

{question}
"""
)

coding_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert Python developer.
Solve the problem step by step:

{question}
"""
)

fun_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a fun and friendly chatbot.
Reply in a casual and engaging way:

{question}
"""
)

# ===============================
# GROQ LLM
# ===============================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Settings")

prompt_type = st.sidebar.selectbox(
    "Choose Prompt Type",
    ["General", "Coding", "Fun"]
)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# ===============================
# CHAT MEMORY (US-04)
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ===============================
# CHAT INPUT
# ===============================
user_input = st.chat_input("Type your message...")

# ===============================
# MAIN CHAT LOGIC
# ===============================
if user_input:

    # store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.chat_message("user").write(user_input)

    # ===============================
    # SELECT PROMPT
    # ===============================
    if prompt_type == "General":
        final_prompt = general_prompt.format(question=user_input)

    elif prompt_type == "Coding":
        final_prompt = coding_prompt.format(question=user_input)

    else:
        final_prompt = fun_prompt.format(question=user_input)

    # ===============================
    # GROQ RESPONSE (US-02)
    # ===============================
    response = llm.invoke(final_prompt)

    st.chat_message("assistant").write(response.content)

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })
