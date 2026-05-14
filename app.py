import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("🤖 AI Chatbot using LangChain")

# ===============================
# LLM SETUP
# ===============================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    streaming=True
)

# ===============================
# PROMPT TEMPLATES
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
# CHAIN SETUP
# ===============================
chat_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = chat_prompt | llm

# ===============================
# SESSION STATE
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    st.session_state.chat_history = []

# ===============================
# CHAT DISPLAY
# ===============================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ===============================
# CHAT INPUT & RESPONSE
# ===============================
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if prompt_type == "General":
        final_prompt = general_prompt.format(question=user_input)
    elif prompt_type == "Coding":
        final_prompt = coding_prompt.format(question=user_input)
    else:
        final_prompt = fun_prompt.format(question=user_input)

    # Limit history to last 10 messages to reduce token usage
    limited_history = st.session_state.chat_history[-10:]

    try:
        with st.chat_message("assistant"):
            full_response = st.write_stream(
                chunk.content for chunk in chain.stream({
                    "input": final_prompt,
                    "history": limited_history
                })
            )

        st.session_state.chat_history.append(HumanMessage(content=final_prompt))
        st.session_state.chat_history.append(AIMessage(content=full_response))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"API Error: {str(e)}")
