import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("AI Chatbot")

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

user_input = st.text_input("Enter your question")

if user_input:
    response = llm.invoke([
        HumanMessage(content=user_input)
    ])

    st.write(response.content)