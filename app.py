from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
import streamlit as st
import json
import os

load_dotenv()

HISTORY_FILE = 'history.txt'

#langchain model selection code
model = ChatGroq(model='llama-3.3-70b-versatile', streaming=True)

#PromptTemplate loading code as in from prompt_generator
general_prompt = load_prompt('general_prompt.json')
coding_prompt  = load_prompt('coding_prompt.json')
fun_prompt     = load_prompt('fun_prompt.json')

#ChatPromptTemplate 
chat_template = ChatPromptTemplate([
    ('system', 'You are a experienced AI assistant(generalized Topics) but if the user ask specific Topic give it specific answers itself and If you are unsure about something, say "I\'m not confident about this" instead of guessing.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])
# Langchain LCEL
chain = chat_template | model

#streamlit code
st.set_page_config(page_title='AI Chatbot', layout='centered')
st.title('AI Chatbot')


if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    if os.path.exists(HISTORY_FILE):
        for line in open(HISTORY_FILE):
            entry = json.loads(line)
            st.session_state.messages.append({'role': 'user',      'content': entry['HumanMessage']})
            st.session_state.messages.append({'role': 'assistant',  'content': entry['AIMessage']})
            st.session_state.chat_history.append(HumanMessage(content=entry['prompt']))
            st.session_state.chat_history.append(AIMessage(content=entry['AIMessage']))
st.sidebar.title('Settings')
prompt_type = st.sidebar.selectbox('Choose Prompt Type', ['General', 'Coding', 'Fun'])
if prompt_type == 'General':
    tone_input   = st.sidebar.selectbox('Tone', ['Formal', 'Casual', 'Technical'])
    detail_input = st.sidebar.selectbox('Detail Level', ['Brief', 'Moderate', 'Detailed'])
elif prompt_type == 'Coding':
    language_input   = st.sidebar.selectbox('Language', ['Python', 'JavaScript', 'C++', 'Java'])
    style_input      = st.sidebar.selectbox('Style', ['Beginner-friendly', 'Advanced'])
    complexity_input = st.sidebar.selectbox('Complexity', ['Simple', 'Optimized'])
else:
    humor_input = st.sidebar.selectbox('Humor Style', ['Witty', 'Silly', 'Dry'])
    vibe_input  = st.sidebar.selectbox('Vibe', ['Chill', 'Hyper', 'Sarcastic'])
if st.sidebar.button('Clear Chat'):
    st.session_state.messages = []
    st.session_state.chat_history = []
    open(HISTORY_FILE, 'w').close()
    st.rerun()
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
user_input = st.chat_input('Type your message!')
if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    st.chat_message('user').write(user_input)
    if prompt_type == 'General':
        final_prompt = general_prompt.format(question=user_input, tone_input=tone_input, detail_input=detail_input)
    elif prompt_type == 'Coding':
        final_prompt = coding_prompt.format(question=user_input, language_input=language_input, style_input=style_input, complexity_input=complexity_input)
    else:
        final_prompt = fun_prompt.format(question=user_input, humor_input=humor_input, vibe_input=vibe_input)
    try:
        with st.chat_message('assistant'):
            full_response = st.write_stream(
                chunk.content for chunk in chain.stream({
                    'input': final_prompt,
                    'chat_history': st.session_state.chat_history[-10:]
                })
            )
        st.session_state.chat_history.append(HumanMessage(content=final_prompt))
        st.session_state.chat_history.append(AIMessage(content=full_response))
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})
        with open(HISTORY_FILE, 'a') as f:
            f.write(json.dumps({'HumanMessage': user_input, 'prompt': final_prompt, 'AIMessage': full_response}) + '\n')
    except Exception as e:
        st.error(f'API Error: {str(e)}')
