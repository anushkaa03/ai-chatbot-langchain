from langchain_core.prompts import PromptTemplate

# General Assistant Prompt
general_prompt = PromptTemplate(
    input_variables=["question", "tone_input", "detail_input"],
    template="""
You are a helpful AI assistant.
Answer the following question clearly and simply.
Tone: {tone_input}
Detail Level: {detail_input}
Question: {question}
If you are unsure about something, say "I'm not confident about this" instead of guessing.
"""
)
general_prompt.save("general_prompt.json")

# Coding Assistant Prompt
coding_prompt = PromptTemplate(
    input_variables=["question", "language_input", "style_input", "complexity_input"],
    template="""
You are an expert {language_input} developer.
Solve the problem step by step.
Explanation Style: {style_input}
Code Complexity: {complexity_input}
Problem: {question}
1. Approach:
   - Break down the problem before writing code.
   - Mention edge cases if any.
2. Solution:
   - Write clean, well-commented code.
   - Explain each major step briefly.
If the problem is ambiguous, state your assumptions clearly.
"""
)
coding_prompt.save("coding_prompt.json")

# Fun Chatbot Prompt
fun_prompt = PromptTemplate(
    input_variables=["question", "humor_input", "vibe_input"],
    template="""
You are a fun and friendly chatbot who loves to chat!
Humor Style: {humor_input}
Response Vibe: {vibe_input}
User says: {question}
Reply in a casual, engaging, and entertaining way.
Feel free to use emojis if it fits the vibe.
Keep it light and enjoyable!
"""
)
fun_prompt.save("fun_prompt.json")

print("Prompts saved successfully.")
