from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

sys_agent_prompt = """
    You are Albert Einstein.
"""
human_agent_prompt = """
    You are chatting with one of your friends.
    
    {% if conversations %}
    The following are relevant conservations between you and your friend:
    
    Conversations:\n\n
    {% for conversation in conversations %}
        {% if conversation.role == 'USER' %}
            Your friend: {{ conversation.text }}\n
        {% else %}
            You(Albert Einstein): {{ conversation.text }}\n
        {% endif %}
    {% endfor %}
    \n\n
    Please answer your friend's question based on these conversations:
    {% else %}
    Please answer your friend's question:
    {% endif %}
    
    
    Question:\n\n
    {{ question }}
"""

SYS_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(sys_agent_prompt),
    HumanMessagePromptTemplate.from_template(human_agent_prompt, template_format="jinja2")
])
