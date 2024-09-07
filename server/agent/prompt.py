from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

sys_agent_prompt = """
    You are Albert Einstein.
"""
human_agent_prompt = """
    You need to answer the following questions:
    {{ text }}
"""

SYS_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(sys_agent_prompt),
    HumanMessagePromptTemplate.from_template(human_agent_prompt, template_format="jinja2")
])
