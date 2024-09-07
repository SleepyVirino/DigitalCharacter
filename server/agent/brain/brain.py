from langchain_openai import ChatOpenAI

from server.agent.brain.memory import MemoryModule
from server.agent.brain.decide import DecisionEngine

from server.agent.config import (
    OPENAI_KEY,
    OPENAI_BASE_URL
)
from server.agent.prompt import (
    SYS_AGENT_PROMPT
)


# 主控制器类，负责调度各个子模块
class BrainModule:
    def __init__(self, llm):
        self.llm = llm
        self.memory_module = MemoryModule()
        self.decision_engine = DecisionEngine()
        self.tools_manager = None

    def response(self, user_input):
        """
        相应用户的多模态信息
        :param user_input: 多模态信息
        :return:
        """

        return (SYS_AGENT_PROMPT | self.llm).invoke(user_input)


if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=OPENAI_KEY,base_url=OPENAI_BASE_URL)
    brain = BrainModule(llm)
    user_input = {
        "text": "Do you know chatgpt?",
    }
    response = brain.response(user_input)
    print(response.content)
