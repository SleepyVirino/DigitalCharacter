from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from server.agent.brain.memory import MemoryModule, MemoryManager
from server.agent.brain.decide import DecisionEngine
from server.agent.brain.embedding import EmbeddingModel

from server.agent.config import (
    OPENAI_KEY,
    OPENAI_BASE_URL
)
from server.agent.prompt import (
    SYS_AGENT_PROMPT
)


# 主控制器类，负责调度各个子模块
class BrainModule:
    def __init__(self, llm, embedding_model,work_memory_path):
        self.llm = llm
        self.memory_manager = MemoryManager(work_memory_path,embedding_model)
        self.decision_engine = DecisionEngine()
        self.tools_manager = None

    def response(self, user_input):
        """
        相应用户的多模态信息
        :param user_input: 多模态信息
        :return:
        """
        chain_input = dict()
        chain_input["question"] = user_input["question"]

        memory = self.memory_manager.generate_memory(user_input["question"])
        chain_input["conversations"] = memory["work_memory"]

        response = (SYS_AGENT_PROMPT | self.llm).invoke(chain_input)

        new_work_memory = [
            {
                "role": "USER",
                "text": user_input["question"],
                "time": user_input["time"],
            },
            {
                "role": "AI",
                "text": response.content,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
        self.memory_manager.record_batch(new_work_memory)

        return response


if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = EmbeddingModel(embedding_model)

    work_memory_path = "memory/work_memory.json"

    brain = BrainModule(llm,embedding_model,work_memory_path)
    user_input = {
        "question": "Do you know gpt?",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    response = brain.response(user_input)
    print(response)
