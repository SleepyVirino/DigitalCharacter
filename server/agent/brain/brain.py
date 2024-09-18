from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from server.agent.brain.memory import MemoryModule, MemoryManager
from server.agent.brain.decide import DecisionEngine
from server.agent.brain.embedding import EmbeddingModel
from server.agent.brain.utils import (
    organize_work_memory,
    organize_work_memories,
    organize_episodic_memories
)

from server.agent.config import (
    OPENAI_KEY,
    OPENAI_BASE_URL
)
from server.agent.prompt import (
    SYS_AGENT_PROMPT
)


# 主控制器类，负责调度各个子模块
class BrainModule:
    def __init__(self, llm, embedding_model, work_memory_path, episodic_memory_path):
        self.llm = llm
        self.memory_manager = MemoryManager(work_memory_path, episodic_memory_path, embedding_model, llm)
        self.decision_engine = DecisionEngine()
        self.tools_manager = None

    def response(self, user_input):
        """
        相应用户的多模态信息
        :param user_input: 多模态信息
        :return:
        """
        chain_input = dict()
        chain_input["text"] = user_input["text"]

        memory = self.memory_manager.generate_memory(user_input["text"])
        chain_input["work_memory"] = organize_work_memories(memory["work_memory"])
        chain_input["episodic_memory"] = organize_episodic_memories(memory["episodic_memory"])
        chain_input["user"] = user_input["src"]
        chain_input["name"] = user_input["name"]

        print(SYS_AGENT_PROMPT.invoke(chain_input))
        response = (SYS_AGENT_PROMPT | self.llm).invoke(chain_input)

        new_work_memory = [
            {
                "role": "USER",
                "text": user_input["text"],
                "time": user_input["time"],
                "place": user_input["place"],
                "src": user_input["src"],
                "dest": user_input["dest"],
            },
            {
                "role": "AI",
                "text": response.content,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "place": user_input["place"],
                "src": user_input["dest"],
                "dest": user_input["src"],
            }
        ]
        for memory in new_work_memory:
            memory["event"] = organize_work_memory(memory)

        self.memory_manager.record_batch(new_work_memory)

        self.memory_manager.learn()

        return response


if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = EmbeddingModel(embedding_model)

    work_memory_path = "memory/work_memory.json"
    episodic_memory_path = "memory/episodic_memory.json"

    brain = BrainModule(llm, embedding_model, work_memory_path, episodic_memory_path)
    while True:
        user_input = {
            "text": input(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "place": "coffee shop",
            "src": "Du Yaoda",
            "dest": "You",
            "name": "Albert Einstein"
        }
        response = brain.response(user_input)
        print(response)
