from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from server.agent.brain.memory import MemoryManager
from server.agent.brain.decide import DecisionEngine
from server.agent.brain.embedding import EmbeddingModel
from server.agent.brain.utils import (
    organize_work_memory,
    organize_work_memories,
    construct_work_memory,
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
    def __init__(self, figure, llm, embedding_model, summary_path, work_memory_path, episodic_memory_path):
        self.figure = figure
        self.llm = llm
        self.memory_manager = MemoryManager(self.figure, summary_path, work_memory_path, episodic_memory_path,
                                            embedding_model, llm)
        self.decision_engine = DecisionEngine()
        self.tools_manager = None

    def response(self, user_input):
        """
        相应用户的多模态信息
        :param user_input: 多模态信息
        :return:
        """
        new_work_memory = construct_work_memory("USER", user_input["text"], user_input["time"], user_input["place"],
                                                user_input["src"], user_input["dest"])
        self.memory_manager.record(work_memory=new_work_memory)

        chain_input = dict()

        memory = self.memory_manager.generate_memory(user_input["text"])
        chain_input["summary"] = self.memory_manager.summary
        chain_input["work_memory"] = organize_work_memories(memory["work_memory"])
        chain_input["episodic_memory"] = organize_episodic_memories(memory["episodic_memory"])
        chain_input["user"] = user_input["src"]
        chain_input["name"] = user_input["name"]

        response = (SYS_AGENT_PROMPT | self.llm).invoke(chain_input)

        new_work_memory = construct_work_memory("AI", response.content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                user_input["place"], user_input["dest"], user_input["src"])
        self.memory_manager.record(work_memory=new_work_memory)


        self.memory_manager.learn()
        self.memory_manager.forget()
        self.memory_manager.self_monitor(user_input["text"])

        return response


if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = EmbeddingModel(embedding_model)

    work_memory_path = "memory/work_memory.json"
    episodic_memory_path = "memory/episodic_memory.json"
    summary_path = "memory/summary.txt"

    figure = "Albert Einstein"

    brain = BrainModule(figure, llm, embedding_model, summary_path, work_memory_path, episodic_memory_path)
    while True:
        user_input = {
            "text": input(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "place": "coffee shop",
            "src": "Du Yaokang",
            "dest": "You",
            "name": figure
        }
        response = brain.response(user_input)
        print(response.content)
