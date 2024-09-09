import json
from datetime import datetime, timedelta

from openai import embeddings


class MemoryModule:
    def __init__(self):
        pass

    def store(self, memory):
        """
        将新记忆存储到对应的记忆库中。
        """
        pass

    def retrieve(self, query):
        """
        根据查询文本检索相关记忆。
        """
        pass

    def delete(self, memory):
        """
        删除记忆
        """
        pass


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class WorkMemory(MemoryModule):
    def __init__(self, path, embedding_model):
        super().__init__()
        self.path = path
        self.embedding_model = embedding_model
        self.memory = json.load(open(path, 'r'))
        # Load embeddings from the stored memory

        self.embeddings = np.array([entry['embedding'] for entry in self.memory])

    def store(self, memory):
        # Compute the embedding for the new memory entry
        memory["embedding"] = self.embedding_model.encode(memory["text"])
        self.memory.append(memory)
        if self.embeddings.size == 0:
            self.embeddings = np.array([memory["embedding"]])
        else:
            self.embeddings = np.vstack([self.embeddings, memory["embedding"]])
        json.dump(self.memory, open(self.path, 'w'))

    def retrieve(self, query, k=3, similarity_threshold=0.8):
        query_embedding = self.embedding_model.encode(query)
        # Compute cosine similarity between the query embedding and stored embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get the indices of the entries with similarity above the threshold
        above_threshold_indices = np.where(similarities >= similarity_threshold)[0]

        if len(above_threshold_indices) == 0:
            return []  # Return an empty list if no similarities meet the threshold

        # Sort the similarities and get the top k indices
        top_k_indices = above_threshold_indices[np.argsort(similarities[above_threshold_indices])[-k:][::-1]]
        result = [self.memory[idx] for idx in top_k_indices]
        return sorted(result, key=lambda x: x['time'])

    def delete(self, date):
        self.memory = [m for m in self.memory if
                       datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S") >= date]
        json.dump(self.memory, open(self.path, 'w'))


class MemoryManager:
    def __init__(self, work_memory_path, embedding_model):
        self.summary = None
        self.embedding_model = embedding_model
        self.work_memory = WorkMemory(work_memory_path, embedding_model)
        self.episodic_memory = None
        self.semantic_memory = None
        pass

    def record(self, work_memory):
        """
        记录工作记忆
        :return:
        """
        self.work_memory.store(work_memory)

    def record_batch(self, work_memory_list):
        for work_memory in work_memory_list:
            self.record(work_memory)

    def learn(self):
        """
        短期工作记忆学习为情景记忆, 语义记忆不变
        :return:
        """
        pass

    def forget(self):
        """
        情景记忆和短期工作记忆会遗忘，语义记忆不变
        :return:
        """
        work_memory_threshold = datetime.now() - timedelta(days=30)
        self.work_memory.delete(work_memory_threshold)

    def self_monitor(self):
        """
        自我监控理论更新对世界的总结
        :return:
        """
        pass

    def generate_memory(self, query):
        memory = {
            "work_memory": self.work_memory.retrieve(query),
        }
        return memory


if __name__ == '__main__':
    from langchain_openai import OpenAIEmbeddings
    from embedding import EmbeddingModel
    from server.agent.config import OPENAI_KEY, OPENAI_BASE_URL

    embedding_model = OpenAIEmbeddings(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = EmbeddingModel(embedding_model)
    work_memory = WorkMemory(path='memory/work_memory.json', embedding_model=embedding_model)

    memory = {
        "role": "USER",
        "text": "Who are you?",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    work_memory.store(memory)
