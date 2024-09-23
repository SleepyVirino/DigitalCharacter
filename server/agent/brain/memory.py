import json

from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN

from server.agent.prompt import SUMMARY_PROMPT, REFLECT_PROMPT
from server.agent.brain.utils import organize_work_memories, organize_episodic_memories

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


class WorkMemory(MemoryModule):
    def __init__(self, path, embedding_model):
        super().__init__()
        self.path = path
        self.embedding_model = embedding_model
        self.memory = json.load(open(path, 'r', encoding='utf-8'))
        # Load embeddings from the stored memory

        self.embeddings = np.array([entry['embedding'] for entry in self.memory])

    def store(self, memory):
        # Compute the embedding for the new memory entry
        memory["embedding"] = self.embedding_model.encode(memory["event"])
        self.memory.append(memory)
        if self.embeddings.size == 0:
            self.embeddings = np.array([memory["embedding"]])
        else:
            self.embeddings = np.vstack([self.embeddings, memory["embedding"]])
        json.dump(self.memory, open(self.path, 'w', encoding='utf-8'))

    def retrieve(self, hours=3, limit=5):
        if not len(self.memory):
            return []

        date = datetime.now() - timedelta(hours=hours)
        result = [m for m in self.memory if
                  datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S") >= date]
        return sorted(result, key=lambda x: x['time'])[-limit:]

    def delete(self, date):
        self.memory = [m for m in self.memory if
                       datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S") >= date]
        json.dump(self.memory, open(self.path, 'w', encoding='utf-8'))


class EpisodicMemory(MemoryModule):
    def __init__(self, path, embedding_model):
        super().__init__()
        self.path = path
        self.embedding_model = embedding_model
        self.memory = json.load(open(path, 'r', encoding='utf-8'))
        # Load embeddings from the stored memory

        self.embeddings = np.array([entry['embedding'] for entry in self.memory])

    def store(self, memory):
        # Compute the embedding for the new memory entry
        memory["embedding"] = self.embedding_model.encode(memory["event"])
        self.memory.append(memory)
        if self.embeddings.size == 0:
            self.embeddings = np.array([memory["embedding"]])
        else:
            self.embeddings = np.vstack([self.embeddings, memory["embedding"]])
        json.dump(self.memory, open(self.path, 'w', encoding='utf-8'))

    def retrieve(self, query, k=3, similarity_threshold=0.5):
        if not len(self.memory):
            return []
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
        json.dump(self.memory, open(self.path, 'w', encoding='utf-8'))


class MemoryManager:
    def __init__(self, figure, summary_path, work_memory_path, episodic_memory_path, embedding_model, llm):
        self.figure = figure
        self.summary = open(summary_path, 'r', encoding="utf-8").read()
        self.summary_path = summary_path
        self.llm = llm
        self.embedding_model = embedding_model
        self.work_memory = WorkMemory(work_memory_path, embedding_model)
        self.episodic_memory = EpisodicMemory(episodic_memory_path, embedding_model)
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

    def learn(self, eps=0.5, min_samples=2):
        """
        将工作记忆聚类后，提炼为情景记忆
        :param eps: DBSCAN中的epsilon参数，决定聚类的半径
        :param min_samples: DBSCAN中的最小样本数，决定形成聚类的最低要求
        """
        # 获取所有工作记忆的嵌入向量
        embeddings = np.array([memory['embedding'] for memory in self.work_memory.memory])

        if len(embeddings) == 0:
            return  # 没有工作记忆时，直接返回

        # 使用DBSCAN对嵌入向量进行聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        labels = clustering.labels_

        # 对每个聚类生成情景记忆
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                # 噪声记忆，不做处理
                continue

            # 选出属于当前聚类的工作记忆
            clustered_memories = [self.work_memory.memory[i] for i in range(len(labels)) if labels[i] == label]

            # 根据聚类的工作记忆生成一个情景记忆
            episodic_memory_text = self._summarize_cluster(clustered_memories)

            # 将情景记忆存储
            episodic_memory = {
                "role": "SYSTEM",
                "event": episodic_memory_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "embedding": self.embedding_model.encode(episodic_memory_text)
            }
            self.episodic_memory.store(episodic_memory)

        # 删除已经学习到的工作记忆
        # self._delete_learned_work_memories(labels)

    def _summarize_cluster(self, clustered_memories):
        """
        对一个聚类的工作记忆进行总结，生成情景记忆文本
        :param clustered_memories: 属于同一个聚类的工作记忆
        :return: 生成的情景记忆文本
        """
        # 获取聚类中所有记忆的文本
        text = organize_work_memories(clustered_memories)

        # 调用 LLM 来生成摘要
        chain_input = {
            "name": self.figure,
            "events": text,
        }
        response = (SUMMARY_PROMPT | self.llm).invoke(chain_input)

        # 返回 LLM 生成的摘要
        return response.content

    def _delete_learned_work_memories(self, labels):
        """
        删除已经学习并转化为情景记忆的工作记忆
        :param labels: 聚类算法对每个记忆的标签
        """
        # 删除所有被分到聚类中的工作记忆，保留噪声(-1标签)的记忆
        self.work_memory.memory = [self.work_memory.memory[i] for i in range(len(labels)) if labels[i] == -1]
        json.dump(self.work_memory.memory, open(self.work_memory.path, 'w', encoding="utf-8"))

    def forget(self):
        """
        情景记忆和短期工作记忆会遗忘，语义记忆不变
        :return:
        """
        work_memory_threshold = datetime.now() - timedelta(days=30)
        self.work_memory.delete(work_memory_threshold)
        self.episodic_memory.delete(work_memory_threshold)

    def self_monitor(self, query):
        """
        自我监控理论更新对世界的总结
        :return:
        """
        memory = self.generate_memory(query)
        chain_input = {"summary": self.summary, "name": self.figure,
                       "work_memory": organize_work_memories(memory["work_memory"]),
                       "episodic_memory": organize_episodic_memories(memory["episodic_memory"])}
        response = (REFLECT_PROMPT | self.llm).invoke(chain_input)
        self.summary = response.content
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(self.summary)

    def generate_memory(self, query):
        memory = {
            "work_memory": self.work_memory.retrieve(),
            "episodic_memory": self.episodic_memory.retrieve(query),
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
