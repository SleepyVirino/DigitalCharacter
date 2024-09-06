import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MemoryModule:
    def __init__(self):
        self.episodic_memory = []
        self.semantic_memory = []

    def _embed_query(self, query):
        # 将查询转化为嵌入向量，这里简化为模拟数据
        return np.random.rand(1, 512)

    def _search_memory(self, query):
        query_vec = self._embed_query(query)
        memory_vectors = np.array([mem['embedding'] for mem in self.episodic_memory])

        similarities = cosine_similarity(query_vec, memory_vectors)
        best_match = np.argmax(similarities)
        return self.episodic_memory[best_match] if similarities[0][best_match] > 0.8 else None

    def store_memory(self, new_memory):
        self.episodic_memory.append(new_memory)
        self._summarize_and_forget()

    def _summarize_and_forget(self):
        # 对记忆进行聚类和总结，简化为伪代码
        pass
