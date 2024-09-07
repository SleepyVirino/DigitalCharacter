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


class MemoryManager:
    def __init__(self):
        self.summary = None
        self.work_memory = None
        self.episodic_memory = None
        self.semantic_memory = None
        pass

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
        pass

    def self_monitor(self):
        """
        自我监控理论更新对世界的总结
        :return:
        """
        pass

