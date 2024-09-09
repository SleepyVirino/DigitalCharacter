from langchain_openai import OpenAIEmbeddings


class EmbeddingModel():
    def __init__(self, model):
        self.model = model

    def encode(self, sentence):
        if isinstance(self.model, OpenAIEmbeddings):
            return self.model.embed_query(sentence)
        else:
            raise Exception('Model must be an OpenAIEmbeddings model')
