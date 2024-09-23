from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from server.agent.brain.brain import BrainModule
from server.agent.brain.embedding import EmbeddingModel

from server.agent.config import (
    OPENAI_KEY,
    OPENAI_BASE_URL
)

context = {}
figure = "Albert Einstein"

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
    embedding_model = EmbeddingModel(embedding_model)

    work_memory_path = "agent/brain/memory/work_memory.json"
    episodic_memory_path = "agent/brain/memory/episodic_memory.json"
    summary_path = "agent/brain/memory/summary.txt"

    brain = BrainModule(figure, llm, embedding_model, summary_path, work_memory_path, episodic_memory_path)
    context["brain"] = brain
    yield
    print("关闭后前执行")


app = FastAPI(lifespan=lifespan)


class UserInput(BaseModel):
    text: str
    place: str
    src: str
    dest: str




@app.post("/chat/")
async def get_response(user_input: UserInput):
    brain = context["brain"]

    user_input = user_input.model_dump()
    user_input["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_input["name"] = figure
    response = brain.response(user_input)
    return {"response": response.content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
