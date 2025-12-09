from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.rag_engine import MyanmarAgriAssistEngine

app = FastAPI(
    title="Myanmar Agri-Assist API",
    description="AI-Powered Agricultural Knowledge Assistant for Burmese language data.",
    version="1.0.0",
)

engine = MyanmarAgriAssistEngine()


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Burmese instruction or message to classify.")


class ClassifyResponse(BaseModel):
    category: str
    confidence: float


class AskRequest(BaseModel):
    question: str = Field(..., description="Burmese agricultural question.")
    top_k: int = Field(3, ge=1, le=10, description="Number of knowledge snippets to retrieve.")


class ContextItem(BaseModel):
    category: str
    instruction: str
    output: str
    score: float


class AskResponse(BaseModel):
    answer: str
    contexts: List[ContextItem]


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return engine.classify_text(request.text)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    response = engine.ask_question(request.question, top_k=request.top_k)
    return AskResponse(**response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
