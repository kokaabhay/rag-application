from pydantic import BaseModel


class IngestResponse(BaseModel):
    message: str
    document_path: str
    document_count: int
    chunk_count: int


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str