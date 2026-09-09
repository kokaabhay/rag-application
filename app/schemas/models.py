from pydantic import BaseModel,Field


class IngestResponse(BaseModel):
    message: str
    document_path: str
    document_count: int
    chunk_count: int