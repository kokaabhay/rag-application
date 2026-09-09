# Customer Support RAG Application

A simple Customer Support Retrieval-Augmented Generation (RAG) application built with FastAPI.

The application allows a user to upload a single knowledge-base document and ask questions about its contents. The system retrieves relevant information using both **semantic vector search** and **keyword search**, combines the results using **Reciprocal Rank Fusion (RRF)**, reranks the retrieved documents using the **OpenRouter Reranking API**, and finally sends the relevant context to an LLM to generate the answer.

---

# 1. Project Overview

The application follows this RAG pipeline:

```text
                    INGESTION
                       │
                       ▼
                 Upload Document
                       │
                       ▼
                  Unstructured
                       │
                       ▼
                    Chunking
                       │
                       ▼
              HuggingFace Embeddings
                       │
              ┌────────┴────────┐
              ▼                 ▼
          ChromaDB             BM25
       Vector Search       Keyword Search
              │                 │
              └────────┬────────┘
                       ▼
                Hybrid Search
                     RRF
                       │
                       ▼
             OpenRouter Reranker
                       │
                       ▼
                  Top Documents
                       │
                       ▼
                  Build Prompt
                       │
                       ▼
                OpenRouter LLM
                       │
                       ▼
                  Final Answer
```

The goal is to combine the strengths of different retrieval methods:

* **Vector search** understands semantic meaning.
* **BM25** handles exact keywords and terminology.
* **RRF** combines both rankings.
* **Reranking** performs a more detailed relevance comparison.
* **LLM** generates the final natural-language response.

---

# 2. Main Features

The application supports:

* Uploading a local knowledge-base document through an API.
* Supported document formats:

  * `.pdf`
  * `.txt`
  * `.docx`
  * `.md`
* Automatically saving the uploaded document as:

```text
data/documents/knowledge_base.<extension>
```

* Replacing the previous knowledge-base document when a new one is uploaded.
* Document parsing using Unstructured.
* Recursive text chunking.
* Hugging Face sentence embeddings.
* Persistent ChromaDB vector storage.
* BM25 keyword retrieval.
* Hybrid retrieval using Reciprocal Rank Fusion.
* OpenRouter-based reranking.
* OpenRouter-based LLM response generation.
* FastAPI REST API.
* Health-check endpoint.
* Fallback response when relevant information cannot be found.

---

# 3. Technology Stack

| Component             | Technology                               |
| --------------------- | ---------------------------------------- |
| API                   | FastAPI                                  |
| Validation            | Pydantic                                 |
| Configuration         | Pydantic Settings                        |
| Document Processing   | Unstructured                             |
| Document Loader       | LangChain Unstructured                   |
| Chunking              | LangChain RecursiveCharacterTextSplitter |
| Embeddings            | Hugging Face / Sentence Transformers     |
| Vector Database       | ChromaDB                                 |
| Keyword Search        | BM25                                     |
| Hybrid Search         | Reciprocal Rank Fusion                   |
| Reranker              | OpenRouter Reranking API                 |
| LLM                   | OpenRouter                               |
| Environment Variables | `.env`                                   |
| Language              | Python                                   |

---

# 4. Project Structure

```text
rag-application/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── huggingface.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── ingest.py
│   │       └── chat.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── service.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompt.py
│   │   └── service.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── chatbot.py
│   │
│   └── schemas/
│       ├── __init__.py
│       └── models.py
│
├── data/
│   └── documents/
│
├── chroma_db/
│
├── tests/
│
├── .env
├── .env.example
├── .gitignore
└── requirements.txt
```

---

# 5. Directory Responsibilities

## `app/main.py`

Creates the FastAPI application and registers the API routers.

It exposes:

```text
GET /health
POST /ingest
POST /chat
```

---

## `app/config.py`

Loads configuration from environment variables.

Example:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Customer Support RAG"

    llm_api_key: str
    llm_base_url: str
    llm_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
```

This keeps secrets and configuration outside the application code.

---

# 6. Hugging Face SSL Configuration

## `app/core/huggingface.py`

This file exists because the development environment has an SSL certificate verification problem when connecting to Hugging Face.

The configuration is:

```python
import requests
from huggingface_hub import configure_http_backend


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory)
```

The purpose is to configure Hugging Face's HTTP session to bypass certificate verification.

The embedding implementation itself is not changed.

The application imports this configuration before initializing the embedding model:

```python
import app.core.huggingface
```

The embedding model remains:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is a development-environment workaround. Certificate verification should preferably be fixed properly in a production environment.

---

# 7. Document Ingestion

The ingestion process begins when the client sends a document to:

```text
POST /ingest
```

The uploaded file is received by FastAPI as an `UploadFile`.

Supported extensions are:

```text
.pdf
.txt
.docx
.md
```

The application saves the uploaded document using the standard name:

```text
data/documents/knowledge_base.<extension>
```

If another knowledge-base document already exists, it is removed first.

Therefore, the application maintains only **one active knowledge-base document**.

---

# 8. Ingestion Pipeline

The ingestion pipeline is:

```text
Uploaded File
     ↓
Temporary File
     ↓
Save as knowledge_base
     ↓
Unstructured Loader
     ↓
Extracted Documents
     ↓
Text Chunking
     ↓
Vector Store
     ↓
BM25 Index
```

---

# 9. Document Loader

File:

```text
app/ingestion/loader.py
```

The loader uses:

```python
from langchain_unstructured import UnstructuredLoader
```

The loader converts the uploaded document into LangChain `Document` objects.

Each document contains information such as:

```python
Document(
    page_content="...",
    metadata={...}
)
```

`page_content` contains the actual extracted text.

Metadata may contain information such as page numbers or document information.

---

# 10. Chunking

File:

```text
app/ingestion/chunker.py
```

The application uses:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```text
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

These values are approximately **characters**, not tokens.

The overlap ensures that information near a chunk boundary is not completely separated from the surrounding context.

For example:

```text
Chunk 1
-------------------------
A B C D E F G H
              ↑
           overlap
                ↓
Chunk 2
-------------------------
        G H I J K L M
```

The chunking process produces smaller documents that are easier to retrieve.

---

# 11. Vector Embeddings

File:

```text
app/retrieval/vector.py
```

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts text into numerical vectors.

For example:

```text
"How can I reset my password?"
```

becomes something conceptually similar to:

```text
[0.12, -0.31, 0.44, ...]
```

The vector represents the semantic meaning of the text.

This allows the application to retrieve text that is semantically similar even when the exact words are different.

---

# 12. ChromaDB

The generated embeddings are stored in ChromaDB.

Configuration:

```text
Database directory:
chroma_db/

Collection:
customer_support
```

ChromaDB provides vector similarity search.

The current vector retrieval limit is:

```text
VECTOR_TOP_K = 10
```

Therefore, vector search attempts to retrieve the top 10 semantically relevant chunks.

---

# 13. Metadata Filtering

Unstructured can produce complex metadata structures.

ChromaDB expects metadata values to be compatible with its metadata requirements.

Therefore, before storing documents, the application uses:

```python
filter_complex_metadata(chunks)
```

This prevents errors caused by complex metadata values.

The actual document text remains in:

```python
document.page_content
```

---

# 14. BM25 Keyword Search

File:

```text
app/retrieval/bm25.py
```

BM25 is a traditional keyword-based information retrieval algorithm.

It is useful when the user uses exact terminology.

For example, if the document contains:

```text
"refund policy"
```

and the user searches:

```text
refund policy
```

BM25 can strongly match those exact terms.

The current configuration is:

```text
BM25_TOP_K = 10
```

The text is tokenized using a simple regular expression:

```python
re.findall(r"\b\w+\b", text.lower())
```

The BM25 index is maintained in memory.

This means:

* It is created when the document is ingested.
* It is available while the application is running.
* Restarting the application clears the in-memory BM25 index.
* The document should be ingested again after a restart.

---

# 15. Hybrid Search

File:

```text
app/retrieval/hybrid.py
```

Instead of relying on only one retrieval strategy, the application performs:

```text
Query
 ├──→ Vector Search
 │
 └──→ BM25 Search
```

Both methods return ranked documents.

The application then combines their rankings using:

## Reciprocal Rank Fusion (RRF)

The formula is:

```text
RRF Score = 1 / (k + rank)
```

where:

```text
k = 60
```

If a document appears in both result lists, its scores are added.

Conceptually:

```text
Vector Search:

Document A → Rank 1
Document B → Rank 2
Document C → Rank 3


BM25:

Document C → Rank 1
Document A → Rank 2
Document D → Rank 3
```

RRF combines them:

```text
Document A → vector score + BM25 score
Document C → vector score + BM25 score
Document B → vector score
Document D → BM25 score
```

This produces a combined ranking.

Current configuration:

```text
HYBRID_TOP_K = 10
RRF_K = 60
```

---

# 16. Why Hybrid Search?

Vector search and BM25 have different strengths.

## Vector Search

Good for:

```text
semantic similarity
different wording
conceptual questions
```

Example:

```text
Document:
"Customers may change their password from the account settings page."

Query:
"Where do I update my login credentials?"
```

The wording is different, but the meaning is similar.

Vector search can identify the relationship.

## BM25

Good for:

```text
exact words
product names
error codes
specific terminology
IDs
```

Example:

```text
Document:
"Error code AUTH-401 occurs when authentication fails."

Query:
"AUTH-401"
```

BM25 is very effective here.

## Hybrid Search

Combining both gives the system a more balanced retrieval strategy.

---

# 17. OpenRouter Reranking

File:

```text
app/retrieval/reranker.py
```

The application does **not** use a local CrossEncoder anymore.

Instead, the retrieved documents are sent to the OpenRouter reranking API.

The endpoint is:

```text
https://openrouter.ai/api/v1/rerank
```

The reranker receives:

```text
Query
+
Retrieved Documents
```

Example:

```json
{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
    "query": "How do I reset my password?",
    "documents": [
        {
            "text": "..."
        },
        {
            "text": "..."
        }
    ],
    "top_n": 3
}
```

The reranker returns relevance scores.

The application uses:

```text
RERANK_TOP_K = 3
```

Therefore, the best three documents are passed forward to the LLM.

---

# 18. Reranking Process

The retrieval process is:

```text
User Query
     ↓
Vector Search → 10
     ↓
BM25 Search → 10
     ↓
RRF
     ↓
Hybrid Results → 10
     ↓
OpenRouter Reranker
     ↓
Top 3
```

The reranker performs a more detailed comparison between:

```text
query
```

and:

```text
document content
```

This helps remove documents that were retrieved initially but are not actually useful for answering the question.

---

# 19. Chatbot Service

File:

```text
app/services/chatbot.py
```

This file contains the main RAG orchestration logic.

The function is:

```python
def chat(query: str) -> str:
```

The process is:

```text
1. Hybrid Search
       ↓
2. Reranking
       ↓
3. Build Context
       ↓
4. Build Prompt
       ↓
5. Generate LLM Answer
```

The chatbot service does not directly deal with:

* FastAPI request handling
* ChromaDB implementation
* BM25 implementation
* LLM HTTP requests

Those responsibilities are separated into their own modules.

This keeps the business logic easier to understand and maintain.

---

# 20. Prompt Construction

File:

```text
app/llm/prompt.py
```

The prompt builder receives:

```text
User Query
+
Retrieved Context
```

Conceptually:

```text
System Instructions
+
Knowledge Base Context
+
User Question
```

The LLM is instructed to answer using the provided knowledge-base information.

The objective is to reduce hallucination by restricting the answer to the retrieved context.

The final response does not include a separate "Sources" section.

---

# 21. LLM Service

File:

```text
app/llm/service.py
```

The LLM is accessed through OpenRouter.

The application uses configuration from:

```text
.env
```

The API key, base URL, and model name are not hard-coded in the application.

The service receives the constructed prompt and returns the generated answer.

---

# 22. Fallback Behavior

The chatbot contains a fallback message:

```text
I'm sorry, but I couldn't find relevant information in the knowledge base to answer your question.
```

The fallback is returned when retrieval or reranking does not produce usable results.

For example:

```text
User Query
    ↓
Hybrid Search
    ↓
No documents
    ↓
Fallback
```

or:

```text
User Query
    ↓
Hybrid Search
    ↓
Reranker
    ↓
No usable results
    ↓
Fallback
```

This is preferable to allowing the LLM to invent information that does not exist in the knowledge base.

---

# 23. API Endpoints

## Health Check

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
    "status": "healthy",
    "application": "Customer Support RAG"
}
```

---

# 24. Ingest Document

```http
POST /ingest
```

The endpoint accepts a multipart file upload.

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
     -F "file=@customer_support.pdf"
```

Supported formats:

```text
.pdf
.txt
.docx
.md
```

Example response:

```json
{
    "message": "Document ingested successfully.",
    "document_path": "data/documents/knowledge_base.pdf",
    "document_count": 1,
    "chunk_count": 25
}
```

The exact chunk count depends on the uploaded document.

---

# 25. Ask a Question

```http
POST /chat
```

Request body:

```json
{
    "query": "How can I reset my password?"
}
```

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d "{\"query\":\"How can I reset my password?\"}"
```

Example response:

```json
{
    "answer": "You can reset your password from the account settings page..."
}
```

The answer is generated from the retrieved knowledge-base context.

---

# 26. API Documentation

When FastAPI is running, interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Alternative OpenAPI documentation:

```text
http://127.0.0.1:8000/redoc
```

The OpenAPI specification is available at:

```text
http://127.0.0.1:8000/openapi.json
```

---

# 27. Environment Variables

Create a `.env` file in the project root.

Example:

```env
LLM_API_KEY="your-api-key"
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-3.5-lightning:free
```

Do not commit the actual `.env` file to Git.

---

# 28. `.env.example`

The repository can contain:

```env
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-3.5-lightning:free
```

This allows another developer to understand which environment variables are required without exposing the real API key.

---

# 29. Installation

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# 30. Requirements

The project requires packages similar to:

```text
fastapi[standard]
pydantic
pydantic-settings

langchain
langchain-community
langchain-text-splitters
langchain-unstructured
langchain-chroma
langchain-huggingface

chromadb

sentence-transformers
rank-bm25

unstructured[all-docs]

python-dotenv
requests
```

---

# 31. Running the Application

From the project root:

```text
C:\Users\...\rag-application
```

run:

```powershell
python -m fastapi dev app/main.py
```

The application will start locally.

Open:

```text
http://127.0.0.1:8000/docs
```

to use the API interactively.

---

# 32. Complete Usage Flow

## Step 1 — Start the application

```powershell
python -m fastapi dev app/main.py
```

---

## Step 2 — Upload a document

Call:

```text
POST /ingest
```

with a PDF, TXT, DOCX, or Markdown file.

---

## Step 3 — Document is saved

The document becomes:

```text
data/documents/knowledge_base.pdf
```

or the appropriate extension.

---

## Step 4 — Document is loaded

Unstructured extracts the document's text.

---

## Step 5 — Text is chunked

The document is divided into smaller chunks.

Current settings:

```text
Chunk size: 500 characters
Overlap: 100 characters
```

---

## Step 6 — Embeddings are generated

Each chunk is converted into a vector using:

```text
all-MiniLM-L6-v2
```

---

## Step 7 — ChromaDB is populated

The vectors are stored in:

```text
chroma_db/
```

---

## Step 8 — BM25 index is created

The same chunks are tokenized and stored in the in-memory BM25 index.

---

## Step 9 — User asks a question

For example:

```text
How can I cancel my subscription?
```

---

## Step 10 — Vector search

The query is converted into an embedding.

ChromaDB finds semantically similar chunks.

```text
Top 10
```

---

## Step 11 — BM25 search

BM25 searches for keyword-level matches.

```text
Top 10
```

---

## Step 12 — RRF

The two ranked lists are combined.

```text
Vector ranking
       +
BM25 ranking
       ↓
RRF ranking
```

---

## Step 13 — Reranking

The hybrid results are sent to the OpenRouter reranking API.

The reranker returns the most relevant documents.

```text
Top 3
```

---

## Step 14 — Context construction

The top documents are joined:

```text
Document 1
+
Document 2
+
Document 3
```

This becomes the context for the LLM.

---

## Step 15 — Prompt creation

The application creates a prompt containing:

```text
Instructions
+
Knowledge Base Context
+
User Question
```

---

## Step 16 — LLM generation

The prompt is sent to the configured OpenRouter LLM.

---

## Step 17 — Final response

The generated answer is returned through:

```text
POST /chat
```

---

# 33. Separation of Responsibilities

One of the important design goals of this project is separation of business logic.

For example:

```text
API Layer
    ↓
Service Layer
    ↓
Retrieval Layer
    ↓
LLM Layer
```

### API Layer

Handles HTTP requests and responses.

```text
app/api/routes/
```

### Ingestion Layer

Handles document loading, saving, and chunking.

```text
app/ingestion/
```

### Retrieval Layer

Handles:

```text
Vector Search
BM25
Hybrid Search
RRF
Reranking
```

```text
app/retrieval/
```

### LLM Layer

Handles:

```text
Prompt Construction
LLM API Calls
```

```text
app/llm/
```

### Service Layer

Coordinates the complete chatbot workflow.

```text
app/services/chatbot.py
```

This prevents the application from putting everything into one large FastAPI route.

---

# 34. Why We Use Both Retrieval and Reranking

Retrieval and reranking have different jobs.

## Retrieval

Retrieval answers:

> "Which documents might be relevant?"

It should be relatively broad.

The system retrieves approximately:

```text
10 documents
```

using hybrid search.

## Reranking

Reranking answers:

> "Which of these retrieved documents are actually the most relevant to this exact question?"

The reranker reduces the results to:

```text
Top 3
```

## LLM

The LLM then answers:

> "Using these relevant documents, how should I explain the answer to the user?"

Therefore:

```text
Retrieval
    ↓
Candidate documents

Reranking
    ↓
Best documents

LLM
    ↓
Natural language answer
```

---

# 35. Why Not Send All Retrieved Documents to the LLM?

Sending too many chunks can:

* increase token usage
* introduce irrelevant information
* make the prompt larger
* make the answer less focused
* increase the chance of confusing the LLM

Therefore:

```text
10 retrieved
     ↓
3 reranked
     ↓
LLM
```

provides a more focused context.

---

# 36. Current Configuration

The important retrieval settings are:

```text
Chunk size:
500 characters

Chunk overlap:
100 characters

Vector Top K:
10

BM25 Top K:
10

Hybrid Top K:
10

RRF K:
60

Reranker Top K:
3
```

These values can be adjusted later based on evaluation.

---

# 37. Important Limitations

This project intentionally keeps the architecture simple.

## One knowledge-base document

Only one document is maintained:

```text
knowledge_base.<extension>
```

Uploading another document replaces the previous one.

---

## BM25 is in memory

The BM25 index is not persisted.

After restarting the application:

```text
BM25 index → lost
```

The document should be ingested again.

---

## Local ChromaDB

ChromaDB is stored locally in:

```text
chroma_db/
```

It is not currently an Azure AI Search deployment.

---

## Local Hugging Face Embeddings

The application currently uses:

```text
all-MiniLM-L6-v2
```

locally.

---

## OpenRouter Dependencies

The reranker and LLM require access to the configured OpenRouter API.

---

# 38. Troubleshooting

## Hugging Face SSL Error

If you see:

```text
CERTIFICATE_VERIFY_FAILED
```

the development environment is having trouble validating Hugging Face's SSL certificate.

The application contains:

```text
app/core/huggingface.py
```

which configures the Hugging Face HTTP backend with:

```python
session.verify = False
```

Do not add another `set_client_factory()` implementation.

For the installed Hugging Face Hub version, the project uses:

```python
configure_http_backend()
```

---

## `set_client_factory` Error

If you see:

```text
AttributeError:
No huggingface_hub attribute set_client_factory
```

search the project for:

```text
set_client_factory
```

and remove the old implementation.

The project should use:

```python
from huggingface_hub import configure_http_backend
```

instead.

---

## BM25 Index Error

If you see:

```text
BM25 index has not been created.
Please ingest a document first.
```

call:

```text
POST /ingest
```

before calling:

```text
POST /chat
```

---

## ChromaDB Metadata Error

If ChromaDB complains about unsupported metadata types, ensure that the ingestion process uses:

```python
filter_complex_metadata(chunks)
```

before storing the chunks.

---

## Unsupported File Type

The application only accepts:

```text
.pdf
.txt
.docx
.md
```

Uploading another extension results in a `400` response.

---

# 39. Security Notes

Never commit:

```text
.env
```

to source control.

The API key should remain in the environment:

```env
LLM_API_KEY=...
```

The `.gitignore` should include:

```text
.env
venv/
__pycache__/
*.pyc
chroma_db/
```

The API key should never be written directly into Python source code.

---

# 40. Testing Strategy

The project can be tested at different levels.

## Health Test

```text
GET /health
```

Expected:

```json
{
    "status": "healthy"
}
```

## Ingestion Test

Upload a supported document and verify:

* request succeeds
* knowledge-base file exists
* chunks are created
* ChromaDB is populated
* BM25 index is created

## Retrieval Test

Ask a question that clearly exists in the document.

Verify that relevant chunks are retrieved.

## Reranking Test

Verify that the OpenRouter reranker returns ranked results.

## End-to-End Test

Perform:

```text
Upload document
       ↓
Ask question
       ↓
Hybrid search
       ↓
Reranking
       ↓
LLM
       ↓
Answer
```

---

# 41. Example End-to-End Scenario

Suppose the uploaded document contains:

```text
Customers can reset their password by opening Account Settings,
selecting Security, and choosing Reset Password.
```

The user asks:

```text
How do I change my password?
```

The system performs:

```text
Query
 ↓
Vector Search
 ↓
BM25
 ↓
RRF
 ↓
Relevant password-reset chunks
 ↓
OpenRouter Reranker
 ↓
Top 3 chunks
 ↓
Prompt
 ↓
LLM
```

The LLM can then produce an answer such as:

```text
You can reset your password by opening Account Settings,
going to Security, and selecting Reset Password.
```

The answer is based on the retrieved knowledge-base context.

---

# 42. Overall Architecture

The complete architecture can be summarized as:

```text
                         FASTAPI
                            │
                 ┌──────────┴──────────┐
                 │                     │
             /ingest                 /chat
                 │                     │
                 ▼                     ▼
           Ingestion Service      Chatbot Service
                 │                     │
                 ▼                     ▼
           Unstructured          Hybrid Search
                 │                ┌────┴────┐
                 ▼                │         │
             Chunking          ChromaDB   BM25
                 │                │         │
                 │                └────┬────┘
                 │                     │
                 │                    RRF
                 │                     │
                 │                     ▼
                 │              OpenRouter Reranker
                 │                     │
                 │                     ▼
                 │                 Context
                 │                     │
                 │                     ▼
                 │                 Prompt
                 │                     │
                 │                     ▼
                 │              OpenRouter LLM
                 │                     │
                 │                     ▼
                 │                  Answer
                 │
                 ├── HuggingFace Embeddings
                 │
                 ├── ChromaDB
                 │
                 └── BM25
```

---

# 43. Future Azure Mapping

The current project is intentionally built locally, but the architecture can later be mapped to Azure services.

| Current Component       | Possible Azure Replacement          |
| ----------------------- | ----------------------------------- |
| Local documents         | Azure Blob Storage                  |
| Unstructured            | Azure document processing/loaders   |
| Hugging Face embeddings | Azure OpenAI embeddings             |
| ChromaDB                | Azure AI Search                     |
| BM25                    | Azure AI Search keyword search      |
| Local hybrid RRF        | Azure AI Search hybrid retrieval    |
| OpenRouter reranker     | Azure-compatible reranking approach |
| OpenRouter LLM          | Azure OpenAI                        |
| FastAPI                 | Azure App Service / Container Apps  |

The important point is that the application separates these responsibilities, making future migration easier.

---

# 44. Design Philosophy

This project intentionally avoids hiding the complete RAG process behind one high-level abstraction.

Instead, the important steps are explicit:

```text
Load
 ↓
Chunk
 ↓
Embed
 ↓
Store
 ↓
Vector Search
 ↓
BM25 Search
 ↓
RRF
 ↓
Rerank
 ↓
Prompt
 ↓
LLM
```

This makes the application easier to:

* understand
* debug
* explain in an interview
* modify
* evaluate
* migrate to Azure

---

# 45. Quick Start

For a quick start:

### 1. Activate environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure `.env`

```env
LLM_API_KEY="your-api-key"
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-3.5-lightning:free
```

### 4. Start FastAPI

```powershell
python -m fastapi dev app/main.py
```

### 5. Open Swagger

```text
http://127.0.0.1:8000/docs
```

### 6. Upload a document

```text
POST /ingest
```

### 7. Ask a question

```text
POST /chat
```

Example:

```json
{
    "query": "What is the refund policy?"
}
```

---

# 46. Final RAG Pipeline

The complete system can be remembered as:

```text
DOCUMENT
   ↓
UNSTRUCTURED
   ↓
CHUNKING
   ↓
HUGGING FACE EMBEDDINGS
   ↓
┌──────────────────────┐
│                      │
▼                      ▼
CHROMADB              BM25
│                      │
└──────────┬───────────┘
           ▼
       HYBRID / RRF
           ↓
 OPENROUTER RERANKER
           ↓
       TOP 3 CHUNKS
           ↓
      PROMPT BUILDER
           ↓
      OPENROUTER LLM
           ↓
      FINAL ANSWER
```

The key idea is:

> **Retrieve broadly, rerank precisely, and generate using only the relevant knowledge-base context.**
