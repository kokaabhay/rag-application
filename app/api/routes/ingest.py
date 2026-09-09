from pathlib import Path
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.service import ingest_document
from app.schemas.models import IngestResponse


router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

#allowed file types pdf,docx,txt,md
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".md",
}


@router.post(
    "",
    response_model=IngestResponse,
)
async def ingest(
    file: UploadFile = File(...)
):
    """
    Upload and ingest the knowledge-base document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Supported types: "
                f"{sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    try:
        # Create a temporary file for the upload
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        chunks = ingest_document(
            str(temp_path),
            extension,
        )

        # Remove temporary file
        temp_path.unlink(missing_ok=True)

        document_path = (
            Path("data/documents")
            / f"knowledge_base{extension}"
        )

        return IngestResponse(
            message="Document ingested successfully.",
            document_path=str(document_path),
            document_count=1,
            chunk_count=len(chunks),
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        )