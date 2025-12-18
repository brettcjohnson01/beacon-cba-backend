import os
import csv
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

# ---- Config ----
DOCS_CSV_PATH = os.getenv("DOCS_CSV_PATH", "data/metadata/documents.csv")
SOURCES_CSV_PATH = os.getenv("SOURCES_CSV_PATH", "data/metadata/sources.csv")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()


# ---- Models ----
class AskRequest(BaseModel):
    question: str


class DocumentMetadata(BaseModel):
    doc_id: str
    title: Optional[str] = None
    agreement_type: Optional[str] = None
    project_type: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    country: Optional[str] = None
    year_signed: Optional[str] = None
    parties: Optional[str] = None
    counterparty_type: Optional[str] = None
    enforceability: Optional[str] = None
    source_id: Optional[str] = None
    raw_filename: Optional[str] = None
    processed_text_filename: Optional[str] = None
    public_ok: Optional[str] = None
    tags: Optional[str] = None


class SourceMetadata(BaseModel):
    source_id: str
    url: Optional[str] = None
    publisher: Optional[str] = None
    access_date: Optional[str] = None
    notes: Optional[str] = None


# ---- Helpers ----
def _read_csv_as_dicts(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [row for row in reader]


def load_documents() -> List[DocumentMetadata]:
    rows = _read_csv_as_dicts(DOCS_CSV_PATH)
    docs: List[DocumentMetadata] = []
    for r in rows:
        # Ensure doc_id exists; skip malformed rows
        if not r.get("doc_id"):
            continue
        docs.append(DocumentMetadata(**r))
    docs.sort(key=lambda d: d.doc_id)
    return docs


def load_sources() -> List[SourceMetadata]:
    rows = _read_csv_as_dicts(SOURCES_CSV_PATH)
    sources: List[SourceMetadata] = []
    for r in rows:
        if not r.get("source_id"):
            continue
        sources.append(SourceMetadata(**r))
    sources.sort(key=lambda s: s.source_id)
    return sources


def _matches(value: Optional[str], wanted: Optional[str]) -> bool:
    if wanted is None:
        return True
    if value is None:
        return False
    return value.strip().lower() == wanted.strip().lower()


def _truthy_str(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    s = v.strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


# ---- Routes ----
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=f"question too long (max {MAX_QUESTION_CHARS} chars)")

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You help communities understand and draft Community Benefits Agreements (CBAs)."},
                {"role": "user", "content": q},
            ],
        )
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentMetadata])
def list_documents(
    agreement_type: Optional[str] = None,
    project_type: Optional[str] = None,
    location_state: Optional[str] = None,
    year_signed: Optional[str] = None,
    public_ok: Optional[bool] = Query(default=True, description="Defaults to true; set false to include non-public docs"),
):
    docs = load_documents()
    out: List[DocumentMetadata] = []

    for d in docs:
        if not _matches(d.agreement_type, agreement_type):
            continue
        if not _matches(d.project_type, project_type):
            continue
        if not _matches(d.location_state, location_state):
            continue
        if not _matches(d.year_signed, year_signed):
            continue

        # public_ok logic: default to only public docs
        doc_public = _truthy_str(d.public_ok)
        if public_ok is True and doc_public is not True:
            continue

        out.append(d)

    return out


@app.get("/documents/{doc_id}", response_model=DocumentMetadata)
def get_document(doc_id: str):
    docs = load_documents()
    for d in docs:
        if d.doc_id == doc_id:
            return d
    raise HTTPException(status_code=404, detail="document not found")


@app.get("/sources", response_model=List[SourceMetadata])
def list_sources():
    return load_sources()


@app.get("/sources/{source_id}", response_model=SourceMetadata)
def get_source(source_id: str):
    sources = load_sources()
    for s in sources:
        if s.source_id == source_id:
            return s
    raise HTTPException(status_code=404, detail="source not found")
