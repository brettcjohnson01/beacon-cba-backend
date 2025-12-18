from typing import TypedDict, Optional


class Chunk(TypedDict):
    doc_id: str
    source_id: str
    chunk_id: str
    text: str

    # Optional but strongly recommended
    page_start: Optional[int]
    page_end: Optional[int]

    # Metadata for filtering / display
    title: Optional[str]
    state: Optional[str]
    city: Optional[str]
    project_type: Optional[str]
    year: Optional[int]
