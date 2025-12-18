from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("data/raw/cba_pdfs")
OUT_DIR = Path("data/processed_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Process all PDFs in the folder
for pdf_path in RAW_DIR.glob("*.pdf"):
    out_path = OUT_DIR / (pdf_path.stem + ".txt")

    reader = PdfReader(str(pdf_path))
    texts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        texts.append(f"\n\n--- page {i+1} ---\n{page_text}")

    out_path.write_text("\n".join(texts), encoding="utf-8")
    print("Wrote:", out_path)
