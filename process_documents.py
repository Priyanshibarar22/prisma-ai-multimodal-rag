import os
import json
from metadata_utils import get_file_metadata
from extract_documents import extract_any

os.makedirs("jsons", exist_ok=True)
files = os.listdir("document_uploads")
print(f"Found {len(files)} file(s) in document_uploads/")

for file in files:
    ext = file.split(".")[-1].lower()
    if ext not in ["pdf", "docx", "txt"]:
        print(f"Skipping '{file}' — not a supported document type.")
        continue

    meta = get_file_metadata(file)
    full_text, error = extract_any(f"document_uploads/{file}")

    if error:
        print(f"❌ Skipping '{file}': {error}")
        continue

    # simple chunking: split every ~800 characters and overlap by 150 characters
    chunk_size = 800
    overlap = 150
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(full_text), step):
        piece = full_text[i:i+chunk_size]
        chunks.append({
            "file_id": meta["file_id"],
            "title": meta["title"],
            "start": None,   # no timestamps for documents
            "end": None,
            "text": piece
        })

    chunks_with_metadata = {"chunks": chunks, "text": full_text}

    try:
        with open(f"jsons/{meta['file_id']}.json", "w") as f:
            json.dump(chunks_with_metadata, f)
        print(f"✅ Processed '{file}' -> {len(chunks)} chunk(s)")
    except Exception as e:
        print(f"❌ Failed to save chunks for '{file}': {e}")
        continue