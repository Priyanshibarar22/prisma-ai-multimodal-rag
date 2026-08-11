import os
import re
import json
import time
import shutil
import pickle
import subprocess
from functools import lru_cache

import numpy as np
import faiss
import whisper
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

from metadata_utils import get_file_metadata
from extract_documents import extract_any

load_dotenv()

JSONS_DIR = "jsons"
UPLOADS_DIR = "temp_uploads"
INDEX_PATH = "vector_store.index"
META_PATH = "vector_store.meta"
HISTORY_DIR = "chat_sessions"



EMBED_VERSION = 2

os.makedirs(HISTORY_DIR, exist_ok=True)


class StaleIndexError(Exception):
    """Raised when the saved index doesn't match the current embedding version."""
    pass


# ---------------------------------------------------------------------------
# Model loading (lazy singletons)
# ---------------------------------------------------------------------------
# lru_cache(maxsize=1) on a zero-argument function works like a "load once,
# reuse forever" cache -- same trick as st.cache_resource, but without
# needing Streamlit imported in this file.

@lru_cache(maxsize=1)
def get_whisper_model():
    return whisper.load_model("base")


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Whisper hallucination cleanup
# ---------------------------------------------------------------------------
# Whisper sometimes invents phrases during silent/noisy stretches (things
# like "Thanks for watching!" or "Subscribe to the channel", or it repeats
# the same short phrase many times in a row). This won't catch everything,
# but it strips the most common junk patterns. Extend the pattern list as
# you spot new hallucinations in your transcripts.

_HALLUCINATION_PATTERNS = [
    r"(?i)thanks? for watching.*",
    r"(?i)please (subscribe|like and subscribe).*",
    r"(?i)see you (in the )?next (video|time).*",
    r"(?i)\[music\]",
    r"(?i)\[applause\]",
]


def clean_text(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for pattern in _HALLUCINATION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    # Collapse a short phrase repeated 3+ times in a row (e.g. "okay okay okay okay")
    cleaned = re.sub(r"\b(\w+(?:\s+\w+){0,4})\b(?:\s+\1\b){2,}", r"\1", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    path = f"{UPLOADS_DIR}/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def process_file(path, filename):
    """
    Extracts + chunks a single uploaded file and writes jsons/<file_id>.json.
    Returns (title, error). error is None on success.
    """
    ext = filename.split(".")[-1].lower()
    meta = get_file_metadata(filename)
    whisper_model = get_whisper_model()

    try:
        if ext in ["mp4", "webm"]:
            audio_path = f"{UPLOADS_DIR}/{meta['file_id']}.mp3"
            result = subprocess.run(["ffmpeg", "-i", path, audio_path, "-y"], capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(audio_path):
                return None, "Video file could not be converted — it may be corrupted."

            transcription = whisper_model.transcribe(
                audio=audio_path,
                task="translate",
                word_timestamps=False,
                temperature=0,
                condition_on_previous_text=False,
            )
            if not transcription["text"].strip():
                return None, "No speech detected in this video."

            chunks = [
                {
                    "file_id": meta["file_id"],
                    "title": meta["title"],
                    "start": s["start"],
                    "end": s["end"],
                    "text": clean_text(s["text"]),
                }
                for s in transcription["segments"]
            ]
            full_text = clean_text(transcription["text"])

        elif ext in ["mp3", "wav"]:
            transcription = whisper_model.transcribe(
                audio=path,
                language="hi",
                task="translate",
                word_timestamps=False,
                temperature=0,
                condition_on_previous_text=False,
            )
            if not transcription["text"].strip():
                return None, "No speech detected in this audio file."

            chunks = [
                {
                    "file_id": meta["file_id"],
                    "title": meta["title"],
                    "start": s["start"],
                    "end": s["end"],
                    "text": clean_text(s["text"]),
                }
                for s in transcription["segments"]
            ]
            full_text = clean_text(transcription["text"])

        elif ext in ["pdf", "docx", "pptx", "txt"]:
            full_text, error = extract_any(path)
            if error:
                return None, error

            # Bigger chunks + overlap: with no overlap, a heading might
            # land in one chunk and its content in the next,
            # so a real answer gets missed even though it's retrieved.
            # Overlapping windows make it far more likely that a
            # heading and its content land in the same chunk together.
            chunk_size = 800
            overlap = 150
            step = chunk_size - overlap
            chunks = []
            chunk_num = 0
            for i in range(0, len(full_text), step):
                piece = full_text[i:i + chunk_size]
                if not piece.strip():
                    continue
                chunk_num += 1
                chunks.append({
                    "file_id": meta["file_id"],
                    "title": f"{meta['title']} (chunk {chunk_num})",
                    "start": None,
                    "end": None,
                    "text": piece,
                })
                if i + chunk_size >= len(full_text):
                    break

        else:
            return None, f"Unsupported file type: .{ext}"

        os.makedirs(JSONS_DIR, exist_ok=True)
        with open(f"{JSONS_DIR}/{meta['file_id']}.json", "w") as f:
            json.dump({"chunks": chunks, "text": full_text}, f)

        return meta["title"], None

    except Exception as e:
        return None, f"Unexpected error: {e}"


def rebuild_vector_store():
    """Reads every json in jsons/, re-embeds everything, saves the FAISS index."""
    if not os.path.exists(JSONS_DIR):
        return False

    embedder = get_embedder()
    all_chunks = []
    for jf in os.listdir(JSONS_DIR):
        if not jf.endswith(".json"):
            continue
        try:
            with open(f"{JSONS_DIR}/{jf}") as f:
                data = json.load(f)
            all_chunks.extend(data["chunks"])
        except Exception:
            continue  # skip a corrupted json silently; already reported when it was created

    if not all_chunks:
        return False

    texts = [c["text"] for c in all_chunks]
    vectors = embedder.encode(texts, show_progress_bar=False, batch_size=16, normalize_embeddings=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"version": EMBED_VERSION, "chunks": all_chunks}, f)
    return True


def has_index():
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def is_index_stale():
    """
    True if an index exists but was built with an older embedding/chunking
    scheme than the code currently uses. Old pickles saved as a bare list
    (pre-versioning) count as stale too.
    """
    if not has_index():
        return False
    try:
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return True
    if isinstance(data, list):
        return True  # pre-versioning format
    return data.get("version") != EMBED_VERSION


def clear_index():
    """Disabled — no longer deletes jsons/ or the vector store, even if the index is stale.
    Files are only ever added, never automatically removed."""
    pass


# ---------------------------------------------------------------------------
# Retrieval + answer generation
# ---------------------------------------------------------------------------

def search(query, k=15, max_distance=None):
    """
    Returns the top-k chunks for `query`.

    max_distance is OFF by default. 
    """
    if not has_index():
        return []
    if is_index_stale():
        raise StaleIndexError(
            "The saved index was built with an older version of the app and is no "
            "longer compatible. Click 'Process Files' again to rebuild it."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta_data = pickle.load(f)
    all_chunks = meta_data["chunks"]

    embedder = get_embedder()
    qvec = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(qvec, dtype="float32"), k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(all_chunks):
            continue
        if max_distance is not None and dist > max_distance:
            continue
        results.append(all_chunks[idx])
    return results


def get_knowledge_base_stats():
    """Returns (file_count, chunk_count, file_list)."""
    if not os.path.exists(JSONS_DIR):
        return 0, 0, []

    file_list = []
    total_chunks = 0

    for jf in os.listdir(JSONS_DIR):
        if not jf.endswith(".json"):
            continue
        try:
            with open(f"{JSONS_DIR}/{jf}") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if chunks:
                title = chunks[0].get("original_filename") or chunks[0].get("title", jf).split(" (chunk")[0]
                file_id = chunks[0].get("file_id", jf.replace(".json", ""))
                file_list.append({"file_id": file_id, "title": title, "chunk_count": len(chunks)})
                total_chunks += len(chunks)
        except Exception:
            continue

    return len(file_list), total_chunks, file_list


def delete_file(file_id):
    """Deletes one specific file's json from the knowledge base."""
    path = f"{JSONS_DIR}/{file_id}.json"
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def clear_knowledge_base():
    """Deliberately, explicitly clears everything — only call this when the user confirms."""
    if os.path.exists(JSONS_DIR):
        shutil.rmtree(JSONS_DIR)
    os.makedirs(JSONS_DIR, exist_ok=True)
    for path in (INDEX_PATH, META_PATH):
        if os.path.exists(path):
            os.remove(path)
            

def delete_session(session_id):
    file_path = f"chat_sessions/{session_id}.json" 
    if os.path.exists(file_path):
        os.remove(file_path)


def answer_query(query, k=15, max_distance=None):
    """Returns (answer_text, retrieved_chunks)."""
    groq_client = get_groq_client()
    if not groq_client:
        return "GROQ_API_KEY is not set. Please configure it to ask questions.", []

    try:
        retrieved = search(query, k=k, max_distance=max_distance)
    except StaleIndexError as e:
        return str(e), []

    if not retrieved:
        return "No documents uploaded yet, or nothing relevant was found. Please upload files first.", []

    context = "\n\n".join(f"[{r['title']}]: {r['text']}" for r in retrieved)
    prompt = f"""You are a RAG assistant.

Use ONLY the information present in the uploaded documents provided in the context.

If the answer cannot be found in the uploaded documents, reply:
"I couldn't find that information in the uploaded files."

Do not make up facts.
Do not use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, retrieved
    except Exception as e:
        return f"Could not get a response from the LLM: {e}", retrieved


# ---------------------------------------------------------------------------
# Chat session persistence
# ---------------------------------------------------------------------------

def load_all_sessions():
    sessions = []
    for f in os.listdir(HISTORY_DIR):
        if f.endswith(".json"):
            try:
                with open(f"{HISTORY_DIR}/{f}") as file:
                    sessions.append(json.load(file))
            except Exception:
                continue
    sessions.sort(key=lambda s: s.get("last_updated", 0), reverse=True)
    return sessions


def save_session(session_id, messages):
    if not messages:
        return
    title = messages[0][0][:50]  # first question, trimmed, becomes the title
    with open(f"{HISTORY_DIR}/{session_id}.json", "w") as f:
        json.dump(
            {"session_id": session_id, "title": title, "messages": messages, "last_updated": time.time()},
            f,
        )