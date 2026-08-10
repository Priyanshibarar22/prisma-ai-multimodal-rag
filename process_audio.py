import os
import json
import whisper
from metadata_utils import get_file_metadata

model = whisper.load_model("base")

os.makedirs("jsons", exist_ok=True)
audio_files = os.listdir("audio_uploads")

for file in audio_files:
    ext = file.split(".")[-1].lower()
    if ext not in ["mp3", "wav"]:
        print(f"Skipping '{file}' — not a supported audio type.")
        continue

    meta = get_file_metadata(file)

    try:
        result = model.transcribe(
            audio=f"audio_uploads/{file}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        if not result["text"].strip():
            print(f"⚠️ '{file}' contains no detectable speech — skipping.")
            continue

        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "file_id": meta["file_id"],
                "title": meta["title"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        with open(f"jsons/{meta['file_id']}.json", "w") as f:
            json.dump(chunks_with_metadata, f)

        print(f"✅ Processed '{file}' -> {meta['file_id']}.json")

    except Exception as e:
        print(f"❌ Failed to process '{file}': {e}")
        continue