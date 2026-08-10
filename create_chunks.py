import whisper
import json
import os

model = whisper.load_model("base")

os.makedirs("jsons", exist_ok=True)
audios = os.listdir("audios")

for audio in audios:
    if not audio.endswith(".mp3"):
        continue  # skip the .meta.json sidecar files sitting in the same folder

    file_id = audio.replace(".mp3", "")

    try:
        with open(f"audios/{file_id}.meta.json") as f:
            meta = json.load(f)
        title = meta["title"]

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # if Whisper found no speech at all, don't create an empty, useless chunk file
        if not result["text"].strip():
            print(f"⚠️ '{title}' contains no detectable speech — skipping.")
            continue

        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "file_id": file_id,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        with open(f"jsons/{file_id}.json", "w") as f:
            json.dump(chunks_with_metadata, f)

        print(f"✅ Transcribed '{title}' -> {len(chunks)} chunk(s)")

    except FileNotFoundError:
        print(f"❌ No metadata file found for '{audio}' — skipping.")
        continue
    except Exception as e:
        print(f"❌ Failed to transcribe '{audio}': {e}")
        continue