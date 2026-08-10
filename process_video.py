#converts the videos to mp3, with error handling
import os
import subprocess
import json
from metadata_utils import get_file_metadata

os.makedirs("audios", exist_ok=True)
files = os.listdir("video_uploads")
print(f"Found {len(files)} file(s) in video_uploads/")

for file in files:
    ext = file.split(".")[-1].lower()
    if ext not in ["mp4", "webm"]:
        print(f"Skipping '{file}' — not a supported video type.")
        continue

    meta = get_file_metadata(file)
    output_path = f"audios/{meta['file_id']}.mp3"

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", f"./video_uploads/{file}", output_path, "-y"],
            capture_output=True, text=True
        )

        # FFmpeg doesn't always throw a Python error on failure — check its exit code
        if result.returncode != 0:
            print(f"❌ Could not convert '{file}'. It may be corrupted or in an unsupported format.")
            continue

        # confirm the output file was actually created and isn't empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"❌ Conversion of '{file}' produced no usable audio.")
            continue

        with open(f"audios/{meta['file_id']}.meta.json", "w") as f:
            json.dump(meta, f)

        print(f"✅ Converted '{file}' -> {meta['file_id']}.mp3")

    except Exception as e:
        print(f"❌ Unexpected error processing '{file}': {e}")
        continue