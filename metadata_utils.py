import uuid
import os

def get_file_metadata(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    file_id = str(uuid.uuid4())[:8]  # unique id, avoids collisions if 2 files have same name
    return {
        "file_id": file_id,
        "title": name_without_ext,   # just use the raw filename as the title
        "original_filename": filename
    }