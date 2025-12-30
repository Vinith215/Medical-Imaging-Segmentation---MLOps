import os
import uuid
import shutil
from pathlib import Path
from config.global_config import GlobalConfig

def handle_upload(uploaded_file):
    """
    Saves a streamlit UploadedFile to temp_uploads for disk-based processing.
    """
    # 1. Create a unique session ID
    session_id = str(uuid.uuid4())
    temp_path = GlobalConfig.ROOT_DIR / "data" / "temp_uploads" / session_id
    temp_path.mkdir(parents=True, exist_ok=True)

    # 2. Write the bytes to disk
    file_path = temp_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, temp_path

def cleanup_temp(path):
    """Removes the temporary directory after inference is done."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"🗑️ Cleaned up temp folder: {path}")