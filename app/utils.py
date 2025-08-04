from pathlib import Path
import uuid, glob

UPLOAD_DIR = Path("uploads")

def latest_session() -> str:
    """
    Returns the session-id of the most recently uploaded file.
    Falls back to a fresh UUID if none exist.
    """
    files = glob.glob(str(UPLOAD_DIR / "*.csv"))
    if not files:
        return str(uuid.uuid4())        # create fresh
    # sort by modification time, newest first
    newest = max(files, key=lambda f: Path(f).stat().st_mtime)
    return Path(newest).stem          