from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def now_iso() -> str:
    return datetime.now().isoformat()


def new_request_id() -> str:
    return str(uuid.uuid4())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def current_user() -> str:
    try:
        return os.getlogin()
    except Exception:
        return os.environ.get("USER") or os.environ.get("USERNAME") or "system"


def ok(data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "error": None, "request_id": request_id}


def fail(code: str, message: str, request_id: str) -> Dict[str, Any]:
    return {"success": False, "data": None, "error": {"code": code, "message": message}, "request_id": request_id}
