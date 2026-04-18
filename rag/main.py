from __future__ import annotations

import uvicorn

from .api import create_app
from .config import RAGConfig


def run() -> None:
    cfg = RAGConfig()
    app = create_app()
    uvicorn.run(app, host=cfg.app_host, port=cfg.app_port, log_level="info")


if __name__ == "__main__":
    run()
