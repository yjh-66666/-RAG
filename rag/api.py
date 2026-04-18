from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import RAGConfig
from .models import IngestTask, QueryRequest, SearchRequest
from .rag_system import EnterpriseRAG
from .utils import fail, new_request_id, ok


def _parse_api_keys(api_keys: str) -> set[str]:
    return {x.strip() for x in (api_keys or "").split(",") if x.strip()}


def create_app() -> FastAPI:
    cfg = RAGConfig()
    rag = EnterpriseRAG(cfg)

    app = FastAPI(title="RAG Project", version="1.1.0")

    static_dir = Path(__file__).parent / "web"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    executor = ThreadPoolExecutor(max_workers=2)
    task_lock = Lock()
    ingest_tasks: Dict[str, IngestTask] = {}

    api_key_set = _parse_api_keys(cfg.api_keys)

    def auth_dep(x_api_key: Optional[str] = Header(None)) -> None:
        if cfg.auth_mode == "none":
            return
        if cfg.auth_mode == "api_key":
            if not x_api_key:
                raise HTTPException(status_code=401, detail="Missing X-API-Key")
            if x_api_key not in api_key_set:
                raise HTTPException(status_code=403, detail="Invalid X-API-Key")
            return
        raise HTTPException(status_code=500, detail="Unsupported AUTH_MODE")

    @app.get("/")
    def index() -> Any:
        html = static_dir / "index.html"
        if html.exists():
            return FileResponse(str(html))
        return ok({"service": "rag", "message": "frontend not found"}, request_id=new_request_id())

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return ok(
            {
                "status": "running",
                "time": datetime.now().isoformat(),
                "vectorstore": cfg.vectorstore_type,
                "auth_mode": cfg.auth_mode,
            },
            request_id=new_request_id(),
        )

    @app.post("/ingest")
    async def ingest(
        files: List[UploadFile] = File(...),
        department: str = Form(...),
        user: str = Form("anonymous"),
        custom_metadata: str = Form("{}"),
        _: None = Depends(auth_dep),
    ) -> Dict[str, Any]:
        rid = new_request_id()

        if len(files) > cfg.max_upload_files:
            return fail("INGEST_TOO_MANY_FILES", f"单次最多上传 {cfg.max_upload_files} 个文件", request_id=rid)

        try:
            meta_obj = json.loads(custom_metadata or "{}")
            if not isinstance(meta_obj, dict):
                return fail("INGEST_INVALID_METADATA", "custom_metadata 必须是 JSON 对象", request_id=rid)
        except Exception:
            return fail("INGEST_INVALID_METADATA", "custom_metadata 不是合法 JSON", request_id=rid)

        upload_dir = Path(cfg.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in {".pdf", ".docx", ".txt"}:
                return fail("INGEST_UNSUPPORTED_FILE", f"不支持文件: {f.filename}", request_id=rid)

            content = await f.read()
            if len(content) > cfg.max_upload_file_mb * 1024 * 1024:
                return fail(
                    "INGEST_FILE_TOO_LARGE",
                    f"文件过大: {f.filename}，超过 {cfg.max_upload_file_mb}MB",
                    request_id=rid,
                )

            target = upload_dir / f"{rid}_{Path(f.filename).name}"
            target.write_bytes(content)
            saved_paths.append(target)

        task = IngestTask(status="pending", progress=0, detail="任务已创建")
        with task_lock:
            ingest_tasks[task.task_id] = task

        def worker() -> None:
            with task_lock:
                t = ingest_tasks[task.task_id]
                t.status = "running"
                t.progress = 10
                t.detail = "开始解析文档"
                t.updated_at = datetime.now()
            try:
                out = rag.ingest_files(saved_paths, department=department, user=user, custom_metadata=meta_obj)
                with task_lock:
                    t = ingest_tasks[task.task_id]
                    t.status = "completed"
                    t.progress = 100
                    t.detail = "入库完成"
                    t.result = out
                    t.updated_at = datetime.now()
            except Exception as e:
                with task_lock:
                    t = ingest_tasks[task.task_id]
                    t.status = "failed"
                    t.progress = 100
                    t.detail = f"入库失败: {e}"
                    t.result = {"error": str(e)}
                    t.updated_at = datetime.now()

        executor.submit(worker)
        return ok({"task_id": task.task_id, "status": task.status}, request_id=rid)

    @app.get("/ingest/status/{task_id}")
    def ingest_status(task_id: str, _: None = Depends(auth_dep)) -> Dict[str, Any]:
        rid = new_request_id()
        with task_lock:
            task = ingest_tasks.get(task_id)
        if not task:
            return fail("INGEST_TASK_NOT_FOUND", f"任务不存在: {task_id}", request_id=rid)
        return ok({"task": task.model_dump(mode="json")}, request_id=rid)

    @app.post("/search")
    def search(req: SearchRequest, _: None = Depends(auth_dep)) -> Dict[str, Any]:
        rid = new_request_id()
        try:
            docs, scores = rag.search(req.query, department=req.department, k=req.k, expr=req.expr)
            return ok(
                {
                    "query": req.query,
                    "results": [
                        {
                            "ref": i + 1,
                            "content": d.page_content,
                            "metadata": d.metadata,
                            "score": float(s),
                        }
                        for i, (d, s) in enumerate(zip(docs, scores))
                    ],
                },
                request_id=rid,
            )
        except Exception as e:
            return fail("SEARCH_ERROR", str(e), request_id=rid)

    @app.post("/query")
    def query(req: QueryRequest, _: None = Depends(auth_dep)) -> Dict[str, Any]:
        rid = new_request_id()
        try:
            history = rag.load_history(req.user, req.department, req.session_id)
            rewritten = rag.rewrite_query(req.question, chat_history=history)
            docs, scores = rag.search(rewritten, department=req.department, k=cfg.top_k, expr=req.expr)
            answer = rag.answer(req.question, docs, chat_history=history)

            history.append({"role": "user", "content": req.question})
            history.append({"role": "assistant", "content": answer})
            rag.save_history(req.user, req.department, req.session_id, history)

            return ok(
                {
                    "intent": rag.detect_intent(req.question),
                    "rewritten_query": rewritten,
                    "question": req.question,
                    "answer": answer,
                    "session_id": req.session_id,
                    "sources": [
                        {
                            "ref": i + 1,
                            "score": float(s),
                            "document_id": d.metadata.get("document_id"),
                            "source_file": d.metadata.get("source_file"),
                            "page_index": d.metadata.get("page_index"),
                            "snippet": d.page_content[:220],
                        }
                        for i, (d, s) in enumerate(zip(docs, scores))
                    ],
                },
                request_id=rid,
            )
        except Exception as e:
            return fail("QUERY_ERROR", str(e), request_id=rid)

    return app
