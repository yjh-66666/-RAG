from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import RAGConfig
from .document_processor import DocumentProcessor
from .embeddings import LocalEmbeddings
from .models import ChatMessage
from .retrieval import HybridRetriever, build_department_expr
from .vector_store import VectorStoreManager


class EnterpriseRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.doc_processor = DocumentProcessor(self.config)
        self.embeddings = LocalEmbeddings(self.config.embedding_model_path, device=self.config.embedding_device)
        self.vector_store = VectorStoreManager(self.config, self.embeddings)
        self.hybrid = HybridRetriever(alpha=self.config.hybrid_alpha)
        self.llm: Optional[ChatOpenAI] = None

        self._history_lock = Lock()

        Path(self.config.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.metadata_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.chat_history_dir).mkdir(parents=True, exist_ok=True)

    def _metadata_file(self) -> Path:
        return Path(self.config.metadata_db_path) / "metadata.json"

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        p = self._metadata_file()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        db = self._load_metadata()
        return db.get(document_id)

    def init_llm(self) -> None:
        if self.llm is not None:
            return
        if not self.config.openai_api_key:
            raise ValueError("未配置 OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=self.config.openai_chat_model,
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            temperature=0.2,
            timeout=40,
            max_tokens=1800,
        )

    @staticmethod
    def detect_intent(question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["document_id", "文档id", "哈希", "file_hash", "上传者", "元数据"]):
            return "doc_meta"
        if any(k in q for k in ["查找", "检索", "有哪些", "相关文档", "搜索"]):
            return "search"
        return "qa"

    def rewrite_query(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        if not self.config.rewrite_enabled:
            return question.strip()
        if not chat_history:
            return question.strip()
        if len(question.strip()) > 40:
            return question.strip()
        try:
            self.init_llm()
            history = chat_history[-20:]
            history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history if m.get("content")])
            prompt = (
                "你是查询改写助手。请将用户最后问题改写为可独立检索的一句话。只输出改写文本。\n\n"
                f"对话历史:\n{history_text}\n\n"
                f"用户问题:\n{question}\n\n"
                "改写结果:"
            )
            resp = self.llm.invoke(prompt)
            out = (getattr(resp, "content", None) or str(resp)).strip()
            return out or question.strip()
        except Exception:
            return question.strip()

    def ingest_files(self, file_paths: List[Path], department: str, user: str, custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        custom_metadata = custom_metadata or {}
        docs, sources = self.doc_processor.ingest_uploaded_files(file_paths, department=department, custom_metadata=custom_metadata)
        if not docs:
            return {"status": "no_documents", "documents": 0, "chunks": 0}

        chunks = self.doc_processor.split_documents(docs)
        for c in chunks:
            c.metadata["department"] = department
            c.metadata["ingest_user"] = user

        self.vector_store.build_or_load(chunks)
        self.doc_processor.persist_metadata(self.config.metadata_db_path, sources)
        return {"status": "ok", "documents": len(sources), "chunks": len(chunks)}

    def _calculate_freshness(self, doc: Document) -> float:
        t = doc.metadata.get("last_modified") or doc.metadata.get("upload_time")
        try:
            dt = datetime.fromisoformat(str(t))
        except Exception:
            return 0.5
        days = max(0, (datetime.now() - dt).days)
        if days <= 90:
            return 1.0
        if days >= 720:
            return 0.0
        return max(0.0, 1 - (days - 90) / 630)

    def _dense_candidates(self, query: str, department: str, fetch_k: int, expr: Optional[str]):
        final_expr = build_department_expr(department, custom_expr=expr)
        dense_raw = self.vector_store.similarity_search_with_score(query, k=fetch_k, expr=final_expr)

        docs = [d for d, _ in dense_raw]
        scores = [float(s) for _, s in dense_raw]

        if self.config.vectorstore_type == "chroma":
            keep_docs, keep_scores = [], []
            for d, s in zip(docs, scores):
                if str(d.metadata.get("department", "")) != department:
                    continue
                keep_docs.append(d)
                keep_scores.append(s)
            return keep_docs, keep_scores

        return docs, scores

    def search(self, query: str, department: str, k: int = 10, expr: Optional[str] = None):
        if self.vector_store.vectorstore is None:
            self.vector_store.build_or_load(documents=None)

        fetch_k = max(self.config.hybrid_fetch_k, k)
        dense_docs, dense_scores = self._dense_candidates(query, department, fetch_k, expr)

        if len(dense_docs) < k and self.config.vectorstore_type == "chroma":
            dense_docs, dense_scores = self._dense_candidates(query, department, max(fetch_k * 3, 50), expr)

        bm25_map = self.hybrid.bm25_score_map(query, dense_docs)
        docs, fused_scores = self.hybrid.fuse(dense_docs, dense_scores, bm25_map, top_k=max(k, fetch_k))

        def _normalize_department(value: str) -> str:
            return str(value or "").strip().lower()

        target_department = _normalize_department(department)
        filtered_docs, filtered_scores = [], []
        for d, s in zip(docs, fused_scores):
            doc_department = _normalize_department(d.metadata.get("department", ""))
            if target_department and doc_department != target_department:
                continue
            filtered_docs.append(d)
            filtered_scores.append(float(s))

        ranked_docs = []
        for d, s in zip(filtered_docs, filtered_scores):
            sim = float(s)
            freshness = self._calculate_freshness(d)
            if sim < self.config.freshness_trigger_similarity:
                score = sim * (1 - self.config.freshness_weight) + freshness * self.config.freshness_weight
            else:
                score = sim
            ranked_docs.append((d, score))

        ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)[:k]
        return [x[0] for x in ranked_docs], [float(x[1]) for x in ranked_docs]

    def _history_file(self, user: str, department: str, session_id: str) -> Path:
        safe = lambda s: re.sub(r"[^a-zA-Z0-9_-]", "_", s or "default")
        return Path(self.config.chat_history_dir) / f"chat_{safe(user)}_{safe(department)}_{safe(session_id)}.jsonl"

    def load_history(self, user: str, department: str, session_id: str) -> List[Dict[str, str]]:
        p = self._history_file(user, department, session_id)
        if not p.exists():
            return []
        out: List[Dict[str, str]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("content"):
                    out.append({"role": str(obj.get("role", "user")), "content": str(obj.get("content", ""))})
            except Exception:
                continue
        return out

    def save_history(self, user: str, department: str, session_id: str, messages: List[Dict[str, str]]) -> None:
        p = self._history_file(user, department, session_id)
        trimmed = messages[-self.config.chat_history_max_turns * 2 :]
        with self._history_lock:
            lines = [json.dumps(ChatMessage(role=m["role"], content=m["content"]).model_dump(), ensure_ascii=False) for m in trimmed]
            p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def answer(self, question: str, docs: List[Document], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        if not docs:
            return "根据当前权限和知识库内容，无法回答该问题。"

        try:
            self.init_llm()
        except Exception:
            refs = "\n".join([f"[{i + 1}] {d.page_content[:180]}" for i, d in enumerate(docs[:5])])
            return f"当前未配置大模型密钥，先返回检索摘要：\n{refs}"

        context = "\n\n".join(
            [f"[{i + 1}] {d.page_content}\n(source={d.metadata.get('source_file')}, document_id={d.metadata.get('document_id')}, page={d.metadata.get('page_index')})" for i, d in enumerate(docs)]
        )
        history_text = ""
        if chat_history:
            history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in chat_history[-20:]])

        prompt = (
            "你是企业知识助手。仅根据提供上下文回答，不要编造。\n"
            "如果证据不足，请回答：根据现有知识库信息，无法回答该问题。\n"
            "回答中请使用引用标记 [1][2]。\n\n"
            f"对话历史:\n{history_text}\n\n"
            f"上下文:\n{context}\n\n"
            f"问题: {question}\n\n"
            "回答:"
        )
        resp = self.llm.invoke(prompt)
        return (getattr(resp, "content", None) or str(resp)).strip()
