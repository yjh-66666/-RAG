from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma, Milvus
from langchain_core.documents import Document

from .config import RAGConfig


class VectorStoreManager:
    def __init__(self, config: RAGConfig, embeddings: Any):
        self.config = config
        self.embeddings = embeddings
        self.vectorstore: Optional[Any] = None

    def _milvus_connection(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "host": self.config.milvus_host,
            "port": str(self.config.milvus_port),
            "db_name": self.config.milvus_db_name,
        }
        if self.config.milvus_user:
            args["user"] = self.config.milvus_user
        if self.config.milvus_password:
            args["password"] = self.config.milvus_password
        return args

    def build_or_load(self, documents: Optional[List[Document]] = None) -> None:
        if self.config.vectorstore_type == "milvus":
            if documents:
                self.vectorstore = Milvus.from_documents(
                    documents,
                    self.embeddings,
                    connection_args=self._milvus_connection(),
                    collection_name=self.config.milvus_collection,
                )
            else:
                self.vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    connection_args=self._milvus_connection(),
                    collection_name=self.config.milvus_collection,
                )
            return

        if documents:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.chroma_path,
                collection_name=self.config.chroma_collection,
            )
        else:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.config.chroma_path,
                collection_name=self.config.chroma_collection,
            )

    def similarity_search_with_score(self, query: str, k: int, expr: Optional[str] = None) -> List[Tuple[Document, float]]:
        if self.vectorstore is None:
            raise RuntimeError("向量库尚未初始化")

        if self.config.vectorstore_type == "milvus":
            kwargs: Dict[str, Any] = {"k": k}
            if expr:
                kwargs["expr"] = expr
            return self.vectorstore.similarity_search_with_score(query, **kwargs)

        # Chroma 不支持 Milvus expr，改为先取后过滤（在上层做）
        return self.vectorstore.similarity_search_with_score(query, k=k)
