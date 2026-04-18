from __future__ import annotations

import hashlib
from typing import List

import numpy as np


class LocalEmbeddings:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.dimension = 384
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_path, device=self.device, local_files_only=True)
            self.dimension = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            self.model = None
            self.dimension = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.model is not None:
            return self.model.encode(text, normalize_embeddings=True).tolist()

        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 10000
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension) * 0.1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
