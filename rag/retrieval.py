from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .text_cleaning import tokenize_zh_en


class HybridRetriever:
    def __init__(self, alpha: float = 0.7):
        self.alpha = max(0.0, min(1.0, alpha))

    def bm25_score_map(self, query: str, docs: List[Document]) -> Dict[str, float]:
        q = tokenize_zh_en(query)
        if not q or not docs:
            return {}

        tokenized = [tokenize_zh_en(d.page_content) for d in docs]
        n = len(tokenized)
        avgdl = sum(len(t) for t in tokenized) / max(1, n)

        df = Counter()
        for toks in tokenized:
            df.update(set(toks))

        k1, b = 1.5, 0.75
        out: Dict[str, float] = {}

        for d, toks in zip(docs, tokenized):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                nt = df.get(t, 0)
                if nt == 0:
                    continue
                idf = math.log((n - nt + 0.5) / (nt + 0.5) + 1)
                f = tf.get(t, 0)
                denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
                s += idf * ((f * (k1 + 1)) / max(denom, 1e-9))
            key = str(d.metadata.get("chunk_uid") or d.metadata.get("document_id") or id(d))
            out[key] = s
        return out

    @staticmethod
    def _normalize_map(m: Dict[str, float]) -> Dict[str, float]:
        if not m:
            return {}
        vals = list(m.values())
        mn, mx = min(vals), max(vals)
        if abs(mx - mn) < 1e-9:
            return {k: 0.5 for k in m}
        return {k: (v - mn) / (mx - mn) for k, v in m.items()}

    def fuse(
        self,
        dense_docs: List[Document],
        dense_scores: List[float],
        bm25_map: Dict[str, float],
        top_k: int,
    ) -> Tuple[List[Document], List[float]]:
        if not dense_docs:
            return [], []

        dense_map: Dict[str, float] = {}
        by_key: Dict[str, Document] = {}
        for d, s in zip(dense_docs, dense_scores):
            key = str(d.metadata.get("chunk_uid") or d.metadata.get("document_id") or id(d))
            dense_map[key] = float(s)
            by_key[key] = d

        nd = self._normalize_map(dense_map)
        nb = self._normalize_map(bm25_map)

        all_keys = set(nd) | set(nb)
        fused: Dict[str, float] = {}
        for k in all_keys:
            fused[k] = self.alpha * nd.get(k, 0.0) + (1 - self.alpha) * nb.get(k, 0.0)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        docs = [by_key[k] for k, _ in ranked if k in by_key]
        scores = [fused[k] for k, _ in ranked if k in by_key]
        return docs, scores


def build_department_expr(department: str, custom_expr: Optional[str] = None) -> str:
    base = f'department == "{department}"'
    if custom_expr:
        return f"({base}) and ({custom_expr})"
    return base
