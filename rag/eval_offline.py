from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import RAGConfig
from .rag_system import EnterpriseRAG


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def _basename(value: str) -> str:
    value = _normalize_text(value)
    if not value:
        return ""
    return value.split("/")[-1].split("\\")[-1]


def _file_variants(name: str) -> Set[str]:
    raw = _basename(name)
    if not raw:
        return set()

    variants = {raw}
    stem = raw
    ext = ""
    if "." in raw:
        stem, ext = raw.rsplit(".", 1)
        variants.add(stem)
        variants.add(ext)

    prefixes = ("public_", "finance_", "general_")
    for v in list(variants):
        for p in prefixes:
            if v.startswith(p):
                variants.add(v[len(p) :])
        parts = v.split("_")
        if len(parts) >= 4 and re.fullmatch(r"[0-9a-f]{8,}", parts[0]):
            stripped = "_".join(parts[2:])
            variants.add(stripped)
            if ext:
                variants.add(f"{stripped}.{ext}")
    return {x for x in variants if x}


def _extract_metadata_keys(doc) -> Set[str]:
    keys = set()
    for field in ("document_id", "parent_document_id", "file_hash", "file_name", "logical_file_name", "source_file"):
        value = str(doc.metadata.get(field, ""))
        if not value:
            continue
        if field in ("file_name", "logical_file_name", "source_file"):
            keys.update(_file_variants(value))
        else:
            keys.add(_normalize_text(value))
    return keys


def _resolve_relevant_keys(sample: Dict) -> Set[str]:
    rel_ids = {_normalize_text(str(x)) for x in sample.get("relevant_document_ids", []) if str(x).strip()}
    rel_files: Set[str] = set()
    for x in sample.get("relevant_file_names", []):
        if not str(x).strip():
            continue
        rel_files.update(_file_variants(str(x)))
    return rel_ids.union(rel_files)


def _extract_predicted_keys(docs) -> List[Set[str]]:
    return [_extract_metadata_keys(d) for d in docs]


def recall_at_k(relevant: Set[str], predicted: List[Set[str]], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    predicted_keys: Set[str] = set()
    for key_set in predicted[:k]:
        predicted_keys.update(key_set)
    return len(relevant.intersection(predicted_keys)) / len(relevant)


def precision_at_k(relevant: Set[str], predicted: List[Set[str]], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = predicted[:k]
    if not topk:
        return 0.0
    predicted_keys: Set[str] = set()
    for key_set in topk:
        predicted_keys.update(key_set)
    return 1.0 if relevant.intersection(predicted_keys) else 0.0


def mrr(relevant: Set[str], predicted: List[Set[str]]) -> float:
    for i, key_set in enumerate(predicted, start=1):
        if relevant.intersection(key_set):
            return 1.0 / i
    return 0.0


def run_eval(dataset_path: str, top_k: int = 10) -> Dict[str, float]:
    cfg = RAGConfig()
    rag = EnterpriseRAG(cfg)

    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    samples = data if isinstance(data, list) else data.get("samples", [])

    if not samples:
        raise ValueError("评测集为空")

    rec5_sum, rec10_sum, p5_sum, p10_sum, mrr_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    n = 0

    for idx, s in enumerate(samples, start=1):
        q = str(s.get("query", "")).strip()
        dept = str(s.get("department", "public"))
        relevant = _resolve_relevant_keys(s)

        if not q:
            print(f"[warn] 第 {idx} 条样本 query 为空，已跳过")
            continue
        if not relevant:
            print(f"[warn] 第 {idx} 条样本没有有效标注，已跳过")
            continue

        docs, scores = rag.search(q, department=dept, k=top_k)
        pred = _extract_predicted_keys(docs)

        if idx <= 3:
            print(f"[debug] 样本 {idx}")
            print(f"  query: {q}")
            print(f"  relevant: {sorted(list(relevant))[:8]}")
            print(f"  predicted_top1_keys: {sorted(list(pred[0]))[:8] if pred else []}")
            print(f"  scores: {scores[:3]}")

        rec5_sum += recall_at_k(relevant, pred, 5)
        rec10_sum += recall_at_k(relevant, pred, 10)
        p5_sum += precision_at_k(relevant, pred, 5)
        p10_sum += precision_at_k(relevant, pred, 10)
        mrr_sum += mrr(relevant, pred)
        n += 1

    if n == 0:
        raise ValueError("评测样本均无有效标注，无法计算指标")

    return {
        "samples": n,
        "top_k": top_k,
        "precision@5": p5_sum / n,
        "precision@10": p10_sum / n,
        "recall@5": rec5_sum / n,
        "recall@10": rec10_sum / n,
        "mrr": mrr_sum / n,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 离线评测")
    parser.add_argument("--dataset", required=True, help="评测集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=10, help="每条 query 的检索数量")
    args = parser.parse_args()

    result = run_eval(args.dataset, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
