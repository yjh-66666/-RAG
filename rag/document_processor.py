from __future__ import annotations

import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig
from .models import DocumentSource
from .text_cleaning import clean_docx_text, clean_pdf_text, clean_txt_text, has_heading
from .utils import current_user, file_sha256


class DocumentProcessor:
    SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

    def __init__(self, config: RAGConfig):
        self.config = config
        self._resume_cache = {}

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif v is None:
                safe[k] = ""
            else:
                safe[k] = json.dumps(v, ensure_ascii=False)
        return safe

    @staticmethod
    def _mask_pii(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"(?<!\d)(1[3-9]\d{9})(?!\d)", "[手机号已脱敏]", text)
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱已脱敏]", text)
        text = re.sub(r"(\d{3})-?\d{4}-?(\d{4})", r"\1****\2", text)
        text = re.sub(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)", "[证件号已脱敏]", text)
        return text

    def _resume_cache_file(self) -> Path:
        return Path(self.config.metadata_db_path) / "ingest_resume.json"

    def _load_resume_cache(self) -> Dict[str, Any]:
        return {}

    def _persist_resume_cache(self) -> None:
        return None

    def _cache_key(self, file_path: Path) -> str:
        return file_sha256(file_path)

    def _get_cached_source(self, file_path: Path) -> Optional[DocumentSource]:
        key = self._cache_key(file_path)
        item = self._resume_cache.get(key)
        if not isinstance(item, dict):
            return None
        try:
            source = DocumentSource.model_validate(item["source"])
            return source
        except Exception:
            return None

    def _save_cached_source(self, file_path: Path, source: DocumentSource) -> None:
        key = self._cache_key(file_path)
        self._resume_cache[key] = {
            "file_path": str(file_path.absolute()),
            "source": source.model_dump(mode="json"),
            "updated_at": datetime.now().isoformat(),
        }
        self._persist_resume_cache()

    @staticmethod
    def _infer_logical_name(file_name: str) -> str:
        name = (file_name or "").strip()
        if not name:
            return ""
        lower = name.lower()
        if lower.endswith((".txt", ".pdf", ".docx")):
            stem, ext = name.rsplit(".", 1)
            parts = stem.split("_")
            if len(parts) >= 4 and re.fullmatch(r"[0-9a-fA-F-]{8,}", parts[0]):
                return "_".join(parts[2:]) + f".{ext}"
        return name

    @staticmethod
    def _infer_logical_department(file_name: str, department: str) -> str:
        name = (file_name or "").strip().lower()
        if name.startswith("finance_"):
            return "finance"
        if name.startswith("public_"):
            return "public"
        return (department or "").strip().lower() or department

    def _extract_pdf_tables_with_marker(self, file_path: Path) -> Dict[int, str]:
        _ = file_path
        table_md_by_page: Dict[int, str] = {}
        try:
            import marker  # type: ignore

            _ = marker
        except Exception:
            return table_md_by_page
        return table_md_by_page

    def _extract_pdf_ocr_with_paddle(self, file_path: Path) -> Dict[int, List[str]]:
        ocr_by_page: Dict[int, List[str]] = {}
        try:
            from paddleocr import PaddleOCR  # type: ignore
            import fitz  # type: ignore

            ocr = PaddleOCR(use_angle_cls=True, lang="ch")
            doc = fitz.open(str(file_path))
            for i, page in enumerate(doc):
                images = page.get_images(full=True)
                texts: List[str] = []
                for img in images:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    img_bytes = base.get("image")
                    if not img_bytes:
                        continue
                    result = ocr.ocr(img_bytes, cls=True)
                    lines: List[str] = []
                    for block in result or []:
                        for item in block or []:
                            if len(item) >= 2 and item[1]:
                                lines.append(str(item[1][0]))
                    if lines:
                        texts.append("\n".join(lines))
                if texts:
                    ocr_by_page[i] = texts
        except Exception:
            return ocr_by_page
        return ocr_by_page

    def _pdf_font_assist_headings(self, file_path: Path) -> List[str]:
        headings: List[str] = []
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words(extra_attrs=["size"])
                    if not words:
                        continue
                    sizes = [float(w.get("size", 0)) for w in words if w.get("size") is not None]
                    if not sizes:
                        continue
                    threshold = max(sizes) * 0.95
                    large_words = [w.get("text", "") for w in words if float(w.get("size", 0)) >= threshold]
                    if large_words:
                        line = " ".join(large_words).strip()
                        if line and len(line) <= 80:
                            headings.append(line)
        except Exception:
            return headings
        return headings

    def _load_single(self, file_path: Path, department: str, custom_metadata: Dict[str, Any]) -> Tuple[List[Document], DocumentSource]:
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTS:
            raise ValueError(f"不支持文件格式: {ext}")

        _ = None

        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()

        pdf_table_md = self._extract_pdf_tables_with_marker(file_path) if ext == ".pdf" else {}
        pdf_ocr_map = self._extract_pdf_ocr_with_paddle(file_path) if ext == ".pdf" else {}
        font_headings = self._pdf_font_assist_headings(file_path) if ext == ".pdf" else []

        source = DocumentSource(
            file_path=str(file_path.absolute()),
            file_name=file_path.name,
            ext=ext,
            file_size=file_path.stat().st_size,
            file_hash=file_sha256(file_path),
            uploaded_by=current_user(),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            department=self._infer_logical_department(file_path.name, department),
            custom_metadata=custom_metadata,
        )

        enriched: List[Document] = []
        for idx, d in enumerate(docs):
            text = d.page_content or ""
            has_table_md = False
            has_ocr = False

            if ext == ".pdf":
                text = clean_pdf_text(text)
                if idx in pdf_table_md and pdf_table_md[idx]:
                    text = f"{text}\n\n[PDF表格Markdown]\n{pdf_table_md[idx]}"
                    has_table_md = True
                if idx in pdf_ocr_map and pdf_ocr_map[idx]:
                    text = f"{text}\n\n[PDF页面图片OCR]\n" + "\n".join(pdf_ocr_map[idx])
                    has_ocr = True
            elif ext == ".docx":
                text = clean_docx_text(text)
            else:
                text = clean_txt_text(text)

            text = self._mask_pii(text)

            metadata = {
                **(d.metadata or {}),
                "document_id": source.document_id,
                "source_file": source.file_path,
                "file_name": self._infer_logical_name(source.file_name),
                "logical_file_name": self._infer_logical_name(source.file_name),
                "ext": source.ext,
                "department": source.department,
                "uploaded_by": source.uploaded_by,
                "upload_time": source.upload_time.isoformat(),
                "last_modified": source.last_modified.isoformat(),
                "file_hash": source.file_hash,
                "custom_metadata": source.custom_metadata,
                "pdf_font_heading_hints": font_headings[:20],
                "page_index": idx,
                "has_table_markdown": has_table_md,
                "has_ocr_text": has_ocr,
                "ocr_confidence": None,
                "extract_pipeline_version": "v2.0",
                "parent_document_id": source.document_id,
                "chunk_type": "page",
            }
            enriched.append(Document(page_content=text, metadata=self._sanitize_metadata(metadata)))

        return enriched, source

    def _load_cached_documents(self, source: DocumentSource, department: str, custom_metadata: Dict[str, Any]) -> List[Document]:
        file_path = Path(source.file_path)
        if not file_path.exists():
            return []
        ext = source.ext
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()
        out: List[Document] = []
        for idx, d in enumerate(docs):
            text = self._mask_pii(self._clean_text_by_ext(d.page_content or "", ext))
            meta = {
                **(d.metadata or {}),
                "document_id": source.document_id,
                "source_file": source.file_path,
                "file_name": self._infer_logical_name(source.file_name),
                "logical_file_name": self._infer_logical_name(source.file_name),
                "ext": source.ext,
                "department": self._infer_logical_department(source.file_name, department),
                "uploaded_by": source.uploaded_by,
                "upload_time": source.upload_time.isoformat(),
                "last_modified": source.last_modified.isoformat(),
                "file_hash": source.file_hash,
                "custom_metadata": custom_metadata,
                "page_index": idx,
                "parent_document_id": source.document_id,
                "chunk_type": "page",
                "resume_cached": True,
            }
            out.append(Document(page_content=text, metadata=self._sanitize_metadata(meta)))
        return out

    @staticmethod
    def _clean_text_by_ext(text: str, ext: str) -> str:
        if ext == ".pdf":
            return clean_pdf_text(text)
        if ext == ".docx":
            return clean_docx_text(text)
        return clean_txt_text(text)

    def ingest_uploaded_files(self, files: List[Path], department: str, custom_metadata: Dict[str, Any]) -> Tuple[List[Document], Dict[str, DocumentSource]]:
        if len(files) > self.config.max_upload_files:
            raise ValueError(f"单次最多上传 {self.config.max_upload_files} 个文件")

        all_docs: List[Document] = []
        source_map: Dict[str, DocumentSource] = {}

        with ThreadPoolExecutor(max_workers=min(4, max(1, len(files)))) as executor:
            future_map = {
                executor.submit(self._load_single, f, department, custom_metadata): f
                for f in files
            }
            for future in as_completed(future_map):
                docs, source = future.result()
                all_docs.extend(docs)
                source_map[source.document_id] = source

        return all_docs, source_map

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        out: List[Document] = []
        for d in documents:
            text = (d.page_content or "").strip()
            if not text:
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            )
            pieces = splitter.split_text(text)
            if len(pieces) == 1 and len(pieces[0]) > self.config.coarse_chunk_size * 2:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.coarse_chunk_size,
                    chunk_overlap=self.config.coarse_chunk_overlap,
                    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
                )
                pieces = splitter.split_text(text)

            parent_id = str(d.metadata.get("document_id") or d.metadata.get("parent_document_id") or "")
            heading_mode = has_heading(text) or bool(d.metadata.get("pdf_font_heading_hints"))

            for i, p in enumerate(pieces):
                meta = {
                    **d.metadata,
                    "parent_document_id": parent_id,
                    "chunk_index": i,
                    "heading_mode": heading_mode,
                    "chunk_uid": str(uuid.uuid4()),
                    "chunk_type": "child",
                    "chunk_length": len(p),
                }
                out.append(Document(page_content=self._mask_pii(p), metadata=meta))

        return out

    @staticmethod
    def persist_metadata(metadata_dir: str, sources: Dict[str, DocumentSource]) -> None:
        os.makedirs(metadata_dir, exist_ok=True)
        path = Path(metadata_dir) / "metadata.json"
        payload = {k: v.model_dump(mode="json") for k, v in sources.items()}
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                old = {}
            old.update(payload)
            payload = old
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
