from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_header_footer_noise(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^第?\s*\d+\s*页(\s*/\s*共?\s*\d+\s*页)?$", s):
            continue
        if re.match(r"^page\s*\d+(\s*/\s*\d+)?$", s, re.IGNORECASE):
            continue
        if re.match(r"^(机密|保密|仅供内部|内部资料)", s):
            continue
        out.append(s)
    return out


def clean_pdf_text(text: str) -> str:
    lines = remove_header_footer_noise(text.splitlines())
    merged = "\n".join(lines)
    merged = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", merged)
    return normalize_whitespace(merged)


def clean_docx_text(text: str) -> str:
    return normalize_whitespace(text)


def clean_txt_text(text: str) -> str:
    return normalize_whitespace(text)


def tokenize_zh_en(text: str) -> List[str]:
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    en_words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return zh_chars + en_words


def has_heading(text: str) -> bool:
    heading_patterns = [
        r"^#{1,6}\s+.+$",
        r"^\d+(\.\d+)*\s+.+$",
        r"^[一二三四五六七八九十]+、.+$",
        r"^\(?[一二三四五六七八九十]+\)\s*.+$",
        r"^第[一二三四五六七八九十百]+[章节条].+$",
    ]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for ln in lines[:120]:
        for p in heading_patterns:
            if re.match(p, ln):
                return True
    return False
