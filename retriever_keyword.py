# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Chunk:
    chunk_id: str   # e.g. "p2-c03"
    page: str       # "p1" / "p2" / "p3"
    text: str

def _tokenize(s: str) -> List[str]:
    s = s.lower()
    raw = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", s)
    tokens: List[str] = []

    for t in raw:
        # 英文数字直接用
        if re.fullmatch(r"[a-z0-9]+", t):
            tokens.append(t)
            continue

        # 中文：用 2-gram（双字切片）+ 本身（短词也能命中）
        t = t.strip()
        if len(t) == 1:
            tokens.append(t)
        else:
            tokens.append(t)  # 保留原词块（有时也有用）
            for i in range(len(t) - 1):
                tokens.append(t[i:i+2])

    return tokens

def build_chunks(pages: Dict[str, str], max_len: int = 220, overlap: int = 40) -> List[Chunk]:
    """
    把 p1/p2/p3 的长文本切成小块，便于检索。
    max_len/overlap 是字符级（够用，先别折腾）。
    """
    chunks: List[Chunk] = []
    for page, text in pages.items():
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + max_len)
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(chunk_id=f"{page}-c{idx:02d}", page=page, text=piece))
                idx += 1
            start = max(0, end - overlap)
            if end >= len(text):
                break
    return chunks

def retrieve_topk(query: str, chunks: List[Chunk], top_k: int = 3) -> List[Tuple[Chunk, int]]:
    """
    朴素检索：按 token 重叠数打分（先把可观测跑通）。
    """
    q_tokens = set(_tokenize(query))
    scored: List[Tuple[Chunk, int]] = []
    for ch in chunks:
        c_tokens = set(_tokenize(ch.text))
        score = len(q_tokens & c_tokens)
        scored.append((ch, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    # 调试阶段：不过滤 0 分，看所有 Top-k
    top = scored[:top_k]
    return top
