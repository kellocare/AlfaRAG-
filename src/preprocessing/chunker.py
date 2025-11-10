#!/usr/bin/env python3
# src/preprocessing/chunker.py
"""
Вход: websites_updated.csv с колонками web_id,url,kind,title,text
Выход: chunks.csv с колонками web_id,chunk_id,title,url,kind,text
"""

import argparse
import os
import re
from tqdm import tqdm
import pandas as pd

def clean_text(s: str) -> str:
    """Простая очистка текста посредством удаления html-теги и лишних пробелов"""
    if not isinstance(s, str):
        return ""
    s = re.sub(r'<script.*?</script>', ' ', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<style.*?</style>', ' ', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<[^>]+>', ' ', s)  # убираем теги
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def chunk_text_by_words(text: str, words_per_chunk: int = 250, overlap: int = 50):
    """Разбиваем текст на чанки по словам с указанным параметром overlap"""
    words = text.split()
    if not words:
        return []
    step = max(1, words_per_chunk - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_csv(args.input)
    expected_cols = {"web_id", "url", "kind", "title", "text"}
    if not expected_cols.issubset(set(df.columns)):
        raise SystemExit(f"Ожидаемые колонки в input CSV: {expected_cols}. Найдено: {set(df.columns)}")

    rows = []
    # итерация по строкам и чанкинг
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing docs"):
        web_id = r["web_id"]
        title = r.get("title", "") or ""
        url = r.get("url", "") or ""
        kind = r.get("kind", "") or ""
        text = clean_text(r.get("text", "") or "")
        chunks = chunk_text_by_words(text, words_per_chunk=args.words_per_chunk, overlap=args.overlap)
        for i, c in enumerate(chunks):
            rows.append({
                "web_id": web_id,
                "chunk_id": f"{web_id}__{i}",
                "title": title,
                "url": url,
                "kind": kind,
                "text": c
            })

    out_df = pd.DataFrame(rows, columns=["web_id", "chunk_id", "title", "url", "kind", "text"])
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} chunks to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--words_per_chunk", type=int, default=250)
    p.add_argument("--overlap", type=int, default=50)
    args = p.parse_args()
    main(args)