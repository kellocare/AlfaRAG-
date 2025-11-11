#!/usr/bin/env python3
# src/preprocessing/chunker.py
"""
Чанкинг websites_updated.xlsx →
  - data/processed/chunks.csv
  - data/processed/chunks.jsonl (для Generator)

Особенности:
- Поддержка .xlsx
- Контекстный префикс: kind: ...; title: ...; text: ...
- Глубокая очистка мусора (?oirutpspid=, tel., footer)
- Чанкинг: 300 слов, overlap=60
- Дедупликация по MD5(text)
- Сохранение JSONL в формате, совместимом с Generator
"""

import argparse
import os
import re
import hashlib
import json
from tqdm import tqdm
import pandas as pd

NOISE_PATTERNS = [
    r'\?oirutpspid=[^&\s]*',
    r'\?oirutpspsc=[^&\s]*',
    r'\?oirutpspjs=[^&\s]*',
    r'#\S*',
    r'tel\.', r'Tel\.', r'тел\.', r'Тел\.', r'тел:', r'Тел:',
    r'©\s*Альфа-Банк',
    r'Пользуясь сайтом',
    r'соглашаетесь с политикой конфиденциальности',
    r'Контакты', r'Поддержка', r'Карта сайта',
    r'Москва', r'Россия',
    r'199\d–202\d',
    r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|pdf)',
    r'\{[^{}]*\}',
]

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for pattern in NOISE_PATTERNS:
        s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    s = re.sub(r'<script.*?</script>', ' ', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<style.*?</style>', ' ', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'([.,;!?])\1+', r'\1', s)
    return s.strip()

def build_contextual_content(kind: str, title: str, text: str) -> str:
    kind = clean_text(kind)
    title = clean_text(title)
    text = clean_text(text)
    return f"kind: {kind}; title: {title}; text: {text}"

def chunk_text_by_words(text: str, words_per_chunk: int = 300, overlap: int = 60) -> list[str]:
    if not text or len(text.strip()) < 20:
        return []
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text] if len(text) >= 50 else []
    step = max(1, words_per_chunk - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Чтение {args.input}...")
    df = pd.read_excel(args.input, engine="openpyxl")  # .xlsx
    required_cols = {"web_id", "url", "kind", "title", "text"}
    if not required_cols.issubset(set(df.columns)):
        raise SystemExit(f"Ожидались колонки: {required_cols}. Есть: {set(df.columns)}")

    df = df.dropna(subset=["web_id", "title", "text", "kind"]).reset_index(drop=True)
    print(f"Обрабатываем {len(df)} документов")

    rows_csv = []
    rows_jsonl = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Чанкинг"):
        try:
            web_id = int(r["web_id"])
        except (ValueError, TypeError):
            continue
        title = str(r.get("title", "")).strip()
        url = str(r.get("url", "")).strip()
        kind = str(r.get("kind", "")).strip()
        raw_text = str(r.get("text", "")).strip()

        full_content = build_contextual_content(kind, title, raw_text)
        chunks = chunk_text_by_words(full_content, words_per_chunk=args.words_per_chunk, overlap=args.overlap)

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{web_id}__{i}"
            # для CSV
            rows_csv.append({
                "web_id": web_id,
                "chunk_id": chunk_id,
                "title": title,
                "url": url,
                "kind": kind,
                "text": chunk_text
            })
            # для JSONL (точно то, что ждёт Generator)
            rows_jsonl.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "url": url,
                "title": title,
                "web_id": web_id,
                "kind": kind
            })

    # Дедупликация по text
    seen_hashes = set()
    unique_csv = []
    unique_jsonl = []
    for csv_row, jsonl_row in zip(rows_csv, rows_jsonl):
        h = hash_text(csv_row["text"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique_csv.append(csv_row)
        unique_jsonl.append(jsonl_row)

    print(f"Удалено дубликатов: {len(rows_csv) - len(unique_csv)}")

    # Сохраняем CSV
    out_df = pd.DataFrame(unique_csv, columns=["web_id", "chunk_id", "title", "url", "kind", "text"])
    out_df.to_csv(args.output, index=False)
    print(f"chunks.csv: {len(out_df)} чанков")

    # Сохраняем JSONL (для Generator)
    jsonl_path = args.output.replace(".csv", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in unique_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"chunks.jsonl: {len(unique_jsonl)} чанков → для Generator")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чанкинг websites_updated.xlsx → chunks.csv + chunks.jsonl")
    parser.add_argument("--input", default="data/raw/websites_updated.xlsx")
    parser.add_argument("--output", default="data/processed/chunks.csv")
    parser.add_argument("--words_per_chunk", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=60)
    args = parser.parse_args()
    main(args)