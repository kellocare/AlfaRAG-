#!/usr/bin/env python3
# src/preprocessing/chunker.py
"""
Чанкинг websites_updated.xlsx → chunks.csv и chunks.jsonl
Вход: data/raw/websites_updated.xlsx
Выход: data/processed/chunks.csv, data/processed/chunks.jsonl

Особенности и изменения:
- Работает с .xlsx (через openpyxl)
- Контекстный префикс: "kind: {kind}; title: {title}; text: {text}"
- Глубокая очистка: ?oirutpspid=, tel., footer-фразы, JSON-фрагменты (из конфига)
- Чанкинг: 300 слов, overlap=60 (пока не из конфига)
- Фильтрация коротких чанков (<50 симв.)
- Дедупликация по MD5(text)
- НОВОЕ: статистика, безопасные проверки, конфигурация
- НОВОЕ: метрики добавляются в конец CSV и JSONL
- НОВОЕ: проверка на дубликаты chunk_id
- НОВОЕ: логгирование в файл chunker.log
- НОВОЕ: проверка длины text перед чанкингом
- НОВОЕ: унификация clean_text
- НОВОЕ: оба файла (CSV и JSONL) генерируются всегда
"""

import argparse
import os
import re
import hashlib
import json
import logging
from tqdm import tqdm
import pandas as pd


# --- Настройка логгирования ---
def setup_logger(output_dir: str):
    log_path = os.path.join(output_dir, "chunker.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("chunker")


# --- Загрузка NOISE_PATTERNS из конфига (в той же папке) ---
def load_noise_patterns(config_path: str = "src/preprocessing/noise_patterns.json"):
    default_patterns = [
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
        r'\{[^{}]*\}', # JSON-фрагменты
    ]

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                return config.get("noise_patterns", default_patterns)
            except Exception:
                print(f"Ошибка загрузки конфига {config_path}, используем стандартные паттерны")
                return default_patterns
    else:
        print(f"Конфиг {config_path} не найден, используем стандартные паттерны")
        return default_patterns


NOISE_PATTERNS = load_noise_patterns()


def clean_text(s: str) -> str:
    """Удаление HTML и шума"""
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
    """Формирование контекстного контента"""
    kind = clean_text(kind)
    title = clean_text(title)
    text = clean_text(text)
    return f"kind: {kind}; title: {title}; text: {text}"


def chunk_text_by_words(text: str, words_per_chunk: int = 300, overlap: int = 60) -> list[str]:
    """Разбиение на чанки по словам"""
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
    """Хэширование текста для дедупликации"""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger = setup_logger(os.path.dirname(args.output))

    logger.info(f"Чтение {args.input}...")
    df = pd.read_excel(args.input, engine="openpyxl")  # .xlsx
    required_cols = {"web_id", "url", "kind", "title", "text"}
    if not required_cols.issubset(set(df.columns)):
        raise SystemExit(f"Ожидались колонки: {required_cols}. Есть: {set(df.columns)}")

    # НОВОЕ: убираем dropna, проверяем в цикле
    logger.info(f"Обрабатываем {len(df)} документов")

    rows_csv = []
    rows_jsonl = []

    web_ids = set()
    seen_chunk_ids = set()  # НОВОЕ: проверка дубликатов chunk_id

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Чанкинг"):
        # Проверки
        web_id_raw = r.get("web_id")
        if pd.isna(web_id_raw):
            logger.warning(f"Пропуск строки: web_id пустой: {r.name}")
            continue
        try:
            web_id = int(web_id_raw)
        except (ValueError, TypeError):
            logger.warning(f"Пропуск строки: web_id не конвертируется в int: {web_id_raw}")
            continue

        title = r.get("title")
        url = r.get("url")
        kind = r.get("kind")
        raw_text = r.get("text")

        if pd.isna(title) or pd.isna(kind) or pd.isna(raw_text):
            logger.warning(f"Пропуск строки: один из обязательных полей пуст: {r.name}")
            continue

        title = str(title).strip()
        url = str(url).strip() if pd.notna(url) else ""
        kind = str(kind).strip()
        raw_text = str(raw_text).strip()

        # НОВОЕ: проверка длины текста перед чанкингом
        if len(raw_text) < 50:
            logger.info(f"Пропуск строки: текст слишком короткий: {r.name}")
            continue

        full_content = build_contextual_content(kind, title, raw_text)
        chunks = chunk_text_by_words(full_content, words_per_chunk=args.words_per_chunk, overlap=args.overlap)

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{web_id}__{i}"

            # НОВОЕ: проверка дубликата chunk_id
            if chunk_id in seen_chunk_ids:
                logger.warning(f"Пропуск дубликата chunk_id: {chunk_id}")
                continue
            seen_chunk_ids.add(chunk_id)

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

        web_ids.add(web_id)

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

    logger.info(f"Удалено дубликатов: {len(rows_csv) - len(unique_csv)}")

    # --- СТАТИСТИКА ---
    chunks_count = len(unique_csv)
    total_len = sum(len(row["text"]) for row in unique_csv)
    avg_len = total_len / chunks_count if chunks_count > 0 else 0
    uniq_web_ids = len(web_ids)

    logger.info(f"Статистика:")
    logger.info(f"   - Чанков: {chunks_count}")
    logger.info(f"   - Средняя длина: {avg_len:.2f}")
    logger.info(f"   - Уникальных web_id: {uniq_web_ids}")

    # --- СОХРАНЕНИЕ CSV ---
    out_df = pd.DataFrame(unique_csv, columns=["web_id", "chunk_id", "title", "url", "kind", "text"])
    out_df.to_csv(args.output, index=False)
    logger.info(f"chunks.csv: {len(out_df)} чанков")

    # --- ДОБАВЛЕНИЕ СТАТИСТИКИ В КОНЕЦ CSV ---
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(f"\n#chunks_count,{chunks_count}\n")
        f.write(f"#avg_len,{avg_len:.2f}\n")
        f.write(f"#uniq_web_ids,{uniq_web_ids}\n")
    logger.info(f"Метрики добавлены в конец {args.output}")

    # --- СОХРАНЕНИЕ JSONL ---
    jsonl_path = args.output.replace(".csv", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in unique_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # --- ДОБАВЛЕНИЕ СТАТИСТИКИ В КОНЕЦ JSONL ---
        stats = {
            "chunks_count": chunks_count,
            "avg_len": round(avg_len, 2),
            "uniq_web_ids": uniq_web_ids
        }
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
    logger.info(f"chunks.jsonl: {len(unique_jsonl)} чанков → для Generator")
    logger.info(f"Метрики добавлены в конец {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чанкинг websites_updated.xlsx → chunks.csv + chunks.jsonl")
    parser.add_argument("--input", default="data/raw/websites_updated.xlsx")
    parser.add_argument("--output", default="data/processed/chunks.csv")
    parser.add_argument("--words_per_chunk", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=60)
    args = parser.parse_args()
    main(args)