#!/usr/bin/env python3
# src/preprocessing/chunker.py
"""
Чанкинг websites_updated.xlsx или websites_updated.csv → chunks.csv и chunks.jsonl
Вход: data/raw/websites_updated.xlsx или .csv
Выход: data/processed/chunks.csv, data/processed/chunks.jsonl

Особенности:
- Работает с .xlsx (через openpyxl) и .csv (через pandas)
- Контекстный префикс: kind, title — отдельно, text — только текст
- Глубокая очистка: ?oirutpspid=, tel., footer-фразы, JSON-фрагменты (из конфига)
- Семантический чанкирование: через Sentence Transformers + Similarity (быстрее, чем кластеризация)
- Фильтрация коротких чанков (<100 симв.)
- Дедупликация по MD5(text)
- НОВОЕ: метрики добавляются в конец CSV и JSONL
- НОВОЕ: логгирование в файл chunker.log
- НОВОЕ: оба файла (CSV и JSONL) генерируются всегда
- НОВОЕ: автоматический выбор CPU/GPU
- НОВОЕ: исправлена ошибка os.makedirs при пустом dirname
"""

import argparse
import os
import re
import hashlib
import json
import logging
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


# --- Автоматический выбор устройства ---
def choose_device():
    if torch.cuda.is_available():
        # Проверим VRAM (в ГБ)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Если VRAM < 4 ГБ — не используем GPU (например, GT 1030)
        if total_memory < 4:
            print(f"⚠️ GPU доступна, но VRAM = {total_memory:.1f} ГБ — используем CPU")
            return 'cpu'
        else:
            print(f"✅ Используем GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
    else:
        print("⚠️ GPU недоступна — используем CPU")
        return 'cpu'


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
                print(f"⚠️ Ошибка загрузки конфига {config_path}, используем стандартные паттерны")
                return default_patterns
    else:
        print(f"⚠️ Конфиг {config_path} не найден, используем стандартные паттерны")
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


def semantic_chunk_by_similarity(text: str, model, max_chunk_len: int = 400, threshold: float = 0.5):
    """
    Семантическое чанкирование через сравнение эмбеддингов соседних предложений
    """
    sentences = [s.strip() for s in re.split(r'\n\s*\n|\. ', text) if s.strip()]
    if not sentences:
        return []

    if len(sentences) < 2:
        # Просто разбиваем по длине
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) < max_chunk_len:
                current_chunk += " " + sent
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sent
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return [c for c in chunks if len(c) >= 100]

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = sentences[0]
    current_emb = embeddings[0].reshape(1, -1)

    for i in range(1, len(sentences)):
        next_emb = embeddings[i].reshape(1, -1)
        similarity = cosine_similarity(current_emb, next_emb)[0][0]

        if similarity > threshold and len(current_chunk) + len(sentences[i]) < max_chunk_len:
            current_chunk += " " + sentences[i]
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentences[i]
            current_emb = next_emb

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= 100]


def hash_text(text: str) -> str:
    """Хэширование текста для дедупликации"""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()


def main(args):
    # Исправленная строка: проверяем, есть ли путь в output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.dirname(args.output))

    logger.info(f"Чтение {args.input}...")

    # Определяем расширение файла
    if args.input.lower().endswith('.xlsx'):
        try:
            df = pd.read_excel(args.input, engine="openpyxl")
        except ImportError:
            raise ImportError("Для чтения .xlsx нужен пакет 'openpyxl'. Установите его: pip install openpyxl")
    elif args.input.lower().endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        raise ValueError(f"Файл {args.input} не является .csv или .xlsx")

    required_cols = {"web_id", "url", "kind", "title", "text"}
    if not required_cols.issubset(set(df.columns)):
        raise SystemExit(f"Ожидались колонки: {required_cols}. Есть: {set(df.columns)}")

    logger.info(f"Обрабатываем {len(df)} документов")

    # Выбираем устройство автоматически
    device = choose_device()

    # Загружаем модель один раз на выбранное устройство
    logger.info(f"Загрузка модели {args.model_name} на {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    rows_csv = []
    rows_jsonl = []

    web_ids = set()
    seen_chunk_ids = set()

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

        if len(raw_text) < 100:
            logger.info(f"Пропуск строки: текст слишком короткий: {r.name}")
            continue

        # Очищаем и чанкуем
        clean_full_text = clean_text(raw_text)
        chunks = semantic_chunk_by_similarity(
            clean_full_text,
            model,
            max_chunk_len=args.max_chunk_len,
            threshold=args.threshold
        )

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{web_id}__{i}"

            if chunk_id in seen_chunk_ids:
                logger.warning(f"Пропуск дубликата chunk_id: {chunk_id}")
                continue
            seen_chunk_ids.add(chunk_id)

            # для CSV: text — только текст, без kind и title
            rows_csv.append({
                "web_id": web_id,
                "chunk_id": chunk_id,
                "title": title,
                "url": url,
                "kind": kind,
                "text": chunk_text
            })
            # для JSONL: тоже самое
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

    logger.info(f"📊 Статистика:")
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
    logger.info(f"✅ Метрики добавлены в конец {args.output}")

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
    logger.info(f"✅ Метрики добавлены в конец {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чанкинг websites_updated.xlsx или .csv → chunks.csv + chunks.jsonl")
    parser.add_argument("--input", default="data/raw/websites_updated.xlsx")
    parser.add_argument("--output", default="data/processed/chunks.csv")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="Модель для эмбеддингов (меньше/быстрее)")
    parser.add_argument("--max_chunk_len", type=int, default=400, help="Максимальная длина чанка (в символах)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Порог схожести для объединения чанков")
    args = parser.parse_args()
    main(args)