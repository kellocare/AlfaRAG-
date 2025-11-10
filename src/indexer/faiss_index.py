#!/usr/bin/env python3
# src/indexer/faiss_index.py
"""
Построение индекса и поиск, поддерживает faiss или fallback на hnswlib

Работаем в два этапа:
1. Строим индекс
python src/indexer/faiss_index.py --build --chunks data/processed/chunks.csv --index_dir indexes --model all-MiniLM-L6-v2

2. Проводим быстрый поиск:
python src/indexer/faiss_index.py --search "Как оформить возврат?" --k 5 --index_dir indexes
"""

import argparse
import os
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from loguru import logger

# пробуем импортировать faiss, иначе пойдет hnswlib
_has_faiss = False
try:
    import faiss
    _has_faiss = True
except Exception:
    _has_faiss = False

_hnswlib = None
if not _has_faiss:
    try:
        import hnswlib
        _hnswlib = hnswlib
    except Exception:
        _hnswlib = None

EMBED_DIM_BY_MODEL = {
    # модель -> эмбединг-измерение (если не знаем — можно вычислить динамически)
    "all-MiniLM-L6-v2": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

def compute_embeddings(texts: List[str], model_name: str, batch_size: int = 64):
    """Вычисление эмбеддингов через sentence-transformers"""
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32"), model

def build_faiss_index(embs: np.ndarray, index_path: str, ef_construction: int = 200):
    """Стройка HNSW faiss-index и сохранение на диск"""
    d = embs.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = ef_construction
    index.add(embs)
    faiss.write_index(index, index_path)
    return index

def build_hnswlib_index(embs: np.ndarray, index_path: str, space: str = "cosine", ef_construction: int = 200):
    """Формирование индекса hnswlib и сохранение бинарником"""
    dim = embs.shape[1]
    num_elements = embs.shape[0]
    p = _hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=32)
    # hnswlib конечно любит не-нормализованные элементы, но мы и так уже нормализовали
    p.add_items(embs, np.arange(num_elements))
    p.set_ef(50)
    p.save_index(index_path)
    return p

def build_index(chunks_csv: str, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    """Главная функция для построения эмбеддингов + индекса + сохранения метаданных"""
    os.makedirs(index_dir, exist_ok=True)
    df = pd.read_csv(chunks_csv)
    if df.empty:
        raise ValueError("Chunks CSV пуст...")

    texts = df["text"].fillna("").tolist()
    logger.info("Вычисление эмбеддингов для {} чанков с помощью модели {}", len(texts), model_name)
    embs, model = compute_embeddings(texts, model_name=model_name)

    # save embeddings (опционально)
    emb_path = os.path.join(index_dir, "embeddings.npy")
    np.save(emb_path, embs)
    logger.info("Эмбеддинги сохранены в {}", emb_path)

    # build index (faiss preferred)
    if _has_faiss:
        index_file = os.path.join(index_dir, "faiss_hnsw.index")
        logger.info("Построение faiss index -> {}", index_file)
        idx = build_faiss_index(embs, index_file)
    elif _hnswlib is not None:
        index_file = os.path.join(index_dir, "hnswlib.index")
        logger.info("Построение hnswlib index -> {}", index_file)
        idx = build_hnswlib_index(embs, index_file)
    else:
        raise RuntimeError("Ни faiss, ни hnswlib не доступны. Установите faiss-cpu или hnswlib.")

    # сохраняем метаданные (DataFrame)
    meta_path = os.path.join(index_dir, "chunks_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(df.to_dict(orient="list"), f)
    logger.info("Метаданные сохранены в {}", meta_path)

    # сохраняем параметры модели
    model_info = {"model_name": model_name, "embedding_dim": embs.shape[1]}
    with open(os.path.join(index_dir, "index_info.pkl"), "wb") as f:
        pickle.dump(model_info, f)
    logger.info("Построение индексов завершено.")
    return True

def load_index_and_meta(index_dir: str):
    """Загрузка индекса (faiss или hnswlib) и метаданные"""
    meta_path = os.path.join(index_dir, "chunks_meta.pkl")
    info_path = os.path.join(index_dir, "index_info.pkl")
    if not os.path.exists(meta_path) or not os.path.exists(info_path):
        raise FileNotFoundError("Индекс или метаданные не найдены. Сначала выполните --build")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    with open(info_path, "rb") as f:
        info = pickle.load(f)

    index_obj = None
    if _has_faiss:
        idx_file = os.path.join(index_dir, "faiss_hnsw.index")
        if os.path.exists(idx_file):
            index_obj = faiss.read_index(idx_file)
    if index_obj is None and _hnswlib is not None:
        idx_file = os.path.join(index_dir, "hnswlib.index")
        if os.path.exists(idx_file):
            p = _hnswlib.Index(space="cosine", dim=info["embedding_dim"])
            p.load_index(idx_file)
            index_obj = p

    if index_obj is None:
        raise FileNotFoundError("Не найден подходящий файл индекса (faiss или hnswlib).")
    return index_obj, meta, info

def search(query: str, index_dir: str, model_name: str = "all-MiniLM-L6-v2", k: int = 5):
    """Поиск: вычисляем эмбеддинг запроса и делаем ANN-search"""
    # загружаем модель (sentence-transformers) и индекс
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    idx, meta, info = load_index_and_meta(index_dir)

    results = []
    if _has_faiss and isinstance(idx, faiss.Index):
        try:
            idx.hnsw.efSearch = max(50, k * 10)
        except Exception:
            pass
        D, I = idx.search(q_emb, k)
        D = D[0].tolist()
        I = I[0].tolist()
        for dist, i in zip(D, I):
            if i < 0:
                continue
            item = {
                "chunk_id": meta["chunk_id"][i],
                "web_id": meta["web_id"][i],
                "title": meta.get("title", [""])[i],
                "url": meta.get("url", [""])[i],
                "kind": meta.get("kind", [""])[i],
                "text": meta["text"][i],
                "score": float(dist)
            }
            results.append(item)
    else:
        # hnswlib
        labels, distances = idx.knn_query(q_emb, k=k)
        for label, dist in zip(labels[0].tolist(), distances[0].tolist()):
            i = int(label)
            item = {
                "chunk_id": meta["chunk_id"][i],
                "web_id": meta["web_id"][i],
                "title": meta.get("title", [""])[i],
                "url": meta.get("url", [""])[i],
                "kind": meta.get("kind", [""])[i],
                "text": meta["text"][i],
                "score": float(dist)
            }
            results.append(item)

    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--chunks", type=str, default="data/processed/chunks.csv")
    p.add_argument("--index_dir", type=str, default="indexes")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--search", type=str)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    if args.build:
        print("Строим index...")
        build_index(args.chunks, args.index_dir, model_name=args.model)
        print("Готово!")

    if args.search:
        print("Запуск поиска...")
        res = search(args.search, args.index_dir, model_name=args.model, k=args.k)
        for i, r in enumerate(res, 1):
            print(f"{i}. [{r['chunk_id']}] score={r['score']:.4f} url={r['url']}")
            print(r['text'][:300].replace("\n", " ") + "...")
            print("-" * 80)