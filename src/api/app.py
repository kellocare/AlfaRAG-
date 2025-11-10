#!/usr/bin/env python3
# src/api/app.py
"""
FastAPI по порту 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List
import os
from loguru import logger


# модуль должен быть в PYTHONPATH
from src.indexer import faiss_index as indexer

app = FastAPI(title="Дора любит рагу")

class SearchItem(BaseModel):
    chunk_id: str
    web_id: str
    title: str
    url: str
    kind: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]

DEFAULT_INDEX_DIR = os.getenv("INDEX_DIR", "indexes")
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=100), model: str = DEFAULT_MODEL, index_dir: str = DEFAULT_INDEX_DIR):
    """
    Поисковая ручка возвращает top-k чанков для запроса q
    """
    try:
        results = indexer.search(q, index_dir=index_dir, model_name=model, k=k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Ошибка в /search: {}", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"query": q, "results": results}

@app.post("/rebuild")
def rebuild(background_tasks: BackgroundTasks, chunks_csv: str = "data/processed/chunks.csv", index_dir: str = DEFAULT_INDEX_DIR, model: str = DEFAULT_MODEL):
    """
    Запускает перестройку индекса в фоне, не блокируя запрос
    """
    # базовые проверки
    if not os.path.exists(chunks_csv):
        raise HTTPException(status_code=400, detail=f"Chunks CSV не найден: {chunks_csv}")
    # фоновая задача
    background_tasks.add_task(indexer.build_index, chunks_csv, index_dir, model)
    return {"status": "rebuild_started", "chunks_csv": chunks_csv, "index_dir": index_dir, "model": model}