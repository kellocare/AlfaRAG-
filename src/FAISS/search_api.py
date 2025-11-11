from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="Vector Search API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int


class VectorSearchEngine:
    def __init__(self, index_dir="indexes", index_name="my_index"):
        self.index_dir = index_dir
        self.index_name = index_name
        self.model = None
        self.index = None
        self.metadata = None
        self.loaded = False

    def load_index(self):
        """Загрузка индекса и метаданных"""
        try:
            index_path = os.path.join(self.index_dir, f"{self.index_name}.faiss")
            meta_path = os.path.join(self.index_dir, f"{self.index_name}_meta.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                raise FileNotFoundError("Index files not found")

            # Загрузка модели
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            # Загрузка индекса
            self.index = faiss.read_index(index_path)

            # Загрузка метаданных
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            self.loaded = True
            print(f"Index loaded successfully: {self.index.ntotal} vectors")

        except Exception as e:
            print(f"Error loading index: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Поиск похожих текстов"""
        if not self.loaded:
            self.load_index()

        # Создание эмбеддинга для запроса
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Поиск в индексе
        scores, indices = self.index.search(query_embedding, top_k)

        # Формирование результатов
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1:  # FAISS возвращает -1 если нет результата
                text = self.metadata['texts'][idx]
                results.append(SearchResult(
                    text=text,
                    score=float(score),
                    rank=rank + 1
                ))

        return results


# Инициализация поискового движка
search_engine = VectorSearchEngine()


@app.on_event("startup")
async def startup_event():
    """Загрузка индекса при старте приложения"""
    search_engine.load_index()


@app.get("/")
async def root():
    return {"message": "Vector Search API", "status": "running"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Эндпоинт поиска похожих текстов"""
    try:
        results = search_engine.search(request.query, request.top_k)
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/health")
async def health():
    """Проверка статуса API"""
    return {
        "status": "healthy",
        "index_loaded": search_engine.loaded,
        "index_size": search_engine.index.ntotal if search_engine.loaded else 0
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)