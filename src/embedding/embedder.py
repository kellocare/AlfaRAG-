import os
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
import threading


class Embedder:
    # Кэш моделей для избежания повторной загрузки
    _models = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'auto'):

        self.model_name = model_name
        self.device = device
        self.model = self._load_model(model_name, device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Инициализирован эмбеддер: {model_name} (размерность: {self.dimension})")

    def _load_model(self, model_name: str, device: str) -> SentenceTransformer:
        with self._lock:
            if model_name not in self._models:
                logger.info(f"Загрузка модели: {model_name}")
                model = SentenceTransformer(model_name, device=device)
                self._models[model_name] = model
            return self._models[model_name]

    def encode(
            self,
            texts: Union[str, List[str]],
            batch_size: int = 32,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = True,
            **kwargs
    ) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            logger.warning("Получен пустой список текстов для кодирования")
            return np.array([])

        logger.info(f"Кодирование {len(texts)} текстов (batch_size={batch_size})")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                **kwargs
            )

            logger.info(f"Успешно создано {len(embeddings)} эмбеддингов")
            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Ошибка при кодировании текстов: {e}")
            raise

    def encode_generator(
            self,
            texts: List[str],
            batch_size: int = 32,
            **kwargs
    ):
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch_texts, batch_size=batch_size, **kwargs)
            yield batch_embeddings

            logger.debug(f"Обработан батч {i // batch_size + 1}/{total_batches}")

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device)
        }


def create_embedder(model_name: str = 'all-MiniLM-L6-v2', **kwargs) -> Embedder:
    return Embedder(model_name=model_name, **kwargs)


if __name__ == "__main__":
    embedder = create_embedder()

    # Тестовые тексты
    test_texts = [
        "Машинное обучение и искусственный интеллект",
        "Глубокое обучение и нейронные сети",
        "Обработка естественного языка NLP",
        "Компьютерное зрение и распознавание образов"
    ]

    print("Информация о модели:", embedder.get_model_info())

    # Кодирование одного текста
    single_embedding = embedder.encode(test_texts[0])
    print(f"Эмбеддинг одного текста: форма {single_embedding.shape}")

    # Кодирование нескольких текстов
    embeddings = embedder.encode(test_texts, batch_size=2)
    print(f"Эмбеддинги нескольких текстов: форма {embeddings.shape}")

    # Постепенная обработка
    print("Постепенная обработка...")
    for i, batch_emb in enumerate(embedder.encode_generator(test_texts, batch_size=1)):
        print(f"[BATCH] {i + 1}: {batch_emb.shape}")