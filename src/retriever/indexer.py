import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import pandas as pd

def read_column_to_list(excel_file, column=0, sheet_name=0, header=0, skip_rows=0):
    try:
        df = pd.read_excel(excel_file,
                          sheet_name=sheet_name,
                          header=header,
                          skiprows=skip_rows)

        if isinstance(column, int):
            selected_column = df.iloc[:, column].tolist()
        elif isinstance(column, str):
            selected_column = df[column].tolist()
        else:
            raise ValueError("Параметр 'column' должен быть int (индекс) или str (имя столбца)")

        return selected_column

    except FileNotFoundError:
        print(f"Ошибка: Файл '{excel_file}' не найден")
        return []
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return []

if __name__ == "__main__":
    data_column_list = read_column_to_list("./data/raw/websites_updated.xlsx", column=1)
    print(f"Получено {len(data_column_list)} элементов:")
    for i, item in enumerate(data_column_list[:2], 1):
        print(f"{i}: {item}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorIndexBuilder:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_type='IVF'):
        self.model = SentenceTransformer(model_name)
        self.index_type = index_type
        self.dimension = self.model.get_sentence_embedding_dimension()

    def create_index(self, dimension):
        """Создание FAISS индекса в зависимости от типа"""
        if self.index_type == 'Flat':
            # Точный поиск
            return faiss.IndexFlatIP(dimension)
        elif self.index_type == 'IVF':
            # Приближенный поиск с кластеризацией
            nlist = 100  # количество кластеров
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            return index
        elif self.index_type == 'HNSW':
            # Графовый метод
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            return index
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def build_index(self, texts, output_dir="data/processed/indexes", index_name="index"):
        """Построение полного индекса"""
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype(np.float32)

        # Нормализация для косинусного сходства
        faiss.normalize_L2(embeddings)

        logger.info("Creating FAISS index...")
        index = self.create_index(self.dimension)

        if self.index_type == 'IVF':
            # Требуется обучение для IVF
            logger.info("Training IVF index...")
            index.train(embeddings)

        logger.info("Adding embeddings to index...")
        index.add(embeddings)

        # Сохранение метаданных
        metadata = {
            'texts': texts,
            'embedding_shape': embeddings.shape,
            'index_type': self.index_type,
            'model_name': 'all-MiniLM-L6-v2',
            'dimension': self.dimension
        }

        # Сохранение всех компонентов
        index_path = os.path.join(output_dir, f"{index_name}.faiss")
        meta_path = os.path.join(output_dir, f"{index_name}_meta.json")
        embeddings_path = os.path.join(output_dir, f"{index_name}_embeddings.npy")

        faiss.write_index(index, index_path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        np.save(embeddings_path, embeddings)

        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {meta_path}")
        logger.info(f"Embeddings saved to {embeddings_path}")

        return index, metadata, embeddings


if __name__ == "__main__":
    # Загрузка данных
    texts = read_column_to_list("./data/raw/websites_updated.xlsx", column=1)

    # Построение индекса
    builder = VectorIndexBuilder(index_type='IVF')
    index, metadata, embeddings = builder.build_index(texts)

    print("Index built successfully!")
    print(f"Index size: {index.ntotal} vectors")
    print(f"Embedding dimension: {metadata['dimension']}")