import faiss
import params
import numpy as np

class HNSWRetrieval:
    def __init__(self, dimension=768, M=16, ef_construction=200, ef_search=128):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Создание HNSW индекса
        self.index = faiss.IndexHNSWFlat(dimension, M)
        
        # Настройка параметров построения
        self.index.hnsw.efConstruction = ef_construction
        
    def add_vectors(self, vectors):
        """Добавление векторов в индекс"""
        if not self.index.is_trained:
            # HNSW не требует обучения, но нужно добавить первый вектор
            self.index.add(vectors)
        else:
            self.index.add(vectors)
    
    def search(self, query_vector, k=5):
        """Поиск k ближайших соседей"""
        # Установка параметра поиска
        self.index.hnsw.efSearch = self.ef_search
        
        # Выполнение поиска
        distances, indices = self.index.search(query_vector, k)
        return distances, indices
    
    def save_index(self, filepath):
        """Сохранение индекса"""
        faiss.write_index(self.index, filepath)
    
    def load_index(self, filepath):
        """Загрузка индекса"""
        self.index = faiss.read_index(filepath)
