from ann_search import HNSWRetrieval
import numpy as np
from params import small_config, medium_config, large_config


if __name__ == "__main__":
    # Создание демо-данных
    dimension = 768
    num_vectors = 10000
    config = large_config
    
    # Случайные векторы (в реальности - эмбеддинги)
    data_vectors = np.random.random((num_vectors, dimension)).astype('float32')
    query_vector = np.random.random((1, dimension)).astype('float32')
    
    # Инициализация HNSW
    hnsw = HNSWRetrieval(dimension=dimension, 
                         M=config['M'], 
                         ef_construction=config['ef_construction'], 
                         ef_search=config['ef_search'])
    
    # Добавление данных
    hnsw.add_vectors(data_vectors)
    
    # Поиск
    distances, indices = hnsw.search(query_vector, k=5)
    print(f"Найденные индексы: {indices}")
    print(f"Расстояния: {distances}")