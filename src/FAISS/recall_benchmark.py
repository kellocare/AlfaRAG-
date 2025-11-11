import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class RecallBenchmark:
    def __init__(self, index_dir="indexes", index_name="my_index"):
        self.index_dir = index_dir
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self):
        """Загрузка данных и индекса"""
        index_path = f"{self.index_dir}/{self.index_name}.faiss"
        meta_path = f"{self.index_dir}/{self.index_name}_meta.json"
        embeddings_path = f"{self.index_dir}/{self.index_name}_embeddings.npy"

        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
        self.embeddings = np.load(embeddings_path)
        self.texts = self.metadata['texts']

    def exact_search(self, query_embedding, top_k=5):
        """Точный поиск (ground truth)"""
        similarities = cos_sim(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities.numpy())[::-1][:top_k]
        return top_indices, similarities[top_indices].numpy()

    def ann_search(self, query_embedding, top_k=5):
        """Приближенный поиск через FAISS"""
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        return indices[0], scores[0]

    def calculate_recall(self, ground_truth, predicted):
        """Вычисление recall@k"""
        ground_truth_set = set(ground_truth)
        predicted_set = set(predicted)
        return len(ground_truth_set & predicted_set) / len(ground_truth_set)

    def run_benchmark(self, test_queries=None, top_k=5):
        """Запуск бенчмарка"""
        if test_queries is None:
            test_queries = [
                "нейронные сети и глубокое обучение",
                "базы данных и SQL",
                "веб разработка и программирование",
                "безопасность и шифрование",
                "облачные технологии"
            ]

        self.load_data()

        recalls = []
        hit_rates = []

        print(f"Running benchmark with {len(test_queries)} queries...")

        for query in tqdm(test_queries):
            # Создание эмбеддинга запроса
            query_embedding = self.model.encode([query])

            # Точный поиск (ground truth)
            gt_indices, gt_scores = self.exact_search(query_embedding, top_k)

            # Приближенный поиск
            ann_indices, ann_scores = self.ann_search(query_embedding, top_k)

            # Расчет recall
            recall = self.calculate_recall(gt_indices, ann_indices)
            recalls.append(recall)

            # Hit rate (хотя бы один правильный результат)
            hit_rate = 1.0 if recall > 0 else 0.0
            hit_rates.append(hit_rate)

            print(f"\nQuery: '{query}'")
            print(f"Recall@{top_k}: {recall:.3f}")
            print(f"Ground truth: {gt_indices}")
            print(f"ANN results: {ann_indices}")

        # Статистика
        mean_recall = np.mean(recalls)
        mean_hit_rate = np.mean(hit_rates)

        print(f"\n{'=' * 50}")
        print(f"BENCHMARK RESULTS (Top-{top_k})")
        print(f"{'=' * 50}")
        print(f"Mean Recall@{top_k}: {mean_recall:.3f}")
        print(f"Hit Rate@{top_k}: {mean_hit_rate:.3f}")
        print(f"Total queries: {len(test_queries)}")

        # Визуализация
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.hist(recalls, bins=10, alpha=0.7, color='skyblue')
        plt.axvline(mean_recall, color='red', linestyle='--', label=f'Mean: {mean_recall:.3f}')
        plt.xlabel('Recall@k')
        plt.ylabel('Frequency')
        plt.title('Recall Distribution')
        plt.legend()

        plt.subplot(1, 2, 2)
        recall_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        recall_counts = [
            len([r for r in recalls if 0.0 <= r < 0.2]),
            len([r for r in recalls if 0.2 <= r < 0.4]),
            len([r for r in recalls if 0.4 <= r < 0.6]),
            len([r for r in recalls if 0.6 <= r < 0.8]),
            len([r for r in recalls if 0.8 <= r <= 1.0])
        ]
        plt.bar(recall_ranges, recall_counts, color='lightgreen')
        plt.xlabel('Recall Range')
        plt.ylabel('Count')
        plt.title('Recall Range Distribution')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('benchmarks/recall_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'mean_recall': mean_recall,
            'mean_hit_rate': mean_hit_rate,
            'recalls': recalls,
            'hit_rates': hit_rates,
            'test_queries': test_queries
        }


if __name__ == "__main__":
    import os

    os.makedirs("benchmarks", exist_ok=True)

    benchmark = RecallBenchmark()
    results = benchmark.run_benchmark(top_k=5)

    # Сохранение результатов
    with open('benchmarks/recall_results.json', 'w') as f:
        json.dump(results, f, indent=2)