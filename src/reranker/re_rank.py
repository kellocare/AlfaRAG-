from sentence_transformers import CrossEncoder

# Загружаем модель для ранжирования
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Пример запроса и документов
query = "Как приготовить пасту карбонара"
docs = [
    "Паста карбонара готовится с беконом и яйцами",
    "Карбонара — это итальянское блюдо из пасты, яиц и сыра пекорино",
    "Пицца с колбасой и сыром — популярное итальянское блюдо"
]

def rank(query, docs):

  # Пары (запрос, документ)
  pairs = [(query, doc) for doc in docs]

  # Предсказание релевантности
  scores = reranker.predict(pairs)

  # Сортировка
  ranked = sorted(zip(scores, docs), reverse=True)

  # Печать топ-5
  print("Топ-5 документов после re-ranking:\n")
  y = []
  for i, (score, doc) in enumerate(ranked[:5], 1):
      #print(f"{i}. ({score:.4f}) {doc}")
      y.append(doc)
  return y
