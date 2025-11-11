"""
Шаблоны промптов и утилиты для сборки контекста для генератора.
"""

from typing import List, Dict, Any, Optional
import textwrap
import re

DEFAULT_MAX_CONTEXT_WORDS= 1800  # приближённый лимит слов контекста


DEFAULT_INSTRUCTION = textwrap.dedent(
    """
    Ты - агент, который отвечает только на основе предоставленных источников из csv-файлов.
    Инструкция:
    1) Используй исключительно информацию, находящуюся в секции "ИСТОЧНИКИ" ниже. Ничего не выдумывай.
    2) Если вопрос не может быть полностью отвечен из источников — честно напиши: "Информации недостаточно" и укажи, каких данных не хватает.
    3) Формат ответа:
       1. Короткий прямой ответ (1–3 абзаца).
       2. Раздел "Источники:" где перечислены использованные источники в виде [N] <url> (N — номер фрагмента).
    4) Для каждого факта, взятого из источников, поставь ссылку в квадратных скобках, например: "Согласно сервису X [1]...".
    5) Отвечай на русском языке.
    """
).strip()


# Промпт, в который будет вставлен вопрос и контекст.
# {query} — вопрос, {context_block} — сгенерированные чанки (нумерованные).
PROMPT_TEMPLATE = textwrap.dedent(
    """
    {instruction}

    Вопрос:
    {query}

    ИСТОЧНИКИ:
    {context_block}

    ОТВЕТ:
    """
).strip()

def _sanitize_text(s: str) -> str:
    """Минимальная нормализация текста, убирая двойные пробелы и управляющие символы"""
    if s is None:
        return ""
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s


def make_context_block(chunks: List[Dict[str, Any]], max_words: int = DEFAULT_MAX_CONTEXT_WORDS) -> Dict[str, Any]:
    """
    Формирует блок контекста для вставки в промпт
    Возвращает dict:
      {
        "block": str,        # готовая строка для вставки в промпт
        "used_chunks": List[chunk],  # реально использованные чанки (последовательность)
      }
    Ограничение по объёму — через число слов (приближённо)
    """
    if not chunks:
        return {"block": "", "used_chunks": []}

    used = []
    total_words = 0
    pieces = []

    for i, ch in enumerate(chunks, start=1):
        text = _sanitize_text(ch.get("text", ""))
        if not text:
            continue
        words = text.split()
        # если один большой чанк превышает лимит, берём первые слова
        if total_words + len(words) > max_words:
            remaining = max_words - total_words
            if remaining <= 0:
                break
            text = " ".join(words[:remaining])
            # помечаем, что данный chunk был усечён
            truncated = True
        else:
            truncated = False

        # формируем запись вида [1] <text> (source: <url>)
        url = ch.get("url") or ch.get("source") or ""
        title = ch.get("title") or ""
        header = f"[{i}]"
        meta = []
        if title:
            meta.append(title)
        if url:
            meta.append(url)
        meta_str = " — ".join(meta) if meta else ""
        piece = f"{header} {text}\n(source: {meta_str})"
        if truncated:
            piece += " [УСЕЧЕНО]" # для списка далее будет именоваться как truncated
        pieces.append(piece)

        used.append({**ch, "index_in_prompt": i, "truncated": truncated})
        total_words += len(text.split())
        if total_words >= max_words:
            break

    block = "\n\n".join(pieces)
    return {"block": block, "used_chunks": used}


def build_prompt(query: str, chunks: List[Dict[str, Any]], *,
                 instruction: Optional[str] = None,
                 max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS) -> Dict[str, Any]:
    """
    Собираем финальный промпт: instruction + query + контекст
    Функция возвращает dict следующего типа
      {
        "prompt": str,          # строка, готовая к передаче в модель
        "used_chunks": [...],   # список чанков, которые вошли в prompt (с полем index_in_prompt)
      }
    """
    instruction = instruction or DEFAULT_INSTRUCTION
    ctx = make_context_block(chunks, max_words=max_context_words)
    context_block = ctx["block"]
    used_chunks = ctx["used_chunks"]
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, query=query, context_block=context_block)
    return {"prompt": prompt, "used_chunks": used_chunks}

CITATION_REGEX = re.compile(r'\[(\d+)\]')  # ищем [1], [2] и т.д.


def extract_citation_indices(answer_text: str) -> List[int]:
    """
    Возвращаем список уникальных индексов источников, упомянутых в ответе, в порядке появления
    """
    found = CITATION_REGEX.findall(answer_text)
    # преобразуем в int, фильтруем и сохраняем порядок
    seen = []
    for f in found:
        try:
            n = int(f)
        except Exception:
            continue
        if n not in seen:
            seen.append(n)
    return seen