"""
Генератор ответов для RAG-пайплайна
"""

from typing import List, Dict, Any, Optional
import os
import logging
import argparse
import json
from dataclasses import dataclass, asdict

from prompt_templates import build_prompt, extract_citation_indices

# попытаемся импортировать transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

logger = logging.getLogger("rag.generator")
logging.basicConfig(level=logging.INFO)


@dataclass
class GenerationResult:
    answer: str
    prompt: str
    used_chunks: List[Dict[str, Any]]
    cited_indices: List[int]
    cited_sources: List[Dict[str, Any]]


class Generator:
    """
    Класс-обёртка для генератора
    Параметры:
      - название модели из хагивагифейс (model_name)
      - Начинка генератора (backend) transformers или mock
      - Устройство вычислений (device) ЦП или CUDA для трансформеров прайм
    """

    def __init__(self,
                 model_name: str = "Mistral-7B-Instruct-v0.2",
                 backend: str = "mock",
                 device: str = "cpu",
                 torch_dtype: Optional[str] = None):
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipeline = None

        if backend == "transformers":
            if not _HAS_TRANSFORMERS:
                raise RuntimeError("Библиотека с трансформ-моделями не установлены, установи пакет transformers.")
            self._load_transformers(model_name, device, torch_dtype)
        elif backend == "mock":
            logger.info("Генератор работает в mock-режиме, модели не загрузятся.")
        else:
            raise ValueError("Неизвестный бэк: %s" % backend)

    def _load_transformers(self, model_name: str, device: str, torch_dtype: Optional[str]):
        """
        Загрузка модели и токенайзера, далее создание пайплайна для генерации текста
        Попытка загрузки делается с device_map='auto' если такое поддерживается
        """
        logger.info("Загрузка модели '%s' на устройстве='%s' (dtype=%s)", model_name, device, torch_dtype)
        # схема такая: если модель маленькая — загружаем обычным способом;
        # если большая — пользователь должен сам подготовить инфру для запуска.
        # здесь я использую pipelineис automodelforcausalLM.
        # Пользователь может настроить переменные среды хаги для локального хранения моделей.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # для некоторых моделей из семейства лламы нужно добавить eos
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            # на маленьких моделях можно выставить устройство автоматически;
            # но по дефолту будет ЦП
            if device == "cpu":
                model.to("cpu")
            else:
                try:
                    model.to("cuda")
                except Exception as e:
                    logger.warning("Не удалось переместить модель на CUDA: %s, поэтому оставляем на ЦП.", e)
            self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device != "cpu" else -1)
            logger.info("Модель и пайплайн готовы к запуску.")
        except Exception as e:
            logger.exception("Ошибка при загрузке модели трансформеров: %s", e)
            raise

    def generate(self, query: str, chunks: List[Dict[str, Any]], *,
                 max_context_words: int = 1800,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 top_p: float = 0.9,
                 do_sample: bool = False,
                 stop_sequences: Optional[List[str]] = None) -> GenerationResult:
        """
        Генерирует ответ для заданного запроса и списка чанков, упорядоченных по релевантности
        Функция вернет GenerationResult, в комментах опишу пайплайн
        """
        # 1) cобираем промпт из связки инструкции, query и контекста
        prompt_obj = build_prompt(query, chunks, max_context_words=max_context_words)
        prompt = prompt_obj["prompt"]
        used_chunks = prompt_obj["used_chunks"]

        # 2) генерация
        if self.backend == "mock":
            # по-черновому конкатенируем первые предложения из первых 2 чанков
            answer_parts = []
            for ch in used_chunks[:3]:
                text = ch.get("text", "")
                # берём первые 40 слов
                piece = " ".join(text.split()[:40])
                answer_parts.append(piece)
            answer = " ".join(answer_parts)
            # добавим блок "ИСТОЧНИКИ" в нужном формате
            src_lines = []
            for ch in used_chunks:
                idx = ch.get("index_in_prompt")
                url = ch.get("url") or ""
                src_lines.append(f"[{idx}] {url}")
            answer += "\n\nИсточники:\n" + "\n".join(src_lines)
        else:
            # пайплайн трансформер-модели
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            if stop_sequences:
                # данная ситуация означает, что у пайплайна нет прямого параметра остановки последовательности
                # поэтому можно уйти в постпроцесс декодирования или использовать
                # генерацию конфига с еос-токеном
                pass
            try:
                res = self.pipeline(prompt, **gen_kwargs)
                if isinstance(res, list) and len(res) > 0:
                    answer = res[0].get("generated_text", "")
                    # уберем промпт из ответа
                    # отрезаем длину промпта, если модель вернула совмещённый текст
                    if answer.startswith(prompt):
                        answer = answer[len(prompt):].strip()
                else:
                    answer = ""
            except Exception as e:
                logger.exception("Ошибка при генерации: %s", e)
                raise

        # 3) извлекаем упомянутые индексы источников через regex
        cited_indices = extract_citation_indices(answer)
        # по списку cited_indices формируем cited_sources из used_chunks по index_in_prompt
        cited_sources = []
        # used_chunks имеют поле index_in_prompt, и в них ищем соответствия
        index_map = {ch["index_in_prompt"]: ch for ch in used_chunks if "index_in_prompt" in ch}
        for idx in cited_indices:
            if idx in index_map:
                ch = index_map[idx]
                cited_sources.append({"index": idx, "url": ch.get("url"), "chunk_id": ch.get("chunk_id"), "title": ch.get("title")})
            else:
                # если не нашли — игнорим
                # это кстати может означать, что модель сослалась на несуществующий индекс
                logger.debug("Цитируемый индекс %s среди used_chunks не нашелся :(", idx)

        result = GenerationResult(
            answer=answer,
            prompt=prompt,
            used_chunks=used_chunks,
            cited_indices=cited_indices,
            cited_sources=cited_sources
        )
        return result


# бонусом накинем интерфейсик в терминал для быстрых локальных тестов
def _load_jsonl_or_json(path: str):
    """
    Поддерживает JSONL и JSON
    Формат объекта или строчки: {"chunk_id":..., "text":..., "url":..., ...}
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        try:
            # если первый символ — '[' вероятно это JSON array
            if first.startswith("["):
                data = json.load(f)
            else:
                # читаем как JSONL
                data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать файл {path} как json/jsonl: {e}")
    return data


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--backend", type=str, default="mock", choices=["mock", "transformers"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--chunks", type=str, required=True)
    p.add_argument("--max_context_words", type=int, default=1800)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    chunks = _load_jsonl_or_json(args.chunks)
    gen = Generator(model_name=args.model, backend=args.backend, device=args.device)
    res = gen.generate(args.query, chunks, max_context_words=args.max_context_words, max_new_tokens=args.max_new_tokens)
    print("=== промпт ===")
    print(res.prompt)
    print("\n=== ответ ===")
    print(res.answer)
    print("\n=== чанков использовано ===")
    for ch in res.used_chunks:
        print(f"[{ch.get('index_in_prompt')}] {ch.get('chunk_id')} url={ch.get('url')}")
    print("\n=== использованные indices ===")
    print(res.cited_indices)
    if res.cited_sources:
        print("\n=== затронутые источники ===")
        print(json.dumps(res.cited_sources, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()