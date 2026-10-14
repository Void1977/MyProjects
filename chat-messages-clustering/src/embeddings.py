"""Эмбеддинги сообщений моделью E5."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-large"


def get_device() -> str:
    """Возвращает доступное устройство: CUDA, Apple Silicon (MPS) или CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_texts(texts: list[str], model_name: str = MODEL_NAME, batch_size: int = 32) -> np.ndarray:
    """Кодирует тексты в L2-нормированные векторы (для e5-large — размерности 1024).

    По рекомендации авторов E5 к каждому тексту добавляется префикс "query: ".
    Вектор текста — среднее векторов его токенов без учёта паддинга.
    Модель загружается на время вызова и затем выгружается из памяти.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    batches = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Эмбеддинги"):
        batch = ["query: " + text for text in texts[start:start + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            token_vectors = model(**encoded).last_hidden_state
        batches.append(_mean_pool(token_vectors, encoded["attention_mask"]).cpu().numpy())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return np.concatenate(batches)


def _mean_pool(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Усредняет векторы токенов по маске внимания и нормирует результат по L2."""
    mask = attention_mask.unsqueeze(-1).float()
    mean = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(mean, p=2, dim=1)
