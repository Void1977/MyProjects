"""Загрузка сообщений и примеров интентов."""

from __future__ import annotations

import pandas as pd


def load_messages(path, min_length: int = 5) -> pd.DataFrame:
    """Читает сообщения и отбрасывает пустые и слишком короткие («да», «ок»)."""
    messages = pd.read_csv(path)
    messages["text_clean"] = messages["text_clean"].fillna("").astype(str).str.strip()
    messages = messages[messages["text_clean"].str.len() >= min_length]
    return messages.reset_index(drop=True)


def load_intents(path) -> list[str]:
    """Читает примеры фраз из интентов, по одной в строке."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
