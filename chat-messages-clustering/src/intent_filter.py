"""Сравнение сообщений с примерами существующих интентов."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_morph = pymorphy3.MorphAnalyzer()


def lemmatize(text: str) -> str:
    """Приводит текст к нижнему регистру, оставляет только буквы и цифры, ставит слова в начальную форму."""
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text.lower())
    return " ".join(_morph.parse(word)[0].normal_form for word in text.split())


def tfidf_similarity(texts: list[str], intents: list[str]) -> np.ndarray:
    """Косинусная близость TF-IDF-векторов по леммам (униграммы и биграммы).

    Строка матрицы — сообщение, столбец — пример интента.
    """
    texts = [lemmatize(text) for text in texts]
    intents = [lemmatize(text) for text in intents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50_000)
    vectorizer.fit(texts + intents)
    return cosine_similarity(vectorizer.transform(texts), vectorizer.transform(intents))


def score_against_intents(
    messages: pd.DataFrame,
    message_embeddings: np.ndarray,
    intents: list[str],
    intent_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Добавляет к сообщениям близость к самому похожему примеру интентов.

    sim_tfidf — близость по словам, sim_emb — по смыслу (эмбеддинги), sim_combined — среднее двух.
    closest_intent — самый похожий по смыслу пример.
    """
    by_words = tfidf_similarity(messages["text_clean"].tolist(), intents)
    by_meaning = cosine_similarity(message_embeddings, intent_embeddings)

    scored = messages.copy()
    scored["sim_tfidf"] = by_words.max(axis=1)
    scored["sim_emb"] = by_meaning.max(axis=1)
    scored["sim_combined"] = (scored["sim_tfidf"] + scored["sim_emb"]) / 2
    scored["closest_intent"] = [intents[i] for i in by_meaning.argmax(axis=1)]
    return scored
