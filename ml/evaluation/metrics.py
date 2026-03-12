"""
ml/evaluation/metrics.py
------------------------
Metricas de avaliacao para sistemas de recomendacao.

Todas as funcoes recebem listas de strings e operam no mesmo padrao:
  - recommended : lista ordenada por score decrescente (top-K do modelo)
  - relevant    : conjunto de itens considerados relevantes para o usuario
                  (tipicamente ratings >= 4 no conjunto de teste)
"""

from __future__ import annotations

import math


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Fracao dos top-K recomendados que sao relevantes.

    Parameters
    ----------
    recommended:
        Lista de product_ids ordenada por score decrescente.
    relevant:
        Lista de product_ids relevantes para o usuario.
    k:
        Tamanho do corte.

    Returns
    -------
    float
        Valor em [0.0, 1.0]. Retorna 0.0 se k <= 0 ou recommended vazio.
    """
    if k <= 0 or not recommended:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / k


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Fracao dos itens relevantes que aparecem nos top-K recomendados.

    Parameters
    ----------
    recommended:
        Lista de product_ids ordenada por score decrescente.
    relevant:
        Lista de product_ids relevantes para o usuario.
    k:
        Tamanho do corte.

    Returns
    -------
    float
        Valor em [0.0, 1.0]. Retorna 0.0 se relevant vazio ou k <= 0.
    """
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain no corte K.

    Assume relevancia binaria: 1 se o item esta em relevant, 0 caso contrario.

    Parameters
    ----------
    recommended:
        Lista de product_ids ordenada por score decrescente.
    relevant:
        Lista de product_ids relevantes para o usuario.
    k:
        Tamanho do corte.

    Returns
    -------
    float
        Valor em [0.0, 1.0]. Retorna 0.0 se relevant vazio ou k <= 0.
    """
    if k <= 0 or not relevant:
        return 0.0

    relevant_set = set(relevant)

    def dcg(items: list[str], cutoff: int) -> float:
        return sum(
            1.0 / math.log2(rank + 2)
            for rank, item in enumerate(items[:cutoff])
            if item in relevant_set
        )

    actual = dcg(recommended, k)
    # IDCG: ordenacao ideal — todos os relevantes nos primeiros postos
    ideal_count = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))

    return actual / ideal if ideal > 0 else 0.0


def mrr(recommended: list[str], relevant: list[str]) -> float:
    """Mean Reciprocal Rank: reciproco do rank do primeiro item relevante.

    Parameters
    ----------
    recommended:
        Lista de product_ids ordenada por score decrescente.
    relevant:
        Lista de product_ids relevantes para o usuario.

    Returns
    -------
    float
        Valor em (0.0, 1.0]. Retorna 0.0 se nenhum item relevante
        aparecer na lista.
    """
    if not relevant or not recommended:
        return 0.0
    relevant_set = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0
