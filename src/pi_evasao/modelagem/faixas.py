"""Faixas de risco (Alto / Médio / Baixo) do dashboard (Etapa 4; ver docs/decisoes/0007).

Papel no pipeline: converte o escore do modelo em três faixas que o orientador entende SEM expor a
probabilidade bruta (os pesos de classe a inflam; ADR 0004). As faixas são definidas por DESEMPENHO
OBSERVADO nos escores *out-of-fold* do modelo final:

- **Baixo:** escore abaixo do limiar F2 (o modelo não alertaria a escola);
- **Médio:** acima do limiar F2, mas abaixo do corte vermelho;
- **Alto:** acima do menor corte em que a precisão observada atinge ``precisao_alvo`` (padrão 60%).

Para cada faixa guarda-se a taxa de escolas que de fato ficaram no grupo de alto risco no ano seguinte,
que o app mostra ao usuário ("das escolas desta faixa, X% ficaram em alto risco").
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

FAIXAS = ("Alto", "Médio", "Baixo")


def cortes_faixas(y, proba, limiar: float, precisao_alvo: float, min_escolas: int = 100) -> dict:
    """Cortes das faixas a partir dos escores OOF.

    ``corte_vermelho`` é o MENOR corte (>= ``limiar``) com precisão observada >= ``precisao_alvo`` e ao
    menos ``min_escolas`` escolas acima dele (evita cortes ruidosos no topo). Se a precisão alvo nunca
    é atingida, usa o percentil 90 dos escores e sinaliza ``precisao_alcancada = False``.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    # precisão observada em cada corte possível
    prec, _, thr = precision_recall_curve(y, proba)
    prec = prec[:-1]
    ordenados = np.sort(proba)
    # quantas escolas têm escore em ou acima de cada corte
    n_acima = len(proba) - np.searchsorted(ordenados, thr, side="left")
    # cortes válidos: acima do limiar, com a precisão alvo e escolas suficientes
    ok = (thr >= limiar) & (prec >= precisao_alvo) & (n_acima >= min_escolas)
    # o vermelho começa no menor corte válido; sem nenhum, usa o percentil 90
    if ok.any():
        corte, alcancada = float(thr[ok].min()), True
    else:
        corte, alcancada = float(max(np.quantile(proba, 0.90), limiar)), False
    return {"limiar_amarelo": float(limiar), "corte_vermelho": corte,
            "precisao_alvo": float(precisao_alvo), "precisao_alcancada": alcancada}


def atribuir_faixa(escores, cortes: dict) -> np.ndarray:
    """Faixa ('Alto', 'Médio' ou 'Baixo') de cada escore."""
    e = np.asarray(escores)
    return np.where(e >= cortes["corte_vermelho"], "Alto", np.where(e >= cortes["limiar_amarelo"], "Médio", "Baixo"))


def validacao_por_faixa(y, proba, cortes: dict) -> dict:
    """Por faixa: nº de pares e taxa observada de alto risco no ano seguinte (validação, sem o teste)."""
    y, faixa = np.asarray(y), atribuir_faixa(proba, cortes)
    return {f: {"n": int((faixa == f).sum()),
                "taxa_alto_risco": float(y[faixa == f].mean()) if (faixa == f).any() else None}
            for f in FAIXAS}
