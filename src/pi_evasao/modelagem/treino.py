"""Treino do modelo de classificação de risco (Etapa 3 do PI).

Decisões (ver docs/decisoes/0003, 0004, 0009 e 0014):

- Um único modelo: ``HistGradientBoostingClassifier`` do scikit-learn (Gradient Boosting; trata NaN
  nativamente, sem dependência de XGBoost). Nos mesmos dados, Random Forest perdeu 0,02 a 0,03 de PR-AUC e a
  regressão logística 0,03 a 0,05, ambos fora do ruído (ADR 0014), então não há o que escolher.
- Dois pisos de comparação: ``baseline_dummy`` (prevê a prevalência, equivale a alertar todas as escolas) e
  ``baseline_persistencia`` (ADR 0009), que ordena as escolas só pela taxa de abandono do ano t e responde
  à pergunta "o modelo é melhor do que ordenar pelo abandono do ano passado?".
- A imputação de NaN vive DENTRO do Pipeline: é ajustada só nos dados de treino de cada fold, o que evita
  vazamento de estatísticas do teste.
- Validação ``fora_do_tempo``: treina em pares antigos e testa no par mais recente (por padrão 2024->2025).
  Replica o uso real (prever o futuro). Não há holdout aleatório: a mesma escola em anos consecutivos não
  são observações independentes, e o teste no tempo é o número honesto.
- Variáveis sem variação no conjunto de TREINO (vazias ou constantes) são excluídas automaticamente antes de
  ajustar, e o fato é registrado: o HistGradientBoosting falha ao discretizar uma coluna sem valores distintos.
- O limiar de decisão NÃO é 0,5: maximiza F-beta (beta=2, recall pesa 4x a precisão) sobre probabilidades
  out-of-fold do treino, com validação cruzada agrupada por escola. O teste é tocado uma única vez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..config import Config
from ..features.engenharia import FEATURES_CAT, FEATURES_NUM

REGIME_TEMPORAL = "fora_do_tempo"
MODELO_FINAL = "gradient_boosting"
MODELOS = (MODELO_FINAL, "baseline_dummy", "baseline_persistencia")


def eh_baseline(nome: str) -> bool:
    """Baselines são pisos de comparação: aparecem nas tabelas, mas nunca são o modelo final."""
    return nome.startswith("baseline_")


class Persistencia(ClassifierMixin, BaseEstimator):
    """Escore = taxa de abandono do ano t (em %, levada a 0-1). Não aprende nada além da mediana,
    usada para escolas sem a taxa. O escore só serve para ORDENAR (PR-AUC, limiar por F2); não é uma
    probabilidade calibrada, então o Brier deste baseline não tem leitura (ADR 0009)."""

    def fit(self, X, y):
        """Guarda só a mediana do abandono do treino, para escolas sem a taxa."""
        self.classes_ = np.array([0, 1])
        self.mediana_ = float(X["taxa_abandono_ef"].median())
        return self

    def predict_proba(self, X):
        """Escore = taxa de abandono do ano t (em 0 a 1), com a mediana no lugar do ausente."""
        p = X["taxa_abandono_ef"].fillna(self.mediana_).clip(0, 100).to_numpy(dtype=float) / 100
        return np.column_stack([1 - p, p])

    def predict(self, X):  # exigido pela API do scikit-learn; o pipeline usa só predict_proba
        """Classe prevista com corte em 0,5 (exigida pela API; o pipeline usa ``predict_proba``)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def colunas_com_variacao(X: pd.DataFrame) -> list[str]:
    """Colunas de X com ao menos 2 valores distintos (as demais não carregam informação)."""
    return [c for c in X.columns if X[c].nunique(dropna=True) >= 2]


def construir_pipelines(cfg: Config, colunas) -> dict[str, Pipeline]:
    """Um Pipeline por modelo, usando só as ``colunas`` informadas (em geral, as com variação)."""
    # só as colunas com variação no treino entram no modelo
    num = [c for c in FEATURES_NUM if c in colunas]
    cat = [c for c in FEATURES_CAT if c in colunas]
    pre = ColumnTransformer([  # NaN tratado nativamente pelo HGB
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
    ])
# o modelo e os dois pisos de comparação
    return {
        MODELO_FINAL: Pipeline([
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                                   class_weight="balanced", random_state=cfg.seed))]),
        "baseline_dummy": Pipeline([("clf", DummyClassifier(strategy="prior"))]),
        "baseline_persistencia": Pipeline([("clf", Persistencia())]),
    }


def escolher_limiar(y, proba, beta: float) -> float:
    """Limiar que maximiza F-beta na curva precisão-recall (ADR 0004)."""
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta**2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((b2 * prec + rec) > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    return float(thr[int(np.argmax(f))])


def _cv(cfg: Config) -> StratifiedGroupKFold:
    """Validação cruzada estratificada e agrupada por escola."""
    return StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)


def proba_out_of_fold(pipe: Pipeline, X, y, grupos, cfg: Config) -> np.ndarray:
    """Probabilidades da classe 1 obtidas por CV agrupada por escola (cada linha prevista por um
    modelo que NÃO viu a sua escola)."""
    return cross_val_predict(pipe, X, y, groups=grupos, cv=_cv(cfg), method="predict_proba")[:, 1]


def indices_temporal(meta: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Treina em pares com ano_ref < ano de corte; testa no par com ano_ref == corte."""
    corte = cfg.ano_ref_teste_temporal
    tr = np.flatnonzero(meta["ano_ref"].to_numpy() < corte)
    te = np.flatnonzero(meta["ano_ref"].to_numpy() == corte)
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(f"Corte temporal {corte} não deixa treino e teste não vazios.")
    return tr, te


def treinar_regime(X, y, grupos, idx_tr, idx_te, cfg: Config) -> dict[str, dict]:
    """Treina todos os modelos em um regime e devolve, por modelo, o que ``avaliar`` precisa."""
    resultados = {}
    Xtr, ytr, gtr = X.iloc[idx_tr], y.iloc[idx_tr], grupos.iloc[idx_tr]
    usadas = colunas_com_variacao(Xtr)  # decidido só com o TREINO, nunca com o teste
    Xtr, Xte, yte = Xtr[usadas], X.iloc[idx_te][usadas], y.iloc[idx_te]
    # cada modelo: limiar por validação cruzada no treino, depois ajuste final e uso no teste
    for nome, pipe in construir_pipelines(cfg, usadas).items():
        oof = proba_out_of_fold(pipe, Xtr, ytr, gtr, cfg)
        limiar = escolher_limiar(ytr, oof, cfg.fbeta)
        pipe.fit(Xtr, ytr)  # teste só é tocado aqui, depois de tudo decidido
        resultados[nome] = {
            "pipeline": pipe, "limiar": limiar,
            "cv_pr_auc": float(average_precision_score(ytr, oof)),
            "y_teste": yte.to_numpy(), "proba_teste": pipe.predict_proba(Xte)[:, 1],
            "X_teste": Xte, "n_treino": len(idx_tr), "n_teste": len(idx_te),
            "colunas_usadas": usadas,
        }
    return resultados


def treinar_final(X, y, grupos, nome_modelo: str, cfg: Config) -> tuple[Pipeline, float, list[str], np.ndarray]:
    """Reajusta o modelo escolhido com TODOS os pares; limiar por OOF sobre todos os dados.

    Devolve também as colunas efetivamente usadas (o dashboard precisa passar exatamente essas) e os
    escores *out-of-fold*, com os quais se definem as faixas de risco (``faixas.py``).
    """
    usadas = colunas_com_variacao(X)
    X = X[usadas]
    pipe = construir_pipelines(cfg, usadas)[nome_modelo]
    oof = proba_out_of_fold(pipe, X, y, grupos, cfg)
    limiar = escolher_limiar(y, oof, cfg.fbeta)
    pipe.fit(X, y)
    return pipe, limiar, usadas, oof
