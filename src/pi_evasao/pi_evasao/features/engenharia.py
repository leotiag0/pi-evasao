"""Engenharia de atributos e construção do dataset de modelagem (Etapa 2 do PI).

Decisões metodológicas implementadas aqui (cada uma tem ADR em docs/decisoes/):

- ADR 0002 - Desenho temporal: as FEATURES vêm do ano ``t`` e o RÓTULO do ano ``t+1``.
  Nos indicadores de rendimento do INEP, aprovação + reprovação + abandono = 100%; prever o
  abandono do MESMO ano a partir das outras duas seria aritmética, não aprendizado. Prever o ano
  seguinte é também o que "alerta precoce" significa.
- ADR 0003 - Rótulo por percentil: ``alto_risco`` = abandono de ``t+1`` >= percentil P do PRÓPRIO
  ano ``t+1`` (padrão P80), o que mantém a prevalência estável apesar da distorção de 2020-21.
- ADR 0014 - Conjunto mínimo: seis atributos, os que sustentam o resultado (abandono, distorção e alunos por
  turma no ano t, as variações de abandono e de distorção e a rede).

Nenhuma coluna do ano ``t+1`` entra em ``X`` (garantido por testes/test_features.py).
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..formato import i as _i, n as _n

DELTAS_BASE = ("taxa_abandono_ef", "tdi_ef")

FEATURES_NUM = ["taxa_abandono_ef", "tdi_ef", "atu_ef", "delta_taxa_abandono_ef", "delta_tdi_ef"]
FEATURES_CAT = ["rede"]
FEATURES = FEATURES_NUM + FEATURES_CAT

ALVO = "alto_risco"
# Metadados do par (t -> t+1). NUNCA entram em X: descrevem o ano-alvo.
META = ["id_escola", "ano_ref", "ano_alvo", "taxa_abandono_alvo", "limiar_alvo", "fonte"]

DESCRICAO = {
    "taxa_abandono_ef": "Taxa de abandono no ano t (%). Legítima: conhecida no momento da predição. Domina a importância (persistência escolar).",
    "tdi_ef": "Taxa de distorção idade-série no ano t (%).",
    "atu_ef": "Média de alunos por turma no ano t.",
    "delta_taxa_abandono_ef": "DERIVADA: taxa_abandono(t) - taxa_abandono(t-1). NaN se a escola não tem o ano t-1 (ex.: 2016, primeiro ano do painel).",
    "delta_tdi_ef": "DERIVADA: variação da distorção idade-série entre t-1 e t. Mesma regra de NaN.",
    "rede": "Rede de ensino (estadual | municipal).",
}

EXCLUIDAS = {
    "taxa_aprovacao_ef": "Colinearidade exata: aprovação = 100 - reprovação - abandono. Nem é lida da fonte.",
    "id_municipio": "Alta cardinalidade (centenas de códigos): convida o modelo a memorizar municípios em vez de generalizar.",
    "ano": "Convida a memorizar tendência temporal e quebra a validação out-of-time.",
    "id_escola": "Identificador: memorização garantida, generalização nenhuma. Usado só para agrupar a validação por escola.",
    "taxa_abandono_alvo": "É o próprio alvo do ano t+1 (vazamento direto).",
    "limiar_alvo": "Percentil calculado sobre a distribuição do ano t+1 (vazamento indireto).",
}


def _preparar_painel(painel: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ao painel as variações de abandono e de distorção entre t-1 e t.

    Cada linha só usa informação do próprio ano e do ano imediatamente anterior (t-1), e só se ele existir.
    """
    p = painel.sort_values(["id_escola", "ano"]).reset_index(drop=True).copy()
    g = p.groupby("id_escola")
    consecutivo = g["ano"].diff() == 1
    for c in DELTAS_BASE:
        p[f"delta_{c}"] = g[c].diff().where(consecutivo)
    return p


def features_para_predicao(painel: pd.DataFrame, ano_ref: int) -> pd.DataFrame:
    """Atributos do ano ``ano_ref`` para prever ``ano_ref + 1``, SEM exigir rótulo (uso real do alerta).

    É o que o dashboard usa: o último ano do painel não tem rótulo (o ano seguinte ainda não aconteceu),
    então ``construir_dataset`` o descarta. Por construção só entram dados até ``ano_ref``: o painel é
    cortado ANTES do preparo, o que impede qualquer informação futura de vazar (teste em test_predicao).
    """
    p = _preparar_painel(painel[painel["ano"] <= ano_ref])
    f = p[p["ano"] == ano_ref][["id_escola", "ano", "fonte", *FEATURES]].rename(columns={"ano": "ano_ref"})
    # Sem nenhum indicador no ano-base a escola não tem ensino fundamental nele: não há o que prever.
    return f.dropna(subset=["taxa_abandono_ef", "tdi_ef", "atu_ef"], how="all").reset_index(drop=True)


def construir_dataset(painel: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Painel escola-ano -> um registro por par (escola, t -> t+1)."""
    p = _preparar_painel(painel)

    feat = p[["id_escola", "ano", "fonte", *FEATURES]].rename(columns={"ano": "ano_ref"})
    alvo = p[["id_escola", "ano", "taxa_abandono_ef"]].rename(
        columns={"ano": "ano_alvo", "taxa_abandono_ef": "taxa_abandono_alvo"})
    alvo["ano_ref"] = alvo["ano_alvo"] - 1

    ds = feat.merge(alvo, on=["id_escola", "ano_ref"], how="inner").dropna(subset=["taxa_abandono_alvo"])

    # ADR 0003: percentil da distribuição do PRÓPRIO ano-alvo, calculado sobre todo o painel.
    limiar_por_ano = p.groupby("ano")["taxa_abandono_ef"].quantile(cfg.percentil_risco)
    ds["limiar_alvo"] = ds["ano_alvo"].map(limiar_por_ano)
    ds[ALVO] = (ds["taxa_abandono_alvo"] >= ds["limiar_alvo"]).astype(int)

    ds = ds[[*META, *FEATURES, ALVO]].sort_values(["ano_ref", "id_escola"]).reset_index(drop=True)
    prevalencia = ds.groupby("ano_alvo")[ALVO].mean()
    resumo = {
        "pares": len(ds), "escolas": int(ds["id_escola"].nunique()),
        "prevalencia_por_ano_alvo": prevalencia.round(4).to_dict(),
        "limiar_por_ano_alvo": limiar_por_ano.round(3).to_dict(),
        "prevalencia_total": round(float(ds[ALVO].mean()), 4),
    }
    return ds, resumo


def separar_xy(ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(X, y, meta): X contém SOMENTE colunas do ano t."""
    return ds[FEATURES].copy(), ds[ALVO].copy(), ds[META].copy()


def markdown_features(ds: pd.DataFrame, resumo: dict, cfg: Config, fonte: str) -> str:
    """Documento ``relatorios/metricas/features.md`` (seção Engenharia de atributos)."""
    # dados sintéticos nunca podem ser confundidos com achados reais: o aviso vai no topo
    aviso = ("> **AVISO:** dados SINTÉTICOS; estatísticas e prevalências descrevem o gerador, não escolas reais.\n\n"
             if fonte == "sintetico" else "")
    # cabeçalho e prevalência por ano-alvo
    L = ["# Engenharia de atributos", "", aviso,
         f"- **Fonte:** {fonte}",
         f"- **Pares (escola, t -> t+1):** {_i(resumo['pares'])} de {_i(resumo['escolas'])} escolas",
         f"- **Rótulo:** `alto_risco` = abandono em t+1 >= percentil {_n(cfg.percentil_risco * 100, 0)}% do próprio ano t+1",
         f"- **Prevalência total de alto risco:** {_n(resumo['prevalencia_total'] * 100, 1)}%", "",
         "## Prevalência e limiar por ano-alvo", "",
         "| Ano-alvo (t+1) | Limiar de abandono (%) | Prevalência de alto risco |", "|---:|---:|---:|"]
    L += [f"| {a} | {_n(resumo['limiar_por_ano_alvo'][a], 2)} | {_n(resumo['prevalencia_por_ano_alvo'][a] * 100, 1)}% |"
          for a in sorted(resumo["prevalencia_por_ano_alvo"])]
    L += ["", "*A prevalência pode superar o percentil nominal por empates: as taxas têm uma casa decimal "
          "e muitas escolas repetem o mesmo valor no limiar.*", "",
          "## Variáveis preditoras (todas do ano t)", "",
          "| Variável | Tipo | % nulos | Média | Desvio | Mín | Máx | Descrição |", "|---|---|---:|---:|---:|---:|---:|---|"]
    # uma linha de estatísticas por atributo numérico
    for c in FEATURES_NUM:
        s = ds[c]
        L.append(f"| `{c}` | num | {_n(s.isna().mean() * 100, 1)} | {_n(s.mean(), 2)} | {_n(s.std(), 2)} | "
                 f"{_n(s.min(), 2)} | {_n(s.max(), 2)} | {DESCRICAO[c]} |")
    # atributos categóricos: só os níveis
    for c in FEATURES_CAT:
        L.append(f"| `{c}` | cat | {_n(ds[c].isna().mean() * 100, 1)} | - | - | - | - | {DESCRICAO[c]} "
                 f"Níveis: {', '.join(sorted(ds[c].dropna().unique()))}. |")
    L += ["", "## Variáveis deliberadamente excluídas de X", "", "| Variável | Motivo |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in EXCLUIDAS.items()]
    L += ["", "NaN estruturais (ex.: deltas de 2020) são tratados no Pipeline do modelo com indicador de "
          "ausência, e não removidos.", ""]
    return "\n".join(L)
