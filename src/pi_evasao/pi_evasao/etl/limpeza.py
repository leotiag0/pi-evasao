"""Limpeza do painel escola-ano (Etapa 1 do PI).

Papel no pipeline: transforma o painel bruto da fonte em um painel consistente e devolve, junto,
um RELATÓRIO de cada intervenção (quantas linhas/valores, e por quê). Esse relatório vira
``relatorios/metricas/qualidade.md`` e a seção "Tratamento de dados" da Metodologia.

Decisões (docs/decisoes/0005 e 0014):
- SEM imputação aqui. Imputar antes do split contaminaria o teste com estatísticas do conjunto
  todo; a imputação vive dentro do Pipeline do scikit-learn, ajustada só no treino.
- Valores negativos são tratados como sentinela de "sem informação" (-1) -> NaN.
  Só valores IMPOSSÍVEIS são descartados: 99% de aprovação, por exemplo, é legítimo.
- Taxas fora de [0, 100] -> NaN.
- Escolas que NUNCA têm taxa de abandono (sem alvo possível, como as sem ensino fundamental) são
  removidas por inteiro; linhas isoladas sem abandono ficam (podem servir de atributo de anos vizinhos).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..formato import i as _i, n as _n
from ..ingestao.base import COLUNAS, INDICADORES, TAXAS


def pct_nulos(df: pd.DataFrame) -> pd.Series:
    """Percentual de ausentes em cada indicador."""
    return (df[list(INDICADORES)].isna().mean() * 100).round(2)


def limpar(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Devolve ``(painel_limpo, relatorio)``. ``relatorio`` é serializável em markdown/JSON."""
    etapas: list[dict] = []
    valores: list[dict] = []
    relatorio = {"linhas_entrada": len(df), "nulos_antes": pct_nulos(df).to_dict()}

    def etapa(nome: str, antes: int, depois: int, motivo: str):
        """Registra no relatório uma etapa que remove linhas (quantas e por quê)."""
        etapas.append({"etapa": nome, "linhas_antes": antes, "linhas_depois": depois,
                       "descartadas": antes - depois, "motivo": motivo})

    df = df.copy()

    # 1. Tipagem e chaves ---------------------------------------------------------------------
    for c in INDICADORES:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("id_escola", "ano", "id_municipio"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    antes = len(df)
    df = df.dropna(subset=["id_escola", "ano", "id_municipio"])
    for c in ("id_escola", "ano", "id_municipio"):
        df[c] = df[c].astype("int64")
    etapa("Tipagem e chaves", antes, len(df), "Linhas sem id_escola, ano ou id_municipio numéricos")

    # 2. Escopo do estudo ---------------------------------------------------------------------
    antes = len(df)
    df = df[df["rede"].isin(cfg.redes) & df["ano"].isin(cfg.anos)]
    etapa("Filtro de escopo", antes, len(df),
          f"Fora do recorte: redes={list(cfg.redes)}, anos={min(cfg.anos)}-{max(cfg.anos)}")

    # 3. Duplicatas ---------------------------------------------------------------------------
    antes = len(df)
    df = df.drop_duplicates(subset=["id_escola", "ano"], keep="first")
    etapa("Deduplicação", antes, len(df), "Mais de uma linha para o mesmo (id_escola, ano); mantida a primeira")

    # 4. Valores impossíveis -> NaN (não descarta linhas) --------------------------------------
    for col in INDICADORES:
        neg = (df[col] < 0).sum()
        if neg:
            valores.append({"coluna": col, "valores": int(neg), "acao": "-> NaN",
                            "motivo": "Valor negativo (sentinela de 'sem informação')"})
        df.loc[df[col] < 0, col] = np.nan
    for col in TAXAS:
        alto = (df[col] > 100).sum()
        if alto:
            valores.append({"coluna": col, "valores": int(alto), "acao": "-> NaN",
                            "motivo": "Taxa percentual acima de 100"})
        df.loc[df[col] > 100, col] = np.nan

    # 5. Escolas sem alvo possível --------------------------------------------------------------
    antes = len(df)
    tem_alvo = df.groupby("id_escola")["taxa_abandono_ef"].transform(lambda s: s.notna().any())
    df = df[tem_alvo]
    etapa("Escolas sem alvo", antes, len(df),
          "Escola sem NENHUM valor de taxa_abandono_ef em todo o período (removida por inteiro)")

    df = df[[*COLUNAS, "fonte"]].sort_values(["id_escola", "ano"]).reset_index(drop=True)
    relatorio.update(etapas=etapas, valores=valores, linhas_saida=len(df),
                     escolas=int(df["id_escola"].nunique()), nulos_depois=pct_nulos(df).to_dict())
    return df, relatorio


def relatorio_markdown(rel: dict, fonte: str) -> str:
    """Renderiza o relatório de qualidade (vira relatorios/metricas/qualidade.md)."""
    # dados sintéticos nunca podem ser confundidos com achados reais: o aviso vai no topo
    aviso = ("> **AVISO:** dados SINTÉTICOS. Os problemas abaixo foram injetados de propósito para "
             "exercitar o ETL; não descrevem a qualidade dos dados reais do INEP.\n\n"
             if fonte == "sintetico" else "")
    # monta o Markdown por seções: resumo, etapas, valores invalidados e nulos
    L = ["# Relatório de qualidade de dados", "", aviso,
         f"- **Fonte:** {fonte}",
         f"- **Linhas de entrada:** {_i(rel['linhas_entrada'])}",
         f"- **Linhas após limpeza:** {_i(rel['linhas_saida'])} ({_i(rel['escolas'])} escolas)", "",
         "## Etapas que removem linhas", "",
         "| Etapa | Antes | Depois | Descartadas | Motivo |", "|---|---:|---:|---:|---|"]
    L += [f"| {e['etapa']} | {_i(e['linhas_antes'])} | {_i(e['linhas_depois'])} | {_i(e['descartadas'])} | {e['motivo']} |"
          for e in rel["etapas"]]
    L += ["", "## Valores invalidados (viram NaN; a linha é mantida)", ""]
    if rel["valores"]:
        L += ["| Coluna | Valores | Ação | Motivo |", "|---|---:|---|---|"]
        L += [f"| `{v['coluna']}` | {_i(v['valores'])} | {v['acao']} | {v['motivo']} |" for v in rel["valores"]]
    else:
        L += ["Nenhum valor impossível encontrado."]
    L += ["", "## Nulos por coluna (% das linhas)", "", "| Coluna | Antes | Depois |", "|---|---:|---:|"]
    L += [f"| `{k}` | {_n(rel['nulos_antes'][k], 2)} | {_n(rel['nulos_depois'][k], 2)} |" for k in INDICADORES]
    L += ["", "*Nulos remanescentes não são imputados aqui: a imputação ocorre dentro do Pipeline de "
          "modelagem, ajustada só no treino (evita vazamento).*", ""]
    return "\n".join(L)
