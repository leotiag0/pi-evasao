"""Previsão do risco de cada escola para o ano seguinte ao último do painel (Etapa 4 do PI).

Papel no pipeline: é a ponte entre o modelo e o dashboard. Aplica o modelo final aos atributos do
ÚLTIMO ano do painel (que não tem rótulo: o ano seguinte ainda não aconteceu) e exporta, em ``app/dados/``,
tudo o que a interface precisa. O app NÃO importa scikit-learn nem este pacote: lê só estes arquivos
(ver docs/decisoes/0007), o que o mantém leve para hospedar e independente do framework.

Arquivos gerados:
- ``predicoes.csv.gz``: uma linha por escola (faixa, posição no ranking, indicadores atuais, nomes e a
  ``diretoria`` de ensino do ADR 0010);
- ``historico.csv.gz``: série anual dos indicadores das mesmas escolas (o Painel Geral usa isto);
- ``modelo_meta.json``: metadados do modelo, desempenho validado e cortes das faixas (página "Sobre").

O escore bruto vai no CSV apenas para auditoria: o app mostra faixa e posição, nunca a probabilidade
(os pesos de classe a inflam; ADR 0004).
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from ..config import Config
from ..erros import ErroAcionavel
from ..etl import banco
from ..features.engenharia import features_para_predicao
from ..ingestao.diretorias import atribuir_diretoria
from ..log import hash_dataframe
from . import persistencia
from .faixas import FAIXAS, atribuir_faixa

ARQ_PREDICOES = "predicoes.csv.gz"
ARQ_HISTORICO = "historico.csv.gz"
ARQ_META = "modelo_meta.json"


def _nomes(df: pd.DataFrame) -> pd.DataFrame:
    """Sem nome da escola ou do município (andaime sintético), usa o código como rótulo."""
    df = df.copy()
    df["nome_escola"] = df["nome_escola"].fillna("Escola " + df["id_escola"].astype(str))
    df["nome_municipio"] = df["nome_municipio"].fillna("Município " + df["id_municipio"].astype(str))
    return df


def gerar_predicoes(cfg: Config, diretorias: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict:
    """Gera e grava os arquivos do dashboard. Devolve um resumo para o registro de execução.

    ``diretorias`` são as fontes (SEDUC, SME) de ``ingestao.diretorias.carregar_fontes``; sem elas a
    coluna ``diretoria`` fica vazia e o app esconde o filtro.
    """
    modelo, meta = persistencia.carregar_modelo(cfg)
    if not meta.get("faixas"):
        raise ErroAcionavel("O modelo salvo não tem as faixas de risco. Retreine com 'pi-evasao treinar'.")
    with banco.conectar(cfg.caminho("banco")) as con:
        painel = banco.ler_painel(con)
    if painel.empty:
        raise ErroAcionavel("Banco vazio. Rode 'pi-evasao etl' primeiro.")

    ano_ref = int(painel["ano"].max())
    f = features_para_predicao(painel, ano_ref)
    # O Pipeline espera exatamente as colunas usadas no treino (em ordem), não todas as FEATURES.
    escore = modelo.predict_proba(f[meta["colunas_numericas"] + meta["colunas_categoricas"]])[:, 1]

    ultimo = painel.loc[painel["ano"] == ano_ref, ["id_escola", "id_municipio", "nome_escola", "nome_municipio"]]
    df = f[["id_escola", "ano_ref", "rede", "atu_ef", "taxa_abandono_ef", "tdi_ef", "delta_taxa_abandono_ef"]].merge(
        ultimo, on="id_escola", how="left")
    df["ano_alvo"] = ano_ref + 1
    df["escore"] = escore
    df["faixa"] = atribuir_faixa(escore, meta["faixas"])
    df["posicao"] = df["escore"].rank(ascending=False, method="first").astype(int)
    df["percentil"] = (df["escore"].rank(pct=True) * 100).round(1)  # 100 = maior risco
    df["dados_incompletos"] = df["taxa_abandono_ef"].isna()         # sem abandono do ano-base: previsão menos confiável
    df = _nomes(df).sort_values("posicao").reset_index(drop=True)
    df["diretoria"] = atribuir_diretoria(df, *diretorias) if diretorias else pd.NA

    historico = painel.loc[painel["id_escola"].isin(df["id_escola"]),
                           ["id_escola", "ano", "taxa_abandono_ef", "tdi_ef", "atu_ef"]]

    destino = cfg.caminho("app_dados")
    df.to_csv(destino / ARQ_PREDICOES, index=False, encoding="utf-8")
    historico.to_csv(destino / ARQ_HISTORICO, index=False, encoding="utf-8")
    meta_app = {**meta, "gerado_em": datetime.now().isoformat(timespec="seconds"), "ano_ref": ano_ref,
                "ano_alvo": ano_ref + 1, "teste_ano_ref": cfg.ano_ref_teste_temporal, "n_escolas": len(df), "hash_predicoes": hash_dataframe(df[["id_escola", "escore"]])}
    (destino / ARQ_META).write_text(json.dumps(meta_app, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    contagem = df["faixa"].value_counts()
    return {"fonte": meta.get("fonte_dados"), "ano_base": ano_ref, "ano_previsto": ano_ref + 1, "escolas": len(df),
            **{f"faixa_{fx.lower()}": int(contagem.get(fx, 0)) for fx in FAIXAS},
            "dados_incompletos": int(df["dados_incompletos"].sum()),
            "com_diretoria": int(df["diretoria"].notna().sum()),
            "hash_predicoes": meta_app["hash_predicoes"]}
