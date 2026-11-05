"""Diretoria de Ensino de cada escola (Etapa 4; ver docs/decisoes/0010).

Papel no pipeline: a Diretoria de Ensino é a unidade em que a gestão distribui supervisão e apoio entre
escolas, então é o filtro que torna o alerta por escola acionável. O INEP não publica essa informação;
ela vem de duas fontes de dados abertos, cruzadas pelo código INEP da escola:

1. **SEDUC-SP** (``ENDERECO_ESCOLAS``): todas as escolas ESTADUAIS, com a Diretoria de Ensino (DE) e o
   código MEC/INEP (``CODESCMEC``). Casa 99,4% das escolas estaduais do projeto.
2. **Prefeitura de São Paulo** (cadastro de escolas da SME): escolas MUNICIPAIS da capital, com a
   Diretoria Regional de Educação (DRE) e o código INEP (``CODINEP``). Casa 100% delas.
3. As demais escolas (municipais do interior) herdam a DE do seu município, quando o município
   pertence a uma única DE (641 dos 644 municípios). Em Campinas e Guarulhos (duas DEs cada) as
   municipais ficam sem diretoria identificada; Santana de Parnaíba não tem escola estadual no
   arquivo da SEDUC, então também fica sem.

Resultado real (``prever`` de 2026-09-28): 97,8% das 10.449 escolas com diretoria (99,4% das estaduais,
96,5% das municipais), em 104 diretorias (91 DEs e 13 DREs).

DE (rede estadual) e DRE (rede municipal da capital) são estruturas diferentes; o rótulo diz qual é.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..log import obter_logger
from .inep import baixar

log = obter_logger("diretorias")

# Versões fixadas (as mais recentes em 2026-09-28); ao atualizar, conferir se o layout mudou.
URL_SEDUC = "https://dados.educacao.sp.gov.br/sites/default/files/ENDERECO_ESCOLAS_062025.csv"
URL_SME = ("https://dados.prefeitura.sp.gov.br/dataset/8da55b0e-b385-4b54-9296-d0000014ddd5/resource/"
           "313d0bea-d2b5-468b-88b6-99412edacf20/download/escolas122025.csv")


def ler_seduc(caminho) -> pd.DataFrame:
    """``id_escola, id_municipio, diretoria`` das escolas estaduais (arquivo da SEDUC-SP)."""
    d = pd.read_csv(caminho, sep=";", encoding="latin1", usecols=["DE", "CD_IBGE", "CODESCMEC"])
    d = d.dropna().rename(columns={"CODESCMEC": "id_escola", "CD_IBGE": "id_municipio"})
    d["diretoria"] = "DE " + d.pop("DE").str.strip().str.title()
    return d.astype({"id_escola": "int64", "id_municipio": "int64"})


def ler_sme(caminho) -> pd.DataFrame:
    """``id_escola, diretoria`` das escolas do cadastro da SME (capital)."""
    d = pd.read_csv(caminho, sep=";", encoding="latin1", usecols=["DIRETORIA", "CODINEP"], dtype=str).dropna()
    d = d[d["CODINEP"].str.strip().str.isdigit()]
    return pd.DataFrame({"id_escola": d["CODINEP"].astype("int64"),
                         "diretoria": "DRE " + d["DIRETORIA"].str.strip().str.title()}).drop_duplicates("id_escola")


def atribuir_diretoria(escolas: pd.DataFrame, seduc: pd.DataFrame, sme: pd.DataFrame) -> pd.Series:
    """Diretoria de cada linha de ``escolas`` (``id_escola``, ``id_municipio``); NaN se não identificada.

    Prioridade: código da escola na SEDUC, depois na SME, depois a DE única do município.
    """
    por_escola = pd.concat([seduc[["id_escola", "diretoria"]], sme]).drop_duplicates("id_escola").set_index("id_escola")
    des = seduc.groupby("id_municipio")["diretoria"].unique()
    por_municipio = des[des.str.len() == 1].str[0]
    ids = escolas["id_escola"].astype("int64")
    resultado = ids.map(por_escola["diretoria"])
    return resultado.fillna(escolas["id_municipio"].astype("int64").map(por_municipio))


def carregar_fontes(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baixa (uma vez, com cache em ``dados/bruto/diretorias``) e lê as duas fontes."""
    cache = cfg.caminho("bruto") / "diretorias"
    cache.mkdir(parents=True, exist_ok=True)
    arquivos = {}
    for nome, url in (("seduc", URL_SEDUC), ("sme", URL_SME)):
        arquivos[nome] = cache / url.rsplit("/", 1)[-1]
        if not arquivos[nome].exists():
            log.info("Baixando %s (diretorias de ensino)", arquivos[nome].name)
            baixar(url, arquivos[nome])
    return ler_seduc(arquivos["seduc"]), ler_sme(arquivos["sme"])
