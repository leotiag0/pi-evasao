"""Banco relacional SQLite (Etapa 1 do PI). Ver docs/decisoes/0005-sqlite-como-banco.md.

Papel no pipeline: o banco é a fonte de verdade entre as fases. ``etl`` grava o painel limpo aqui;
``features`` lê daqui e grava ``dataset_modelagem``; ``treinar`` lê ``dataset_modelagem``.
Só ``sqlite3`` da biblioteca padrão: zero dependência extra e um arquivo único que qualquer
integrante abre (DB Browser, VS Code) sem servidor.

Tabelas: ``painel`` (PK id_escola+ano), ``dataset_modelagem`` (recriada a cada execução de
``features``) e ``execucao_etl`` (auditoria).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..ingestao.base import COLUNAS

DDL = """
DROP TABLE IF EXISTS painel;
CREATE TABLE painel (
    id_escola INTEGER NOT NULL, ano INTEGER NOT NULL, id_municipio INTEGER NOT NULL, rede TEXT NOT NULL,
    nome_escola TEXT, nome_municipio TEXT,
    taxa_abandono_ef REAL, tdi_ef REAL, atu_ef REAL,
    fonte TEXT NOT NULL,
    PRIMARY KEY (id_escola, ano)
);
CREATE TABLE IF NOT EXISTS execucao_etl (
    id INTEGER PRIMARY KEY AUTOINCREMENT, executado_em TEXT NOT NULL, fonte TEXT NOT NULL,
    linhas_entrada INTEGER, linhas_saida INTEGER, escolas INTEGER, hash_painel TEXT
);
"""


@contextmanager
def conectar(caminho: Path):
    """Abre o SQLite, confirma a transação ao final e sempre fecha a conexão."""
    con = sqlite3.connect(caminho)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def gravar_painel(con: sqlite3.Connection, painel: pd.DataFrame) -> None:
    """Recria a tabela e grava o painel limpo (idempotente)."""
    con.executescript(DDL)
    painel[[*COLUNAS, "fonte"]].to_sql("painel", con, if_exists="append", index=False)


def registrar_execucao(con: sqlite3.Connection, fonte: str, entrada: int, saida: int,
                       escolas: int, hash_painel: str) -> None:
    """Anota na tabela de auditoria uma execução do ETL (fonte, contagens e hash do painel)."""
    con.execute(
        "INSERT INTO execucao_etl (executado_em, fonte, linhas_entrada, linhas_saida, escolas, hash_painel) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), fonte, entrada, saida, escolas, hash_painel))


def ler_painel(con: sqlite3.Connection) -> pd.DataFrame:
    """Painel escola-ano completo, ordenado por escola e ano."""
    return pd.read_sql_query("SELECT * FROM painel ORDER BY id_escola, ano", con)


def gravar_tabela(con: sqlite3.Connection, nome: str, df: pd.DataFrame) -> None:
    """Grava (ou substitui) uma tabela a partir de um DataFrame."""
    df.to_sql(nome, con, if_exists="replace", index=False)


def ler_tabela(con: sqlite3.Connection, nome: str) -> pd.DataFrame:
    """Lê uma tabela inteira do banco."""
    return pd.read_sql_query(f"SELECT * FROM {nome}", con)
