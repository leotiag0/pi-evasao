"""Fixtures compartilhadas. Os testes usam dados sintéticos pequenos e diretórios temporários."""

from __future__ import annotations

import dataclasses

import pytest

from pi_evasao.cli import cmd_pipeline
from pi_evasao.config import carregar_config
from pi_evasao.etl.limpeza import limpar
from pi_evasao.ingestao.sintetico import FonteSintetica


@pytest.fixture(scope="session")
def cfg():
    """Config dos testes: mesma da produção, mas com janela própria (2020-2024).

    Os testes usam dados sintéticos e anos fixos; assim não quebram quando o recorte real do
    ``config.yaml`` muda (ADR 0014).
    """
    return dataclasses.replace(carregar_config(), anos=(2020, 2021, 2022, 2023, 2024), ano_ref_teste_temporal=2023)


@pytest.fixture()
def cfg_tmp(cfg, tmp_path):
    """Config pequena e isolada: escreve tudo em ``tmp_path`` (nada toca o repositório)."""
    return dataclasses.replace(cfg, raiz=tmp_path, n_escolas_sintetico=800, cv_folds=3)


@pytest.fixture(scope="session")
def painel_bruto(cfg):
    """Painel sintético pequeno, gerado uma vez por sessão de testes."""
    return FonteSintetica(600, cfg.seed, cfg.persistencia_sintetico).carregar(cfg.anos, cfg.uf, cfg.redes)


@pytest.fixture(scope="session")
def painel_limpo(painel_bruto, cfg):
    """O mesmo painel depois da limpeza do ETL."""
    return limpar(painel_bruto, cfg)[0]


@pytest.fixture(scope="session")
def executado(cfg, tmp_path_factory):
    """Pipeline completo (fonte sintética, dados pequenos) executado UMA vez em diretório temporário."""
    tmp = tmp_path_factory.mktemp("pipeline")
    c = dataclasses.replace(cfg, raiz=tmp, n_escolas_sintetico=800, cv_folds=3)
    return c, cmd_pipeline(c, fonte="sintetico")
