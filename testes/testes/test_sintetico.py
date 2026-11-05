"""Testes do gerador sintético (andaime dos testes): determinismo, esquema canônico e mecanismo embutido."""

import pandas as pd

from pi_evasao.ingestao.base import COLUNAS
from pi_evasao.ingestao.sintetico import FonteSintetica


def _gerar(cfg, n=300, seed=42, sujeira=True):
    """Gera um painel sintético com os parâmetros dos testes."""
    return FonteSintetica(n, seed, cfg.persistencia_sintetico, sujeira).carregar(cfg.anos, cfg.uf, cfg.redes)


def test_deterministico_mesma_seed(cfg):
    """A mesma semente gera exatamente o mesmo painel."""
    pd.testing.assert_frame_equal(_gerar(cfg, seed=7), _gerar(cfg, seed=7))


def test_seeds_diferentes_geram_dados_diferentes(cfg):
    """Sementes diferentes geram painéis diferentes."""
    assert not _gerar(cfg, seed=1)["taxa_abandono_ef"].equals(_gerar(cfg, seed=2)["taxa_abandono_ef"])


def test_schema_canonico_e_marca_de_fonte(cfg):
    """As colunas são as do esquema e toda linha marca a fonte sintética."""
    df = _gerar(cfg)
    assert list(df.columns) == [*COLUNAS, "fonte"]
    assert (df["fonte"] == "sintetico").all()
    assert set(df["ano"]) == set(cfg.anos)
    assert set(df["rede"]) <= set(cfg.redes)


def test_sujeira_e_injetada_quando_pedida(cfg):
    """A sujeira (duplicatas, negativos, ausentes) só aparece quando pedida."""
    sujo, limpo = _gerar(cfg, n=2000), _gerar(cfg, n=2000, sujeira=False)
    assert sujo.duplicated(["id_escola", "ano"]).any() and not limpo.duplicated(["id_escola", "ano"]).any()
    assert (sujo["taxa_abandono_ef"] < 0).any() and sujo.isna().any().any()
    assert not (limpo["taxa_abandono_ef"] < 0).any()


def test_efeito_pandemia_reduz_abandono_registrado(cfg):
    """O abandono de 2020 é menor que o de 2022, como no gerador."""
    df = _gerar(cfg, n=1500, sujeira=False)
    med = df.groupby("ano")["taxa_abandono_ef"].median()
    assert med[2020] < med[2022]


def test_ha_persistencia_entre_anos(cfg):
    """Sem persistência escola-a-escola o modelo t -> t+1 não teria sinal para aprender."""
    df = _gerar(cfg, n=1500, sujeira=False).pivot(index="id_escola", columns="ano", values="taxa_abandono_ef")
    assert df[2022].corr(df[2023]) > 0.3
