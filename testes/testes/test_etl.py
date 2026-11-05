"""Testes do ETL (Etapa 1): limpeza sem imputação, deduplicação e ida e volta no SQLite (ADR 0005)."""

import pandas as pd

from pi_evasao.etl import banco
from pi_evasao.etl.limpeza import limpar
from pi_evasao.ingestao.base import COLUNAS, INDICADORES


def _linha(**kw):
    """Uma linha válida do painel, com campos substituíveis."""
    base = dict(id_escola=35000001, ano=2022, id_municipio=3550308, rede="estadual", nome_escola="ESCOLA A",
                nome_municipio="São Paulo", taxa_abandono_ef=2.0, tdi_ef=12.0, atu_ef=28.0, fonte="sintetico")
    base.update(kw)
    return base


def test_invalidos_viram_nan_e_duplicata_e_removida(cfg):
    """Valores impossíveis viram ausentes e a duplicata sai."""
    df = pd.DataFrame([
        _linha(),
        _linha(),                                   # duplicata exata de (id_escola, ano)
        _linha(ano=2023, taxa_abandono_ef=-1.0),    # sentinela negativa
        _linha(ano=2024, tdi_ef=112.0),             # taxa acima de 100
    ])
    limpo, rel = limpar(df, cfg)
    assert len(limpo) == 3
    assert limpo["taxa_abandono_ef"].isna().sum() == 1 and limpo["tdi_ef"].isna().sum() == 1
    assert next(e for e in rel["etapas"] if e["etapa"] == "Deduplicação")["descartadas"] == 1


def test_escola_sem_nenhum_abandono_e_removida_por_inteiro(cfg):
    """Escola sem nenhuma taxa de abandono não tem alvo e é removida."""
    df = pd.DataFrame([_linha(), _linha(id_escola=35000002, taxa_abandono_ef=float("nan")),
                       _linha(id_escola=35000002, ano=2023, taxa_abandono_ef=float("nan"))])
    limpo, _ = limpar(df, cfg)
    assert set(limpo["id_escola"]) == {35000001}


def test_fora_do_recorte_e_descartado(cfg):
    """Rede ou ano fora do recorte do estudo são descartados."""
    df = pd.DataFrame([_linha(), _linha(id_escola=35000003, rede="federal"), _linha(id_escola=35000004, ano=2019)])
    limpo, _ = limpar(df, cfg)
    assert list(limpo["id_escola"]) == [35000001]


def test_contabilidade_do_relatorio_fecha(painel_bruto, cfg):
    """Linhas de entrada menos as descartadas dá as de saída."""
    limpo, rel = limpar(painel_bruto, cfg)
    assert rel["linhas_entrada"] - sum(e["descartadas"] for e in rel["etapas"]) == rel["linhas_saida"] == len(limpo)
    assert not limpo.duplicated(["id_escola", "ano"]).any()
    assert not (limpo[list(INDICADORES)] < 0).any().any()
    assert not (limpo[["taxa_abandono_ef", "tdi_ef"]] > 100).any().any()


def test_ida_e_volta_no_sqlite(painel_limpo, tmp_path):
    """O painel gravado e relido do SQLite é igual ao original."""
    with banco.conectar(tmp_path / "t.db") as con:
        banco.gravar_painel(con, painel_limpo)
        lido = banco.ler_painel(con)
    assert len(lido) == len(painel_limpo) and set(lido.columns) == set(COLUNAS) | {"fonte"}
    pd.testing.assert_series_equal(lido["taxa_abandono_ef"], painel_limpo["taxa_abandono_ef"], check_names=False)
