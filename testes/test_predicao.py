"""Testes da previsão para o dashboard (Etapa 4): atributos sem rótulo, faixas de risco e arquivos gerados."""

import json

import numpy as np
import pandas as pd

from pi_evasao.features.engenharia import FEATURES, construir_dataset, features_para_predicao
from pi_evasao.modelagem.faixas import FAIXAS, atribuir_faixa, cortes_faixas, validacao_por_faixa


# ---- atributos de um ano sem rótulo ---------------------------------------------------------------
def test_features_para_predicao_ignora_dados_posteriores(painel_limpo, cfg):
    """Alterar tudo depois de ``ano_ref`` não pode mudar os atributos (proteção contra vazamento)."""
    base = features_para_predicao(painel_limpo, 2023)
    alterado = painel_limpo.copy()
    posterior = alterado["ano"] > 2023
    for col in ("taxa_abandono_ef", "tdi_ef", "atu_ef"):
        alterado.loc[posterior, col] = 999.0
    pd.testing.assert_frame_equal(base, features_para_predicao(alterado, 2023))


def test_features_para_predicao_coincide_com_o_dataset_de_treino(painel_limpo, cfg):
    """O que o app usa para prever é o mesmo que o modelo viu no treino (mesmo cálculo)."""
    ds, _ = construir_dataset(painel_limpo, cfg)
    treino = ds[ds["ano_ref"] == 2022].set_index("id_escola")[FEATURES]
    pred = features_para_predicao(painel_limpo, 2022).set_index("id_escola")[FEATURES]
    comum = treino.index.intersection(pred.index)
    assert len(comum) > 0.9 * len(treino)
    pd.testing.assert_frame_equal(treino.loc[comum].sort_index(), pred.loc[comum].sort_index())


def test_ultimo_ano_tem_atributos_mesmo_sem_rotulo(painel_limpo, cfg):
    """O último ano do painel gera atributos mesmo sem rótulo (uso real do alerta)."""
    ultimo = max(cfg.anos)
    ds, _ = construir_dataset(painel_limpo, cfg)
    assert ultimo not in set(ds["ano_ref"])            # o dataset de treino não o tem (sem rótulo)
    f = features_para_predicao(painel_limpo, ultimo)
    assert len(f) > 0 and set(FEATURES) <= set(f.columns)


# ---- faixas de risco -------------------------------------------------------------------------------
def _escores(n=30000, seed=0):
    """Escores e rótulos simulados para testar as faixas."""
    rng = np.random.default_rng(seed)
    e = rng.beta(2, 5, n)
    return (rng.random(n) < np.clip(e * 1.6, 0, 1)).astype(int), e


def test_cortes_das_faixas_e_taxas_monotonicas():
    """As taxas observadas crescem de Baixo a Alto e o corte vermelho cumpre a precisão alvo."""
    y, e = _escores()
    cortes = cortes_faixas(y, e, limiar=0.10, precisao_alvo=0.60)
    assert cortes["precisao_alcancada"] and cortes["corte_vermelho"] >= cortes["limiar_amarelo"]
    val = validacao_por_faixa(y, e, cortes)
    assert val["Baixo"]["taxa_alto_risco"] < val["Médio"]["taxa_alto_risco"] < val["Alto"]["taxa_alto_risco"]
    assert val["Alto"]["taxa_alto_risco"] >= 0.60      # a definição do corte vermelho
    assert sum(v["n"] for v in val.values()) == len(y)


def test_precisao_alvo_inatingivel_cai_no_percentil_90():
    """Sem atingir a precisão alvo, o corte vermelho usa o percentil 90."""
    y, e = _escores()
    cortes = cortes_faixas(y, e, limiar=0.10, precisao_alvo=1.01)
    assert cortes["precisao_alcancada"] is False and cortes["corte_vermelho"] >= 0.10


def test_faixa_medio_vazia_quando_a_precisao_ja_e_alta_no_limiar():
    """Se a precisão no limiar F2 já atinge o alvo, o vermelho começa no limiar e não há faixa Médio."""
    y, e = _escores()
    cortes = cortes_faixas(y, e, limiar=0.25, precisao_alvo=0.60)
    val = validacao_por_faixa(y, e, cortes)
    assert val["Médio"]["n"] == 0 and val["Médio"]["taxa_alto_risco"] is None   # o app precisa aceitar isso


def test_atribuir_faixa():
    """Os escores caem na faixa certa em cada corte."""
    cortes = {"limiar_amarelo": 0.3, "corte_vermelho": 0.6}
    assert list(atribuir_faixa([0.1, 0.3, 0.59, 0.6, 0.9], cortes)) == ["Baixo", "Médio", "Médio", "Alto", "Alto"]


# ---- arquivos que o dashboard lê -------------------------------------------------------------------
def test_pipeline_gera_os_arquivos_do_dashboard(executado):
    """O pipeline grava previsões, histórico e metadados coerentes entre si."""
    c, resumo = executado
    pasta = c.raiz / "app" / "dados"
    pred = pd.read_csv(pasta / "predicoes.csv.gz")
    hist = pd.read_csv(pasta / "historico.csv.gz")
    meta = json.loads((pasta / "modelo_meta.json").read_text(encoding="utf-8"))

    assert len(pred) == resumo["escolas_previstas"] > 0
    assert set(pred["faixa"]) <= set(FAIXAS) and pred["faixa"].notna().all()
    assert sorted(pred["posicao"]) == list(range(1, len(pred) + 1))       # ranking sem buracos nem empates
    assert pred["percentil"].between(0, 100).all()
    assert (pred["ano_alvo"] == pred["ano_ref"] + 1).all()
    assert pred["nome_escola"].str.startswith("Escola ").all()             # sem cadastro do INEP: usa o código
    assert set(hist["id_escola"]) == set(pred["id_escola"])
    assert meta["ano_ref"] == int(pred["ano_ref"].iloc[0]) and meta["fonte_dados"] == "sintetico"
    assert set(meta["faixas"]["validacao"]) == set(FAIXAS)


def test_nome_ausente_usa_o_codigo_como_rotulo():
    """Sem nome da escola ou do município, o rótulo é o código."""
    from pi_evasao.modelagem.predicao import _nomes
    df = pd.DataFrame({"id_escola": [1, 2, 3], "id_municipio": [10, 10, 20], "nome_escola": ["EMEF A", None, None],
                       "nome_municipio": ["São Paulo", "São Paulo", None]})
    r = _nomes(df)
    assert list(r["nome_escola"]) == ["EMEF A", "Escola 2", "Escola 3"]
    assert list(r["nome_municipio"]) == ["São Paulo", "São Paulo", "Município 20"]
