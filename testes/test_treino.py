"""Etapa 3 ponta a ponta em dados sintéticos pequenos, com tudo escrito em diretório temporário."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from pi_evasao.erros import ErroAcionavel
from pi_evasao.modelagem import persistencia
from pi_evasao.modelagem.treino import escolher_limiar


def test_pipeline_produz_todos_os_artefatos(executado):
    """O pipeline completo grava modelo, métricas, figuras, banco e dicionário."""
    c, resumo = executado
    for rel in ("modelos/modelo_final.joblib", "modelos/modelo_meta.json", "relatorios/metricas/comparativo.csv",
                "relatorios/metricas/comparativo.md", "relatorios/metricas/qualidade.md",
                "relatorios/metricas/features.md", "relatorios/figuras/figuras.md", "docs/dicionario_dados.md",
                "dados/evasao.db", "relatorios/figuras/pr_fora_do_tempo.png"):
        assert (c.raiz / rel).exists(), rel
    assert resumo["fonte"] == "sintetico"


def test_cada_comando_deixa_registro_de_execucao(executado):
    """Cada comando deixa seu registro em ``docs/execucoes/``."""
    c, _ = executado
    nomes = {p.name.split("-", 4)[-1] for p in (c.raiz / "docs" / "execucoes").glob("*.md")}
    assert {"ingerir.md", "etl.md", "features.md", "treinar.md", "avaliar.md", "pipeline.md"} <= nomes


def test_modelo_carrega_com_metadados_e_prediz(executado):
    """O modelo salvo carrega com os metadados e devolve probabilidades válidas."""
    c, _ = executado
    modelo, meta = persistencia.carregar_modelo(c)
    assert meta["fonte_dados"] == "sintetico" and "SINTETICOS" in meta["aviso"]
    assert set(meta["versoes"]) >= {"scikit-learn", "pandas"} and 0 < meta["limiar_decisao"] < 1
    X = pd.DataFrame([{col: 1.0 for col in meta["colunas_numericas"]}]).assign(rede="estadual")
    p = modelo.predict_proba(X)[0, 1]
    assert 0.0 <= p <= 1.0


def test_modelos_superam_o_baseline_fora_do_tempo(executado):
    """O modelo e a persistência superam o piso da prevalência fora do tempo."""
    c, _ = executado
    df = pd.read_csv(c.raiz / "relatorios" / "metricas" / "comparativo.csv")
    d = df.set_index("modelo")
    assert set(d.index) == {"gradient_boosting", "baseline_dummy", "baseline_persistencia"}
    assert (d.drop("baseline_dummy")["pr_auc"] > d.loc["baseline_dummy", "pr_auc"] + 0.05).all()


def test_nenhuma_metrica_suspeita_de_vazamento(executado):
    """Nenhuma PR-AUC do modelo passa de 0,95 (sinal de vazamento)."""
    c, resumo = executado
    assert resumo["possivel_vazamento"] is False
    df = pd.read_csv(c.raiz / "relatorios" / "metricas" / "comparativo.csv")
    assert df[~df.modelo.str.startswith("baseline_")]["pr_auc"].max() < 0.95


def test_teste_fora_do_tempo_nao_mistura_anos(cfg):
    """Treino e teste ficam em lados opostos do ano de corte."""
    from pi_evasao.modelagem.treino import indices_temporal
    meta = pd.DataFrame({"ano_ref": [2020, 2021, 2022, 2023, 2023, 2021]})
    tr, te = indices_temporal(meta, cfg)  # corte do teste: 2023
    assert set(meta["ano_ref"].iloc[tr]) == {2020, 2021, 2022} and set(meta["ano_ref"].iloc[te]) == {2023}


def test_limiar_f2_privilegia_recall_mais_que_f1():
    """O limiar por F2 é menor ou igual ao por F1 (mais alertas, mais recall)."""
    rng = np.random.default_rng(1)
    p = rng.random(4000)
    y = (rng.random(4000) < p * 0.6).astype(int)
    assert escolher_limiar(y, p, beta=2.0) <= escolher_limiar(y, p, beta=1.0)


def test_carregar_modelo_inexistente_e_erro_acionavel(cfg, tmp_path):
    """Sem modelo salvo, o erro manda rodar ``treinar``."""
    with pytest.raises(ErroAcionavel, match="treinar"):
        persistencia.carregar_modelo(dataclasses.replace(cfg, raiz=tmp_path))


def test_meta_e_json_valido(executado):
    """Os metadados gravados são um JSON com a métrica do teste fora do tempo."""
    c, _ = executado
    meta = json.loads((c.raiz / "modelos" / "modelo_meta.json").read_text(encoding="utf-8"))
    assert set(meta["metricas_avaliacao"]) == {"fora_do_tempo"} and meta["modelo"] == "gradient_boosting"


def test_persistencia_ordena_pelo_abandono_do_ano_t():
    """O baseline de persistência (ADR 0009) é só a taxa de abandono do ano t; NaN vira a mediana do treino."""
    from pi_evasao.modelagem.treino import Persistencia, eh_baseline
    X = pd.DataFrame({"taxa_abandono_ef": [0.0, 2.0, 10.0, np.nan], "rede": ["estadual"] * 4})
    p = Persistencia().fit(X, [0, 0, 1, 0]).predict_proba(X)[:, 1]
    assert list(p) == [0.0, 0.02, 0.10, 0.02]
    assert eh_baseline("baseline_persistencia") and not eh_baseline("gradient_boosting")


def test_taxa_por_percentil_separa_o_topo_da_base():
    """Faixa 0 = menor escore. Escore perfeito: toda a base sem risco, todo o topo em risco."""
    from pi_evasao.modelagem.avaliacao import taxa_por_percentil
    proba = np.linspace(0, 1, 100)
    t = taxa_por_percentil((proba > 0.8).astype(int), proba, n_faixas=10)
    assert len(t) == 10 and t[0] == 0.0 and t[-1] == 1.0
