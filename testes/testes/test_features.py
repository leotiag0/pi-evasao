"""Testes da Etapa 2. O de vazamento é o que protege a validade do trabalho inteiro (ADR 0002)."""

import numpy as np
import pandas as pd

from pi_evasao.features.engenharia import ALVO, FEATURES, META, construir_dataset

PROIBIDAS_EM_X = {"taxa_aprovacao_ef", "taxa_abandono_alvo", "limiar_alvo", "ano_alvo", "ano_ref", "id_escola",
                  "id_municipio", "ano", ALVO}


def test_X_nao_contem_colunas_proibidas():
    """Nenhum atributo vem do ano-alvo nem é identificador; são exatamente seis."""
    assert not (set(FEATURES) & PROIBIDAS_EM_X)
    assert not (set(FEATURES) & set(META))
    assert not any(c.endswith(("_alvo", "_t1")) for c in FEATURES)
    assert len(FEATURES) == 6  # conjunto mínimo do ADR 0014


def test_vazamento_comportamental_X_ignora_o_ano_alvo(painel_limpo, cfg):
    """Embaralha TUDO do último ano (o ano-alvo do último par) e exige X idêntico.

    Se alguma feature de um par (t -> t+1) dependesse de dados de t+1, o X mudaria.
    """
    ultimo = max(cfg.anos)
    base, _ = construir_dataset(painel_limpo, cfg)

    alterado = painel_limpo.copy()
    rng = np.random.default_rng(0)
    m = alterado["ano"] == ultimo
    for col in ("tdi_ef", "atu_ef", "taxa_abandono_ef"):
        alterado.loc[m, col] = rng.permutation(alterado.loc[m, col].to_numpy())
    novo, _ = construir_dataset(alterado, cfg)

    # Compara POR CHAVE: embaralhar o abandono de t+1 move também os NaN, e a existência de um par
    # depende de o rótulo ser não-nulo. Isso muda QUAIS pares existem, não o conteúdo de X.
    comum = base.merge(novo, on=["id_escola", "ano_ref"], suffixes=("_0", "_1"))
    assert len(comum) > 0.95 * len(base)
    for col in FEATURES:
        pd.testing.assert_series_equal(comum[f"{col}_0"], comum[f"{col}_1"], check_names=False,
                                       obj=f"feature {col} (VAZAMENTO se diferir)")
    assert (comum[f"{ALVO}_0"] != comum[f"{ALVO}_1"]).any(), "y deveria refletir o ano-alvo alterado"


def test_um_registro_por_escola_e_par(painel_limpo, cfg):
    """Um par por escola e ano de referência, sempre com o ano-alvo logo depois."""
    ds, _ = construir_dataset(painel_limpo, cfg)
    assert not ds.duplicated(["id_escola", "ano_ref"]).any()
    assert (ds["ano_alvo"] == ds["ano_ref"] + 1).all()
    assert set(ds["ano_ref"]) <= set(cfg.anos[:-1])


def test_rotulo_por_percentil_mantem_prevalencia_estavel(painel_limpo, cfg):
    """A prevalência de alto risco fica perto de 20% em todos os anos."""
    _, resumo = construir_dataset(painel_limpo, cfg)
    for ano, prev in resumo["prevalencia_por_ano_alvo"].items():
        assert 0.15 <= prev <= 0.32, f"prevalência {prev:.2%} fora do esperado em {ano}"


def test_delta_e_nan_sem_ano_anterior_e_correto_com_ele(painel_limpo, cfg):
    """A variação é ausente no primeiro ano e igual à diferença real nos demais."""
    ds, _ = construir_dataset(painel_limpo, cfg)
    primeiro = min(cfg.anos)
    assert ds.loc[ds["ano_ref"] == primeiro, "delta_taxa_abandono_ef"].isna().all()

    p = painel_limpo.set_index(["id_escola", "ano"])["taxa_abandono_ef"]
    amostra = ds[(ds["ano_ref"] == 2022) & ds["delta_taxa_abandono_ef"].notna()].head(20)
    for _, r in amostra.iterrows():
        esperado = p[(r["id_escola"], 2022)] - p[(r["id_escola"], 2021)]
        assert np.isclose(r["delta_taxa_abandono_ef"], esperado)
