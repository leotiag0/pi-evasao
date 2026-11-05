"""Gerador sintético de painel escola-ano (andaime de teste).

Papel no pipeline (Etapa 1): permite testar o ETL, as features e o modelo sem rede nem download (a suíte
de testes roda offline). Os dados reais do INEP não exigem conta nem credencial
(ADR 0014), então o andaime NÃO é necessário para desenvolver: é só para os testes.

AVISO METODOLÓGICO: os dados são fabricados por este código. Qualquer métrica obtida com eles mede a
capacidade do modelo de recuperar o mecanismo aqui embutido, NÃO o risco de escolas reais. Toda linha
carrega ``fonte='sintetico'``.

Mecanismo embutido (ordens de grandeza aproximadas do INEP; não é ajuste a dados reais):

- Cada escola tem uma *vulnerabilidade latente* que evolui como AR(1) (persistência configurável). Sem
  persistência entre anos não haveria sinal temporal e o modelo t -> t+1 nada teria a aprender.
- Abandono e distorção idade-série crescem com a vulnerabilidade; alunos por turma são quase estáticos.
- 2020 e 2021 têm abandono artificialmente baixo (aprovação em massa / ensino remoto).
- Entrada e saída de escolas ao longo dos anos (painel desbalanceado).
- Opcionalmente injeta "sujeira" (NaN, sentinela -1, taxa > 100, linhas duplicadas) para exercitar o ETL.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .base import COLUNAS, FonteDados

_FATOR_PANDEMIA = {2020: 0.45, 2021: 0.75}
_PESO_REDE = {"estadual": 0.6, "municipal": 0.4}


class FonteSintetica(FonteDados):
    """Fonte fabricada para os testes: mesma forma do painel real, mecanismo conhecido."""
    nome = "sintetico"

    def __init__(self, n_escolas: int = 5000, seed: int = 42, persistencia: float = 0.8, sujeira: bool = True):
        """Define tamanho, semente, persistência entre anos e se injeta sujeira."""
        self.n, self.seed, self.rho, self.sujeira = n_escolas, seed, persistencia, sujeira

    def carregar(self, anos: Sequence[int], uf: str, redes: Sequence[str]) -> pd.DataFrame:
        """Gera o painel escola-ano sintético (reprodutível pela semente)."""
        rng = np.random.default_rng(self.seed)
        n, K, rho = self.n, len(anos), self.rho
        # atributos fixos de cada escola: código, rede, município e vulnerabilidade latente
        ids = 35_000_000 + np.arange(1, n + 1) * 3
        pesos = np.array([_PESO_REDE.get(r, 1.0) for r in redes], dtype=float)
        rede = rng.choice(list(redes), size=n, p=pesos / pesos.sum())
        cods = 3_500_000 + np.sort(rng.choice(np.arange(100, 6000), 645, replace=False))
        pm = rng.lognormal(0, 1.0, 645)
        id_municipio = rng.choice(cods, size=n, p=pm / pm.sum())
        vuln = rng.standard_normal(n)
        atu_base = rng.normal(27, 4, n)
        # algumas escolas entram ou saem ao longo dos anos (painel desbalanceado)
        entrada = np.where(rng.random(n) < 0.05, rng.integers(1, max(K, 2), n), 0)
        saida = entrada + rng.geometric(0.015, n)

        # evolução anual: a vulnerabilidade é um AR(1) e o abandono cresce com ela
        quadros, v = [], vuln.copy()
        for k, ano in enumerate(anos):
            if k > 0:
                v = rho * v + np.sqrt(1 - rho**2) * rng.standard_normal(n)
            aband = np.clip(1.3 * np.exp(0.55 * v + 0.35 * rng.standard_normal(n)) * _FATOR_PANDEMIA.get(ano, 1.0),
                            0, 40).round(1)
            q = pd.DataFrame({
                "id_escola": ids, "ano": ano, "id_municipio": id_municipio, "rede": rede,
                "nome_escola": pd.NA, "nome_municipio": pd.NA, "taxa_abandono_ef": aband,
                "tdi_ef": np.clip(9 + 6 * v + rng.normal(0, 3.5, n), 0, 70).round(1),
                "atu_ef": np.clip(atu_base + rng.normal(0, 1, n), 8, 45).round(1),
            })
            quadros.append(q[(entrada <= k) & (k < saida)])

        # junta os anos, injeta a sujeira (se pedida) e marca a fonte
        df = pd.concat(quadros, ignore_index=True)
        if self.sujeira:
            df = self._injetar_sujeira(df, rng)
        df["fonte"] = self.nome
        return df[[*COLUNAS, "fonte"]].sort_values(["id_escola", "ano"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _injetar_sujeira(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Estraga uma pequena fração dos dados, de forma reprodutível, para exercitar o ETL."""
        df = df.copy()
        m = len(df)
        for col in ("taxa_abandono_ef", "tdi_ef", "atu_ef"):
            df.loc[rng.random(m) < 0.015, col] = np.nan
        for col in ("taxa_abandono_ef", "tdi_ef"):
            df.loc[rng.random(m) < 0.003, col] = -1.0  # sentinela de "sem informação"
        invalidas = rng.random(m) < 0.0005
        df.loc[invalidas, "tdi_ef"] = 100 + rng.uniform(1, 20, invalidas.sum())
        duplicadas = df.iloc[rng.choice(m, int(0.002 * m), replace=False)]
        return pd.concat([df, duplicadas], ignore_index=True)
