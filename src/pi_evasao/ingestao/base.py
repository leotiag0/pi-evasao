"""Interface comum das fontes de dados e esquema do painel escola-ano (Etapa 1).

Toda fonte devolve um DataFrame com as colunas de ``COLUNAS`` e uma coluna ``fonte`` que identifica a
origem. Essa coluna atravessa ETL, banco e modelo até o dashboard, para que dados sintéticos nunca sejam
confundidos com dados reais. O esquema é o mínimo que o modelo usa (ADR 0014): três indicadores do INEP.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from ..formato import i as _i, n as _n

# coluna -> (descrição, unidade)
DICIONARIO = {
    "id_escola": ("Código INEP da escola", ""),
    "ano": ("Ano do Censo Escolar", ""),
    "id_municipio": ("Código IBGE do município", ""),
    "rede": ("Rede de ensino (estadual | municipal)", ""),
    "nome_escola": ("Nome da escola (vazio no andaime sintético)", ""),
    "nome_municipio": ("Nome do município (vazio no andaime sintético)", ""),
    "taxa_abandono_ef": ("Taxa de abandono, ensino fundamental total (ALVO do ano seguinte)", "%"),
    "tdi_ef": ("Taxa de distorção idade-série, ensino fundamental total", "%"),
    "atu_ef": ("Média de alunos por turma, ensino fundamental total", "alunos"),
}
COLUNAS = tuple(DICIONARIO)
INDICADORES = ("taxa_abandono_ef", "tdi_ef", "atu_ef")
TAXAS = ("taxa_abandono_ef", "tdi_ef")  # percentuais: valores fora de [0, 100] são impossíveis


class FonteDados(ABC):
    """Contrato de toda fonte: devolver o painel escola-ano no esquema de ``COLUNAS``."""
    nome: str

    @abstractmethod
    def carregar(self, anos: Sequence[int], uf: str, redes: Sequence[str]) -> pd.DataFrame:
        """Painel escola-ano com as colunas de ``COLUNAS`` e ``fonte``."""


def markdown_dicionario(df: pd.DataFrame, fonte: str) -> str:
    """Dicionário de dados + cobertura por ano (vira docs/dicionario_dados.md)."""
    aviso = ("> **AVISO:** dados SINTÉTICOS. Cobertura e estatísticas descrevem o gerador, não o INEP.\n\n"
             if fonte == "sintetico" else "")
    L = ["# Dicionário de dados", "", aviso, f"- **Fonte desta carga:** {fonte}",
         f"- **Linhas:** {_i(len(df))} | **Escolas:** {_i(df['id_escola'].nunique())} | "
         f"**Anos:** {int(df['ano'].min())}-{int(df['ano'].max())}", "",
         "| Coluna | Unidade | % preenchido | Descrição |", "|---|---|---:|---|"]
    L += [f"| `{c}` | {un or '-'} | {_n(df[c].notna().mean() * 100, 1)} | {desc} |" for c, (desc, un) in DICIONARIO.items()]
    L += ["", "## Cobertura por ano", "", "| Ano | Escolas | % com taxa de abandono |", "|---:|---:|---:|"]
    L += [f"| {ano} | {_i(g['id_escola'].nunique())} | {_n(g['taxa_abandono_ef'].notna().mean() * 100, 1)} |"
          for ano, g in df.groupby("ano")]
    return "\n".join(L + [""])
