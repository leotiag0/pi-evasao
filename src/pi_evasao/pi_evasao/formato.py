"""Formatação numérica em pt-BR para os documentos gerados (vírgula decimal, ponto de milhar)."""

from __future__ import annotations


def n(x, casas: int = 3) -> str:
    """Decimal: ``n(0.5512) -> '0,551'``."""
    return f"{x:.{casas}f}".replace(".", ",")


def i(x) -> str:
    """Inteiro com separador de milhar: ``i(18387) -> '18.387'``."""
    return f"{int(x):,}".replace(",", ".")
