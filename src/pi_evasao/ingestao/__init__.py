"""Etapa 1 - ingestão. ``obter_fonte`` escolhe o adaptador: inep | sintetico."""

from __future__ import annotations

from ..config import Config
from ..erros import ErroAcionavel
from .base import FonteDados


def obter_fonte(nome: str, cfg: Config) -> FonteDados:
    """Devolve o adaptador de dados pelo nome (``inep`` ou ``sintetico``)."""
    if nome == "sintetico":
        from .sintetico import FonteSintetica
        return FonteSintetica(cfg.n_escolas_sintetico, cfg.seed, cfg.persistencia_sintetico,
                              cfg.sujeira_sintetico)
    if nome == "inep":
        from .inep import FonteInep
        return FonteInep(cfg)
    raise ErroAcionavel(f"Fonte desconhecida: {nome!r}. Use 'inep' ou 'sintetico'.")
