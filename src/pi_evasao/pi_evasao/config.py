"""Carrega ``config/config.yaml`` em um objeto imutável.

Papel no pipeline: fonte única dos parâmetros do estudo (recorte, seed, percentil do rótulo,
beta do F-beta, caminhos). O relatório cita este arquivo como origem dos parâmetros.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_PADRAO = RAIZ / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Parâmetros do estudo, imutáveis, lidos de ``config/config.yaml``."""
    raiz: Path
    seed: int
    uf: str
    redes: tuple[str, ...]
    anos: tuple[int, ...]
    fonte_padrao: str
    percentil_risco: float
    fbeta: float
    cv_folds: int
    ano_ref_teste_temporal: int
    n_escolas_sintetico: int
    persistencia_sintetico: float
    sujeira_sintetico: bool
    precisao_vermelho: float
    caminhos: dict

    def caminho(self, chave: str) -> Path:
        """Caminho absoluto de uma chave de ``caminhos`` (criando o diretório se for pasta)."""
        p = self.raiz / self.caminhos[chave]
        if chave != "banco":
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p


def carregar_config(caminho: Path | str | None = None) -> Config:
    """Lê o YAML (o padrão ou outro caminho) e devolve a ``Config``."""
    caminho = Path(caminho) if caminho else CONFIG_PADRAO
    # cada parâmetro do YAML vira um campo do objeto imutável (o relatório cita este arquivo como origem)
    with open(caminho, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    return Config(
        raiz=RAIZ,
        seed=c["seed"],
        uf=c["recorte"]["uf"],
        redes=tuple(c["recorte"]["redes"]),
        anos=tuple(c["recorte"]["anos"]),
        fonte_padrao=c["fonte_padrao"],
        percentil_risco=c["rotulo"]["percentil"],
        fbeta=c["modelagem"]["fbeta"],
        cv_folds=c["modelagem"]["cv_folds"],
        ano_ref_teste_temporal=c["modelagem"]["ano_ref_teste_temporal"],
        n_escolas_sintetico=c["sintetico"]["n_escolas"],
        persistencia_sintetico=c["sintetico"]["persistencia"],
        sujeira_sintetico=c["sintetico"]["sujeira"],
        precisao_vermelho=c.get("faixas", {}).get("precisao_vermelho", 0.60),
        caminhos=c["caminhos"],
    )
