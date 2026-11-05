"""Serialização do modelo final com metadados (Etapa 3 do PI).

Papel no pipeline: o dashboard da Etapa 4 carrega ``modelo_final.joblib``. Sem metadados,
um ``.joblib`` gerado com outra versão do scikit-learn ou com outra ordem de colunas quebra em
SILÊNCIO (ou, pior, prevê errado). ``modelo_meta.json`` registra versões, colunas esperadas,
limiar, métricas, fonte dos dados e hash do dataset, e ``carregar_modelo`` valida versões.
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib import metadata

import joblib

from ..config import Config
from ..erros import ErroAcionavel

ARQ_MODELO = "modelo_final.joblib"
ARQ_META = "modelo_meta.json"


def _gravar_meta(cfg: Config, meta: dict) -> None:
    """Único ponto que grava ``modelo_meta.json`` (UTF-8, acentos preservados)."""
    (cfg.caminho("modelos") / ARQ_META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def salvar_modelo(cfg: Config, pipeline, meta: dict) -> None:
    """Serializa o modelo final com um JSON de metadados (versões, colunas, limiar, hash)."""
    pasta = cfg.caminho("modelos")
    joblib.dump(pipeline, pasta / ARQ_MODELO)
    meta = {
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "versoes": {p: metadata.version(p) for p in ("scikit-learn", "pandas", "numpy", "joblib")},
        **meta,
    }
    _gravar_meta(cfg, meta)


def ler_meta(cfg: Config) -> dict:
    """Lê o JSON de metadados do modelo salvo."""
    return json.loads((cfg.caminho("modelos") / ARQ_META).read_text(encoding="utf-8"))


def atualizar_meta(cfg: Config, **campos) -> None:
    """Acrescenta campos ao JSON de metadados (usado pela avaliação)."""
    meta = ler_meta(cfg)
    meta.update(campos)
    _gravar_meta(cfg, meta)


def carregar_modelo(cfg: Config):
    """Devolve ``(pipeline, meta)``; falha com instrução se o ambiente divergir do treino."""
    pasta = cfg.caminho("modelos")
    if not (pasta / ARQ_MODELO).exists():
        raise ErroAcionavel("Modelo não encontrado. Rode 'pi-evasao treinar' (ou 'pipeline') primeiro.")
    meta = ler_meta(cfg)
    atual = metadata.version("scikit-learn")
    if meta["versoes"]["scikit-learn"] != atual:
        raise ErroAcionavel(
            f"Modelo treinado com scikit-learn {meta['versoes']['scikit-learn']}, ambiente tem {atual}. "
            "Retreine com 'pi-evasao treinar'.")
    return joblib.load(pasta / ARQ_MODELO), meta
