"""Logging padronizado e registro automático de execuções.

Papel no pipeline: cumpre o requisito do grupo de "documentar todos os processos para fins de
relatório". O decorador ``registrar_execucao`` grava, para CADA comando da CLI, um arquivo datado
em ``docs/execucoes/`` com parâmetros, fonte de dados, contagens, tempo e versões das bibliotecas.
Assim o relatório é montado sobre evidência gerada pela própria execução, não sobre memória.

Contrato: a função decorada recebe ``(cfg, **kwargs)`` e devolve um ``dict`` "resumo" com o que
quiser destacar (ex.: ``linhas_entrada``, ``linhas_saida``, ``fonte``, ``hash_dataset``).
Falhas também são registradas (status ``falha``) e a exceção é relançada.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import platform
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def obter_logger(nome: str = "pi_evasao") -> logging.Logger:
    """Logger nomeado; configura o formato padrão na primeira chamada."""
    log = logging.getLogger(nome)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMATO, datefmt="%H:%M:%S")
    return log


def versoes() -> dict[str, str]:
    """Versões do ambiente, para reprodutibilidade (entra em todo registro de execução)."""
    out = {"python": platform.python_version(), "sistema": platform.platform()}
    for pacote in ("pandas", "numpy", "scikit-learn", "matplotlib", "joblib", "openpyxl"):
        try:
            out[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            pass
    return out


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash curto e determinístico do conteúdo de um DataFrame (não depende do índice)."""
    h = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(h + "|".join(map(str, df.columns)).encode()).hexdigest()[:16]


def _fmt(v) -> str:
    """Formata um valor para a tabela de resumo (decimais sem zeros à direita)."""
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def _escrever_registro(cfg, comando, ts, duracao, status, params, resumo, erro) -> Path:
    """Grava o registro Markdown de uma execução em ``docs/execucoes/`` (sem sobrescrever outro)."""
    pasta: Path = cfg.caminho("execucoes")
    # nome com data e hora; se já existir, acrescenta um número para nunca sobrescrever
    base = pasta / f"{ts:%Y-%m-%d-%H%M%S}-{comando}"
    arquivo = base.with_suffix(".md")
    n = 2
    while arquivo.exists():
        arquivo = Path(f"{base}-{n}.md")
        n += 1

    # monta o Markdown: cabeçalho, parâmetros, resumo e versões do ambiente
    linhas = [
        f"# Execução: `{comando}`",
        "",
        f"- **Data/hora:** {ts:%Y-%m-%d %H:%M:%S}",
        f"- **Status:** {status}",
        f"- **Duração:** {duracao:.1f} s",
        "",
        "## Parâmetros",
        "",
    ]
    linhas += [f"- `{k}` = `{v}`" for k, v in params.items()] or ["- (nenhum)"]
    linhas += ["", "## Resumo", ""]
    if resumo:
        linhas += ["| Item | Valor |", "|---|---|"]
        linhas += [f"| {k} | {_fmt(v)} |" for k, v in resumo.items()]
    else:
        linhas += ["(sem resumo)"]
    linhas += ["", "## Versões do ambiente", "", "| Componente | Versão |", "|---|---|"]
    linhas += [f"| {k} | {v} |" for k, v in versoes().items()]
    # em caso de falha, o traceback entra no registro
    if erro:
        linhas += ["", "## Erro", "", "```", erro.strip(), "```"]
    arquivo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return arquivo


def registrar_execucao(comando: str):
    """Decorador: registra a execução do comando em ``docs/execucoes/`` (sucesso ou falha)."""

    def deco(fn):
        """Envolve a função do comando com o registro automático."""

        @functools.wraps(fn)
        def wrapper(cfg, **kwargs):
            """Cronometra o comando, registra sucesso ou falha e devolve o resumo."""
            log = obter_logger(comando)
            ts, t0 = datetime.now(), time.perf_counter()  # marca o início para medir a duração
            log.info("início | %s", {k: v for k, v in kwargs.items() if v is not None})
            try:
                resumo = fn(cfg, **kwargs) or {}
            except Exception:
                # a falha também deixa registro (com o traceback) e a exceção segue para a CLI
                _escrever_registro(
                    cfg, comando, ts, time.perf_counter() - t0, "falha",
                    kwargs, {}, traceback.format_exc(),
                )
                raise
            dur = time.perf_counter() - t0
            arquivo = _escrever_registro(cfg, comando, ts, dur, "sucesso", kwargs, resumo, None)
            log.info("concluído em %.1f s | registro: %s", dur, arquivo.relative_to(cfg.raiz))
            return resumo

        return wrapper

    return deco
