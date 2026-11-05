"""Planilhas OFICIAIS do INEP: a única fonte de dados reais do projeto (Etapa 1, ADR 0014).

Três arquivos por escola e por ano, em ``download.inep.gov.br`` (página "Indicadores Educacionais"):

- rendimento (taxa de abandono; também traz rede, município e nome da escola);
- distorção idade-série (TDI);
- média de alunos por turma (ATU).

Estes três arquivos são o conjunto mínimo que sustenta o modelo (ADR 0014). Não exige conta em nenhum
serviço. Disponíveis de 2016 em diante (2015 tem outra estrutura de arquivos e ficou de fora).

Este módulo baixa os zips em sequência e com repetição (o servidor derruba conexões simultâneas), lê só a
UF do estudo e guarda o resultado em CSV pequeno em ``dados/bruto/inep/``.

**A leitura é POSICIONAL**: as colunas de identificação ficam nas mesmas posições em todos os anos, mas os
nomes do cabeçalho mudam (``PK_COD_ENTIDADE`` em 2016, ``CO_ENTIDADE`` depois) e o cabeçalho do ATU de
2016 troca código e nome da escola. O total do ensino fundamental é achado pelo nome do cabeçalho
(``..._FUN`` ou ``FUN_CAT_0``) e a quantidade de colunas achadas é conferida: se o INEP mudar o layout, falha
alto em vez de ler a coluna errada.

Requer ``openpyxl``. A primeira execução baixa ~1,4 GB e leva alguns minutos; as seguintes usam o cache.
"""

from __future__ import annotations

import re
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..config import Config
from ..erros import ErroAcionavel
from ..log import obter_logger
from .base import COLUNAS, FonteDados

log = obter_logger("inep")

URL = "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/{ano}/{arquivo}"
ANO_MIN = 2016
# posições fixas das colunas de identificação (conferidas nos três arquivos, 2016 a 2025)
POS_ANO, POS_UF, POS_MUN, POS_NOME_MUN, POS_ESCOLA, POS_NOME_ESCOLA, POS_REDE = 0, 2, 3, 4, 5, 6, 8
TOTAL_EF = re.compile(r"FUN(_CAT_0)?$")  # cabeçalho do total do ensino fundamental (as demais colunas têm sufixo)


def nome_rendimento(ano: int) -> str:
    """O INEP mudou o nome do arquivo de rendimento duas vezes."""
    return {2016: "TAXA_REND_2016_ESCOLAS.zip", 2017: "TAXA_REND_2017_ESCOLAS.zip",
            2018: "TX_REND_ESCOLAS_2018.zip"}.get(ano, f"tx_rend_escolas_{ano}.zip")


def arquivos(ano: int) -> dict[str, str]:
    """Indicador -> nome do zip do ano. ``rend`` traz 3 totais (aprovação, reprovação, abandono); só o último importa."""
    return {"rend": nome_rendimento(ano), "tdi": f"TDI_{ano}_ESCOLAS.zip", "atu": f"ATU_{ano}_ESCOLAS.zip"}


def _para_numero(v) -> float:
    """Converte a célula em número; ``--`` (sem informação) e texto viram NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):  # '--' = sem informação
        return float("nan")


def ler_planilha(xlsx: Path, indicador: str, uf: str) -> pd.DataFrame:
    """Linhas da UF de uma planilha: ``ano, id_escola`` + ``valor`` (e, no rendimento, rede, município e nomes)."""
    import openpyxl

    # modo somente leitura: as planilhas têm dezenas de milhares de linhas
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    try:
        col, linhas = None, []
        for lin in wb[wb.sheetnames[0]].iter_rows(values_only=True):
            if col is None:
                # o cabeçalho com nomes de máquina é a linha com NO_REGIAO na 2ª coluna (acima dele há título e rótulos)
                if len(lin) > 9 and lin[1] == "NO_REGIAO":  # linha dos nomes de máquina do cabeçalho
                    achadas = [j for j, nome in enumerate(lin) if nome and TOTAL_EF.search(str(nome).upper())]
                    esperadas = 3 if indicador == "rend" else 1
                    if len(achadas) != esperadas:
                        raise ErroAcionavel(f"{xlsx.name}: {len(achadas)} colunas de total do fundamental, esperado "
                                            f"{esperadas}: o layout do INEP mudou? Confira os cabeçalhos.")
                    col = achadas[-1]  # no rendimento: aprovação, reprovação, abandono
                continue
            # depois do cabeçalho, cada linha é uma escola; guarda só as da UF do estudo
            if len(lin) > col and lin[POS_UF] == uf:
                base = [lin[POS_ANO], lin[POS_ESCOLA], _para_numero(lin[col])]
                if indicador == "rend":
                    base += [lin[POS_MUN], str(lin[POS_REDE]).lower(), lin[POS_NOME_ESCOLA], lin[POS_NOME_MUN]]
                linhas.append(base)
        if col is None:
            raise ErroAcionavel(f"Cabeçalho não encontrado em {xlsx.name}: o layout do INEP mudou?")
    finally:
        wb.close()  # no Windows, sem isto o diretório temporário não é apagado e esconde o erro real
    nomes = ["ano", "id_escola", "valor"] + (["id_municipio", "rede", "nome_escola", "nome_municipio"] if indicador == "rend" else [])
    # converte para números; linhas sem ano ou código (rodapé da planilha) são descartadas
    df = pd.DataFrame(linhas, columns=nomes)
    for c in ("ano", "id_escola", *(["id_municipio"] if indicador == "rend" else [])):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ano", "id_escola"]).astype({"ano": "int64", "id_escola": "int64"})
    if indicador == "rend":
        df["id_municipio"] = df["id_municipio"].astype("Int64")
        for c in ("nome_escola", "nome_municipio"):
            df[c] = df[c].astype("string").str.strip()
    return df.drop_duplicates(["ano", "id_escola"])


def baixar(url: str, destino: Path, tentativas: int = 4) -> None:
    """Baixa ``url`` para ``destino``, repetindo com espera crescente (queda de conexão é comum em arquivos de dezenas de MB)."""
    for i in range(tentativas):
        try:
            # Sem "Accept", o portal da SEDUC-SP responde 403 a clientes Python (ADR 0010).
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=300) as r, open(destino, "wb") as f:
                while bloco := r.read(1 << 20):
                    f.write(bloco)
            return
        except OSError as e:  # URLError, ConnectionResetError, timeout: todos herdam de OSError
            destino.unlink(missing_ok=True)
            if i == tentativas - 1:
                raise ErroAcionavel(f"Falha ao baixar {url} após {tentativas} tentativas: {e}. "
                                    "Verifique a conexão e a URL de origem.") from e
            espera = 2 ** (i + 1)
            log.warning("Falha ao baixar %s (%s); nova tentativa em %d s", url.rsplit("/", 1)[-1], e, espera)
            time.sleep(espera)


def _tarefa(args: tuple) -> pd.DataFrame:
    """Um arquivo: usa o cache CSV, ou lê o zip já baixado e grava o cache. (Roda em processo filho.)"""
    indicador, zip_local, destino, uf = args
    destino, zip_local = Path(destino), Path(zip_local)
    # cache: se o CSV já existe, não lê o xlsx de novo
    if destino.exists():
        return pd.read_csv(destino)
    # extrai só o xlsx do zip, numa pasta temporária apagada ao final
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_local) as z:
            nome = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
            z.extract(nome, tmp)
        df = ler_planilha(Path(tmp) / nome, indicador, uf)
    df.to_csv(destino, index=False)
    zip_local.unlink(missing_ok=True)  # o CSV do cache basta; o zip tem dezenas de MB
    return df


def carregar_indicadores(cfg: Config, anos: Sequence[int], uf: str) -> dict[str, pd.DataFrame]:
    """Uma tabela por indicador (``rend``, ``tdi``, ``atu``), todos os anos empilhados."""
    if min(anos) < ANO_MIN:
        raise ErroAcionavel(f"O INEP publica estes arquivos por escola só a partir de {ANO_MIN}; recorte pede {min(anos)}.")
    # cache em CSV pequeno por (UF, arquivo), para não baixar nem ler o xlsx duas vezes
    cache = cfg.caminho("bruto") / "inep"
    cache.mkdir(parents=True, exist_ok=True)
    # um arquivo por indicador e por ano
    tarefas = [(ind, a, cache / nome, cache / f"{uf}_{nome[:-4]}.csv", uf)
               for a in anos for ind, nome in arquivos(a).items()]
    log.info("Planilhas oficiais do INEP: %d arquivos (%d-%d); cache em %s", len(tarefas), min(anos), max(anos), cache)

    for _, ano, zip_local, csv, _ in tarefas:  # 1) download sequencial dos que ainda não estão em cache
        if not csv.exists() and not zip_local.exists():
            log.info("Baixando %s", zip_local.name)
            baixar(URL.format(ano=ano, arquivo=zip_local.name), zip_local)

    with ProcessPoolExecutor(max_workers=min(4, len(tarefas))) as ex:  # 2) leitura dos xlsx em paralelo
        dfs = list(ex.map(_tarefa, [(i, str(z), str(c), u) for i, _, z, c, u in tarefas]))
    por_ind: dict[str, list] = {"rend": [], "tdi": [], "atu": []}
    for (ind, *_), df in zip(tarefas, dfs):
        por_ind[ind].append(df)
    return {ind: pd.concat(v, ignore_index=True) for ind, v in por_ind.items()}


class FonteInep(FonteDados):
    """Fonte real: monta o painel escola-ano a partir das três planilhas oficiais do INEP."""
    nome = "inep"

    def __init__(self, cfg: Config):
        """Guarda a configuração (pastas de cache e recorte)."""
        self.cfg = cfg

    def carregar(self, anos: Sequence[int], uf: str, redes: Sequence[str]) -> pd.DataFrame:
        """Junta rendimento, distorção e alunos por turma e filtra as redes do estudo."""
        # o rendimento é a base (traz rede, município e nomes); os outros indicadores entram por escola e ano
        t = carregar_indicadores(self.cfg, anos, uf)
        df = t["rend"].rename(columns={"valor": "taxa_abandono_ef"})
        # só as redes do estudo (a planilha traz também federal e privada)
        df = df[df["rede"].isin([r.lower() for r in redes])]
        for ind, coluna in (("tdi", "tdi_ef"), ("atu", "atu_ef")):
            df = df.merge(t[ind].rename(columns={"valor": coluna}), on=["ano", "id_escola"], how="left")
        df["fonte"] = self.nome
        return df[[*COLUNAS, "fonte"]].sort_values(["id_escola", "ano"]).reset_index(drop=True)
