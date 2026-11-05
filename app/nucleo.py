"""Núcleo do dashboard (Etapa 4): dados, faixas de risco, formatação, acesso e feedback.

O app é AUTOCONTIDO: usa só pandas, plotly e streamlit e lê os arquivos que ``pi-evasao prever`` grava em
``app/dados/``. Não importa scikit-learn nem o pacote ``pi_evasao``, o que o mantém leve para hospedar e
independente do framework (ADR 0007). Cada página nova é um arquivo em ``paginas/``.

Duas regras de comunicação vêm do ADR 0004 e valem para todas as páginas:
- nunca exibir a probabilidade bruta (os pesos de classe a inflam); mostrar faixa, posição e a taxa
  observada na validação;
- a cor da faixa nunca aparece sozinha: sempre com forma e texto (acessibilidade).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ_APP = Path(__file__).resolve().parent

FAIXAS = ("Alto", "Médio", "Baixo")
# "Prioridade de apoio" em vez de "risco": a lista pode circular, e o rótulo não deve estigmatizar a escola (avaliação v1)
ROTULO = {"Alto": "Prioridade alta", "Médio": "Prioridade média", "Baixo": "Prioridade baixa"}
SIMBOLO = {"Alto": "▲", "Médio": "■", "Baixo": "●"}   # formas distintas: não depende só da cor
COR = {"Alto": "#d03b3b", "Médio": "#fab219", "Baixo": "#0ca30c"}   # paleta de status validada (skill dataviz)
FUNDO = {"Alto": "#f4c7c7", "Médio": "#fde7b0", "Baixo": "#c9ecc9"}  # tons claros p/ células, texto escuro por cima
# O Streamlit não tem tradução própria: o texto do filtro vazio ("Choose options") é trocado aqui
PLACEHOLDER = "Escolha uma ou mais opções"


def rotulo_faixa(faixa: str) -> str:
    """Texto da faixa com símbolo (a cor nunca aparece sozinha)."""
    return f"{SIMBOLO[faixa]} {ROTULO[faixa]}"


# ---- formatação pt-BR ---------------------------------------------------------------------------
def fmt(valor, casas: int = 1) -> str:
    """Decimal em pt-BR; ausente vira um traço."""
    return "—" if valor is None or pd.isna(valor) else f"{valor:.{casas}f}".replace(".", ",")


def fmt_i(valor) -> str:
    """Inteiro com ponto de milhar."""
    return "—" if valor is None or pd.isna(valor) else f"{int(valor):,}".replace(",", ".")


# ---- dados --------------------------------------------------------------------------------------
def dir_dados() -> Path:
    """Pasta dos arquivos gerados por ``prever`` (troca por variável de ambiente, útil nos testes)."""
    return Path(os.environ.get("PI_EVASAO_APP_DADOS", RAIZ_APP / "dados"))


def dir_feedback() -> Path:
    """Pasta onde o feedback dos usuários é gravado em CSV."""
    return Path(os.environ.get("PI_EVASAO_APP_FEEDBACK", RAIZ_APP / "dados_feedback"))


def _exigir_dados(pasta: Path) -> None:
    """Interrompe a página com uma instrução clara se os arquivos do dashboard não existem."""
    if not (pasta / "predicoes.csv.gz").exists():
        st.error("Os dados do dashboard não foram encontrados. Rode `pi-evasao prever` para gerá-los.")
        st.stop()


# A chave do cache é (pasta, versão do arquivo): dados de outra pasta ou regravados por um novo ``prever``/deploy
# nunca reaproveitam o cache antigo (o processo do Streamlit Cloud sobrevive entre deploys).
@st.cache_data(show_spinner=False)
def _predicoes(pasta: str, versao: int) -> pd.DataFrame:
    """Lê as previsões por escola (com cache por pasta e versão do arquivo)."""
    df = pd.read_csv(Path(pasta) / "predicoes.csv.gz")
    if "diretoria" not in df:  # dados gerados antes do ADR 0010
        df["diretoria"] = pd.NA
    return df


@st.cache_data(show_spinner=False)
def _historico(pasta: str, versao: int) -> pd.DataFrame:
    """Lê a série anual dos indicadores (com cache por pasta e versão do arquivo)."""
    return pd.read_csv(Path(pasta) / "historico.csv.gz")


@st.cache_data(show_spinner=False)
def _meta(pasta: str, versao: int) -> dict:
    """Lê os metadados do modelo (com cache por pasta e versão do arquivo)."""
    return json.loads((Path(pasta) / "modelo_meta.json").read_text(encoding="utf-8"))


def _versao(arquivo: str) -> int:
    """Data de modificação do arquivo de dados, que entra na chave do cache."""
    return (dir_dados() / arquivo).stat().st_mtime_ns


def carregar_predicoes() -> pd.DataFrame:
    """Previsões de risco por escola para o ano seguinte."""
    _exigir_dados(dir_dados())
    return _predicoes(str(dir_dados()), _versao("predicoes.csv.gz"))


def carregar_historico() -> pd.DataFrame:
    """Histórico anual de abandono, distorção e alunos por turma."""
    _exigir_dados(dir_dados())
    return _historico(str(dir_dados()), _versao("historico.csv.gz"))


def carregar_meta() -> dict:
    """Metadados do modelo: desempenho, cortes das faixas e datas."""
    _exigir_dados(dir_dados())
    return _meta(str(dir_dados()), _versao("modelo_meta.json"))


def opcoes_diretoria(pred: pd.DataFrame) -> list[str]:
    """Diretorias presentes nas previsões (vazio se ``prever`` rodou sem as fontes de diretoria; ADR 0010)."""
    return sorted(pred["diretoria"].dropna().unique())


def e_socioeducativa(nomes: pd.Series) -> pd.Series:
    """Unidades escolares dentro de centros socioeducativos (Fundação CASA): abandono muito alto por rotatividade de
    internos, não por evasão; misturadas às escolas regulares, distorcem médias e rankings."""
    return nomes.str.contains("SOCIOEDUC", case=False, na=False)


def explicacao_escola(linha: pd.Series, pred: pd.DataFrame, ano_ref: int) -> str:
    """Texto curto, próprio da escola, dizendo em quais indicadores ela está entre as 25% mais altas.
    Descreve onde a escola se destaca frente às demais, não a contribuição de cada indicador no modelo."""
    textos = {"taxa_abandono_ef": f"abandono de {fmt(linha['taxa_abandono_ef'])}% em {ano_ref}",
              "tdi_ef": f"distorção idade-série de {fmt(linha['tdi_ef'])}%",
              "delta_taxa_abandono_ef": f"aumento de {fmt(linha['delta_taxa_abandono_ef'])} p.p. no abandono em relação a {ano_ref - 1}",
              "atu_ef": f"turmas com {fmt(linha['atu_ef'])} alunos em média"}
    destaques = []
    for coluna, texto in textos.items():
        valor = linha[coluna]
        if pd.notna(valor) and (pred[coluna] < valor).mean() >= 0.75:
            destaques.append(f"{texto} (maior que {fmt(100 * (pred[coluna] < valor).mean(), 0)}% das escolas)")
    if destaques:
        frase = "Nesta escola chamam atenção: " + "; ".join(destaques) + "."
    else:
        frase = "Nenhum indicador desta escola está entre os 25% mais altos; a faixa vem da combinação deles."
    if pd.isna(linha["delta_taxa_abandono_ef"]):
        frase += (f" Sem taxa de abandono de {ano_ref - 1}: não dá para saber se {ano_ref} foi um pico isolado "
                  "ou um problema que se repete.")
    return frase


def _latin1(texto) -> str:
    """As fontes padrão do PDF só têm latin-1: troca símbolos fora dele (acentos do português são preservados)."""
    texto = str(texto)
    for de, para in {"—": "-", "≈": "~", "→": "->", "▲ ": "", "■ ": "", "● ": ""}.items():
        texto = texto.replace(de, para)
    return texto.encode("latin-1", "replace").decode("latin-1")


def pdf_tabela(titulo: str, informacoes: list[str], tabela: pd.DataFrame, larguras: tuple[int, ...] | None = None) -> bytes:
    """PDF A4 paisagem com título, linhas de contexto (filtros, data, avisos) e a tabela já formatada como texto."""
    from fpdf import FPDF   # só aqui: o resto do app não depende do fpdf2

    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _latin1(titulo), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    for linha in informacoes:
        pdf.multi_cell(0, 5, _latin1(linha), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", size=7)
    with pdf.table(col_widths=larguras, line_height=4, repeat_headings=1, text_align="LEFT") as t:
        for valores in [tabela.columns, *tabela.itertuples(index=False)]:
            linha = t.row()
            for v in valores:
                linha.cell(_latin1(v))
    return bytes(pdf.output())


def risco_esperado(percentil: pd.Series, taxas: list[float]) -> pd.Series:
    """Taxa de alto risco observada, no teste fora do tempo, para escolas naquela posição do ranking
    (``taxas`` vem de ``modelo_meta.json``; ``percentil`` 100 = maior risco). Somada sobre uma lista, dá
    quantas escolas dela devem de fato ficar em alto risco: é uma taxa de grupo, não a chance de uma escola."""
    n = len(taxas)
    faixa = ((percentil / 100) * n).astype(int).clip(0, n - 1)
    return faixa.map(dict(enumerate(taxas)))


def taxa_geral_de_alto_risco(validacao: dict) -> float:
    """Taxa de alto risco no ano seguinte considerando todas as faixas (a referência de comparação)."""
    total = sum(v["n"] for v in validacao.values())
    return sum(v["n"] * (v["taxa_alto_risco"] or 0) for v in validacao.values()) / total


def aviso_de_fonte(meta: dict) -> None:
    """Faixa vermelha no topo se o modelo foi treinado com dados sintéticos (nunca pode passar despercebido)."""
    if meta.get("fonte_dados") == "sintetico":
        st.error("**DADOS SINTÉTICOS.** Esta tela usa dados fabricados para teste do software. "
                 "Nada aqui descreve escolas reais.")


# ---- acesso -------------------------------------------------------------------------------------
def _segredo(nome: str, variavel_de_ambiente: str) -> str | None:
    """Segredo do Streamlit (``.streamlit/secrets.toml``) ou variável de ambiente; None se não houver."""
    try:
        valor = st.secrets.get(nome)
    except Exception:  # sem arquivo de segredos
        valor = None
    return valor or os.environ.get(variavel_de_ambiente)


def exigir_acesso() -> None:
    """Pede senha se ``senha_app`` estiver configurada (2ª camada; a 1ª é o convite da hospedagem)."""
    # sem senha configurada (ou já autenticado), a página abre direto
    esperada = _segredo("senha_app", "PI_EVASAO_SENHA")
    if not esperada or st.session_state.get("acesso_ok"):
        return
    st.title("Acesso restrito")
    st.write("Este protótipo é de uso restrito à equipe do projeto e aos avaliadores convidados.")
    digitada = st.text_input("Senha", type="password", key="senha_acesso")
    # comparação em tempo constante, para não vazar a senha pelo tempo de resposta
    if digitada and hmac.compare_digest(digitada.encode(), esperada.encode()):
        st.session_state["acesso_ok"] = True
        st.rerun()
    elif digitada:
        st.error("Senha incorreta.")
    st.stop()


def senha_admin_confere(digitada: str) -> bool:
    """Confere a senha da área da equipe, em tempo constante."""
    esperada = _segredo("senha_admin", "PI_EVASAO_SENHA_ADMIN")
    return bool(esperada and digitada and hmac.compare_digest(digitada.encode(), esperada.encode()))


def admin_configurado() -> bool:
    """Diz se há senha de administrador configurada (sem ela, a área da equipe fica fechada)."""
    return bool(_segredo("senha_admin", "PI_EVASAO_SENHA_ADMIN"))


# ---- feedback -----------------------------------------------------------------------------------
# Destino principal: uma planilha Google, via o Apps Script ``app/planilha_feedback.gs`` publicado como app da web
# (segredos ``feedback_url`` e ``feedback_token``). Só usa a biblioteca padrão: nenhuma dependência nova.
# Sem a planilha configurada (uso local, testes) ou se ela falhar, a resposta vai para ``feedback.csv``, que no
# Streamlit Cloud se perde quando o app reinicia ou hiberna: por isso a planilha é o destino de produção.
def _planilha(pedido: dict) -> dict:
    """Envia um pedido ao Apps Script da planilha e devolve a resposta JSON. O token vai no corpo, nunca na URL."""
    corpo = json.dumps({"token": _segredo("feedback_token", "PI_EVASAO_FEEDBACK_TOKEN"), **pedido}, default=str)
    req = urllib.request.Request(_segredo("feedback_url", "PI_EVASAO_FEEDBACK_URL"), corpo.encode("utf-8"),
                                 {"Content-Type": "application/json"})
    # o Apps Script responde com um redirecionamento; o urllib o segue sozinho (como GET, que é o esperado)
    with urllib.request.urlopen(req, timeout=20) as resposta:
        dados = json.load(resposta)
    if "erro" in dados:
        raise RuntimeError(f"planilha recusou o pedido: {dados['erro']}")
    return dados


def planilha_configurada() -> bool:
    """Diz se o feedback vai para a planilha Google (sem ela, fica só no disco do servidor)."""
    return bool(_segredo("feedback_url", "PI_EVASAO_FEEDBACK_URL"))


def _csv_feedback() -> Path:
    return dir_feedback() / "feedback.csv"


def salvar_feedback(registro: dict) -> str:
    """Grava uma resposta na planilha (ou no CSV local, se não houver planilha ou ela falhar). Devolve o destino."""
    linha = {"criado_em": datetime.now().isoformat(timespec="seconds"), **registro}
    if planilha_configurada():
        try:
            _planilha({"acao": "gravar", "registro": linha})
            return "planilha"
        except Exception:  # rede fora, token errado...: não perder a resposta, cair no CSV
            logging.exception("Falha ao gravar feedback na planilha; gravando no CSV local")
    arquivo = _csv_feedback()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([linha]).to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False, encoding="utf-8")
    return "csv"


def ler_feedback() -> pd.DataFrame:
    """Respostas já gravadas: as da planilha mais as que caíram no CSV local (vazio se não há nenhuma)."""
    partes = []
    if planilha_configurada():
        linhas = _planilha({"acao": "ler"})["linhas"]   # 1ª linha = cabeçalho
        if linhas:
            partes.append(pd.DataFrame(linhas[1:], columns=linhas[0]))
    if _csv_feedback().exists():
        partes.append(pd.read_csv(_csv_feedback()))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


# ---- regras de negócio do CEU (Módulo 3: simulador) ----------------------------------------------
# Gatilhos reativos hoje em uso, informados pela gestão (ver docs/relatorio/01-introducao.md):
# 5 dias consecutivos -> contato com a família; 20% de faltas acumuladas -> Busca Ativa Escolar;
# 10 dias consecutivos -> abandono oficial (perda de vaga). São regras POR ALUNO; o alerta precoce
# do Módulo 2 é por ESCOLA e independe de qualquer falta ter ocorrido (ver docs/relatorio/05-limitacoes.md).
SITUACAO_CEU = {
    "sem_alerta": "🟢 Sem alerta",
    "contato_familia": "🟡 Contato com a família",
    "busca_ativa": "🟠 Busca Ativa Escolar",
    "abandono_oficial": "🔴 Abandono oficial (perda de vaga)",
}


def situacao_ceu(faltas_consecutivas: int, pct_faltas_acumuladas: float) -> str:
    """Situação de um aluno segundo os 3 gatilhos reativos do CEU. Quando mais de uma regra se
    aplica, prevalece a mais grave (10 dias > 20% > 5 dias)."""
    if faltas_consecutivas >= 10:
        return SITUACAO_CEU["abandono_oficial"]
    if pct_faltas_acumuladas >= 20:
        return SITUACAO_CEU["busca_ativa"]
    if faltas_consecutivas >= 5:
        return SITUACAO_CEU["contato_familia"]
    return SITUACAO_CEU["sem_alerta"]
