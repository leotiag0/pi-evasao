"""Painel geral de indicadores (Módulo 1 do enunciado do PI).

Evolução histórica da taxa de abandono, com filtros por rede, diretoria, município, escola e ano letivo.
Médias são simples entre escolas: o conjunto mínimo de dados (ADR 0014) não tem matrículas para ponderar.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nucleo import (PLACEHOLDER, aviso_de_fonte, e_socioeducativa, carregar_historico, carregar_meta, carregar_predicoes, fmt, fmt_i,
                    opcoes_diretoria)

COR_REDE = {"estadual": "#2a78d6", "municipal": "#eb6834"}


def media(grupo, coluna: str) -> float:
    """Média simples entre as escolas do grupo, ou NaN se vazio.

    Precisa ser NaN, não ``None``: quando TODO grupo de um ``groupby().apply()`` devolve ``None``, o pandas
    passa a devolver um DataFrame vazio em vez da Series esperada. Com NaN (float), o tipo fica consistente.
    """
    g = grupo[coluna].dropna()
    return float(g.mean()) if len(g) else float("nan")


# previsões e histórico; o histórico só traz o código da escola, então nome, rede e município vêm das previsões
meta = carregar_meta()
pred = carregar_predicoes()
# O histórico só traz id_escola; nome, rede e município vêm das previsões (atributos do ano mais recente).
hist = carregar_historico().merge(pred[["id_escola", "rede", "nome_municipio", "nome_escola", "diretoria"]],
                                  on="id_escola", how="left")

st.title("Painel geral de indicadores")
st.caption("Evolução histórica da taxa de abandono nas escolas do recorte do estudo "
           "(São Paulo, redes estadual e municipal, ensino fundamental).")
aviso_de_fonte(meta)

anos = sorted(hist["ano"].unique())
opcoes_escola = hist[["id_escola", "nome_escola", "nome_municipio"]].drop_duplicates("id_escola").copy()
opcoes_escola["rotulo"] = opcoes_escola["nome_escola"] + " — " + opcoes_escola["nome_municipio"]

# filtros da barra lateral (vazio = tudo)
with st.sidebar:
    st.header("Filtros")
    redes = st.multiselect("Rede", sorted(hist["rede"].dropna().unique()), key="painel_rede", placeholder=PLACEHOLDER)
    diretorias = []
    if opcoes_diretoria(pred):
        diretorias = st.multiselect("Diretoria de ensino", opcoes_diretoria(pred), key="painel_diretoria", placeholder=PLACEHOLDER,
                                    help="DE (rede estadual e municípios do interior) ou DRE (rede municipal da capital).")
    municipios = st.multiselect("Município", sorted(hist["nome_municipio"].dropna().unique()), key="painel_municipio", placeholder=PLACEHOLDER)
    escolas_rotulo = st.multiselect("Escola", sorted(opcoes_escola["rotulo"]), key="painel_escola", placeholder=PLACEHOLDER,
                                    help="Nome — Município, para não confundir escolas homônimas de cidades diferentes.")
    periodo = st.select_slider("Ano letivo (período)", options=anos, value=(anos[0], anos[-1]), key="painel_periodo")
    ocultar_casa = st.checkbox("Ocultar unidades socioeducativas (Fundação CASA)", value=True, key="painel_casa",
                               help="O abandono nessas unidades reflete a rotatividade de internos, não a evasão, e distorce "
                                    "médias e o ranking de municípios.")

# aplica os filtros escolhidos, um a um
df = hist[hist["ano"].between(*periodo)]
if ocultar_casa:
    df = df[~e_socioeducativa(df["nome_escola"])]
if redes:
    df = df[df["rede"].isin(redes)]
if diretorias:
    df = df[df["diretoria"].isin(diretorias)]
if municipios:
    df = df[df["nome_municipio"].isin(municipios)]
if escolas_rotulo:
    ids = opcoes_escola.loc[opcoes_escola["rotulo"].isin(escolas_rotulo), "id_escola"]
    df = df[df["id_escola"].isin(ids)]

if df.empty:
    st.info("Nenhum dado para os filtros selecionados.")
    st.stop()

# indicadores do ano mais recente do filtro e variação contra o ano anterior
ultimo_ano = int(df["ano"].max())
atual = df[df["ano"] == ultimo_ano]
media_atual = media(atual, "taxa_abandono_ef")
anterior = df[df["ano"] == ultimo_ano - 1]
media_anterior = media(anterior, "taxa_abandono_ef") if not anterior.empty else float("nan")
delta = None if pd.isna(media_atual) or pd.isna(media_anterior) else media_atual - media_anterior

c1, c3, c4 = st.columns(3)
c1.metric(f"Escolas em {ultimo_ano}", fmt_i(atual["id_escola"].nunique()))
c3.metric(f"Abandono médio em {ultimo_ano}", "—" if pd.isna(media_atual) else f"{fmt(media_atual)}%",
          help="Média simples entre as escolas do filtro.")
if delta is None:
    c4.metric("Variação vs. ano anterior", "—")
else:
    sinal = "+" if delta >= 0 else ""
    c4.metric("Variação vs. ano anterior", f"{sinal}{fmt(delta)} p.p.", delta=f"{sinal}{fmt(delta)} p.p.",
              delta_color="inverse")

# série anual da média de abandono, uma linha por rede
st.subheader("Evolução da taxa de abandono, por rede")
serie = (df.groupby(["ano", "rede"]).apply(lambda g: media(g, "taxa_abandono_ef"), include_groups=False)
        .rename("taxa").reset_index().dropna(subset=["taxa"]))
fig = go.Figure()
for rede, grupo in serie.groupby("rede"):
    grupo = grupo.sort_values("ano")
    cor = COR_REDE.get(rede, "#898781")
    fig.add_trace(go.Scatter(x=grupo["ano"], y=grupo["taxa"], mode="lines+markers", name=rede.capitalize(),
                             line=dict(width=2, color=cor),
                             marker=dict(size=7, color=cor, line=dict(color="rgba(255,255,255,0.9)", width=2))))
fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  xaxis=dict(dtick=1, showgrid=False), yaxis=dict(title="Abandono médio (%)", gridcolor="rgba(128,128,128,0.25)"))
st.plotly_chart(fig, key="painel_evolucao")

st.subheader(f"Municípios com maior abandono médio em {ultimo_ano}")
por_municipio = (atual.groupby("nome_municipio").apply(lambda g: media(g, "taxa_abandono_ef"), include_groups=False)
                 .rename("taxa").reset_index().dropna(subset=["taxa"]).sort_values("taxa", ascending=False).head(10))
if por_municipio.empty:
    st.caption("Sem dados suficientes para o ranking neste filtro.")
else:
    fig2 = go.Figure(go.Bar(x=por_municipio["taxa"], y=por_municipio["nome_municipio"], orientation="h",
                            marker_color="#2a78d6"))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Abandono médio (%)", gridcolor="rgba(128,128,128,0.25)"),
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, key="painel_municipios")

st.subheader("Escolas no filtro")
colunas = ["nome_escola", "nome_municipio", "rede", "taxa_abandono_ef", "tdi_ef", "atu_ef"]
renomeio = {"nome_escola": "Escola", "nome_municipio": "Município", "rede": "Rede", "taxa_abandono_ef": f"Abandono {ultimo_ano} (%)",
            "tdi_ef": "Distorção idade-série (%)", "atu_ef": "Alunos por turma"}
tabela = atual[colunas].rename(columns=renomeio).sort_values(f"Abandono {ultimo_ano} (%)", ascending=False)
estilo = tabela.style.format(fmt, subset=[f"Abandono {ultimo_ano} (%)", "Distorção idade-série (%)", "Alunos por turma"])
st.dataframe(estilo, hide_index=True, height=380)
st.caption("Para o detalhe de uma escola e a estimativa de risco para o ano seguinte, veja a página **Alerta precoce**.")
