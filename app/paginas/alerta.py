"""Alerta precoce por escola (Módulo 2 do enunciado do PI).

Mostra, para cada escola, a faixa de risco estimada para o ano seguinte, com filtros, exportação e um detalhe
que explica o "porquê" comparando os indicadores da escola com os das demais. Não exibe probabilidade.
"""

import plotly.graph_objects as go
import streamlit as st

from datetime import date

from nucleo import (FAIXAS, FUNDO, PLACEHOLDER, ROTULO, aviso_de_fonte, carregar_historico, carregar_meta, carregar_predicoes,
                    e_socioeducativa, explicacao_escola, fmt, fmt_i, opcoes_diretoria, pdf_tabela, risco_esperado,
                    rotulo_faixa, taxa_geral_de_alto_risco)

# dados gerados por ``pi-evasao prever``: metadados, previsões por escola e histórico anual
meta, pred, hist = carregar_meta(), carregar_predicoes(), carregar_historico()
ano_ref, ano_alvo = int(meta["ano_ref"]), int(meta["ano_alvo"])
validacao = meta["faixas"]["validacao"]
taxa_base = taxa_geral_de_alto_risco(validacao)

st.title("Alerta precoce de evasão escolar")
st.caption(f"Risco estimado para **{ano_alvo}**, com base nos indicadores de **{ano_ref}**. É uma estimativa estatística "
           "**por escola**, para priorizar apoio: não avalia a escola nem prevê o que acontecerá com alunos.")
aviso_de_fonte(meta)

# ---- filtros ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Filtros")
    redes = st.multiselect("Rede", sorted(pred["rede"].dropna().unique()), key="filtro_rede", placeholder=PLACEHOLDER,
                           help="Vazio = todas as redes.")
    diretorias = []
    if opcoes_diretoria(pred):
        diretorias = st.multiselect("Diretoria de ensino", opcoes_diretoria(pred), key="filtro_diretoria", placeholder=PLACEHOLDER,
                                    help="DE (rede estadual e municípios do interior) ou DRE (rede municipal da capital). "
                                         "Vazio = todas.")
    municipios = st.multiselect("Município", sorted(pred["nome_municipio"].unique()), key="filtro_municipio", placeholder=PLACEHOLDER,
                                help="Vazio = todos os municípios.")
    faixas = st.multiselect("Faixa de risco", FAIXAS, default=list(FAIXAS), format_func=rotulo_faixa, key="filtro_faixa", placeholder=PLACEHOLDER)
    busca = st.text_input("Buscar escola (nome ou código INEP)", key="filtro_busca")
    ocultar_casa = st.checkbox("Ocultar unidades socioeducativas (Fundação CASA)", value=True, key="filtro_casa",
                               help="Escolas dentro de centros de internação: o abandono reflete a rotatividade de internos, "
                                    "não a evasão, e distorce a comparação com escolas regulares.")

df = pred
if ocultar_casa:
    df = df[~e_socioeducativa(df["nome_escola"])]
if redes:
    df = df[df["rede"].isin(redes)]
if diretorias:
    df = df[df["diretoria"].isin(diretorias)]
if municipios:
    df = df[df["nome_municipio"].isin(municipios)]
if busca.strip():
    b = busca.strip().lower()
    df = df[df["nome_escola"].str.lower().str.contains(b, regex=False) | df["id_escola"].astype(str).str.contains(b, regex=False)]
sem_faixa = df   # o detalhe da escola ignora o filtro de faixa, para comparar escolas de faixas diferentes
df = df[df["faixa"].isin(faixas)]

# ---- visão geral --------------------------------------------------------------------------------
colunas = st.columns(4)
colunas[0].metric("Escolas exibidas", fmt_i(len(df)))
for coluna, faixa in zip(colunas[1:], FAIXAS):
    coluna.metric(rotulo_faixa(faixa), fmt_i((df["faixa"] == faixa).sum()))
    taxa = validacao[faixa]["taxa_alto_risco"]
    coluna.caption("Faixa sem escolas na validação." if taxa is None else
                   f"Na validação, {fmt(100 * taxa, 0)}% das escolas desta faixa ficaram em alto risco no ano seguinte "
                   f"(média geral: {fmt(100 * taxa_base, 0)}%).")

# ---- capacidade de atendimento (precisão entre as N primeiras) -----------------------------------
if meta.get("taxa_por_percentil") and not df.empty:
    st.subheader("Quantas escolas a equipe consegue acompanhar?")
    n = st.number_input("Número de escolas", min_value=1, max_value=len(df), value=min(50, len(df)), step=10,
                        key="capacidade", help="As N primeiras da lista filtrada, na ordem de prioridade.")
    esperado = risco_esperado(df["percentil"], meta["taxa_por_percentil"])
    no_topo, no_filtro = esperado.loc[df.nsmallest(n, "posicao").index].sum(), esperado.sum()
    c1, c2 = st.columns(2)
    c1.metric(f"Em alto risco, entre as {fmt_i(n)} primeiras", f"≈ {fmt_i(round(no_topo))} ({fmt(100 * no_topo / n, 0)}%)",
              help="Precisão esperada: quantas destas escolas devem ficar entre as 20% com maior abandono no ano seguinte.")
    c2.metric("De cada 10 escolas em risco neste filtro, a lista alcança", f"{fmt(10 * no_topo / no_filtro, 0)}" if no_filtro else "—",
              help="Das escolas deste filtro (não do Estado inteiro) que devem ficar em alto risco, quantas estão entre as N primeiras.")
    st.caption(f"Estimativa pelo que aconteceu no teste fora do tempo ({meta['teste_ano_ref']}→{meta['teste_ano_ref'] + 1}) "
               f"com escolas na mesma posição do ranking. Para comparar: escolhendo {fmt_i(n)} escolas ao acaso, "
               f"≈ {fmt(100 * taxa_base, 0)}% estariam em alto risco.")

# ---- tabela -------------------------------------------------------------------------------------
st.subheader("Escolas por prioridade")
df = df.sort_values("posicao")
tabela = df.assign(
    **{"Ordem no filtro": range(1, len(df) + 1)},
    Faixa=df["faixa"].map(rotulo_faixa),
    Aviso=df["dados_incompletos"].map({True: "dados incompletos", False: ""}),
).rename(columns={
    "posicao": "Posição em SP", "id_escola": "Código INEP", "nome_escola": "Escola", "nome_municipio": "Município", "rede": "Rede", "diretoria": "Diretoria",
    "taxa_abandono_ef": f"Abandono {ano_ref} (%)", "tdi_ef": "Distorção idade-série (%)",
    "delta_taxa_abandono_ef": "Variação do abandono (p.p.)", "atu_ef": "Alunos por turma",
})[["Ordem no filtro", "Posição em SP", "Código INEP", "Faixa", "Escola", "Município", "Diretoria", "Rede", f"Abandono {ano_ref} (%)", "Distorção idade-série (%)",
    "Variação do abandono (p.p.)", "Alunos por turma", "Aviso"]]
fundo_por_rotulo = {rotulo_faixa(f): FUNDO[f] for f in FAIXAS}
estilo = (tabela.style
          .map(lambda v: f"background-color: {fundo_por_rotulo.get(v, '')}; color: #0b0b0b", subset=["Faixa"])
          .format(lambda v: fmt(v, 1), subset=[f"Abandono {ano_ref} (%)", "Distorção idade-série (%)",
                                                "Variação do abandono (p.p.)", "Alunos por turma"])
          .format(fmt_i, subset=["Ordem no filtro", "Posição em SP"]))
st.dataframe(estilo, hide_index=True, height=420)
st.caption("**Ordem no filtro** recomeça em 1 dentro da seleção; **Posição em SP** é a posição entre todas as escolas do Estado.")


LIMITE_PDF = 1000


def _pdf() -> bytes:
    """PDF da lista filtrada (até ``LIMITE_PDF`` escolas), gerado só quando o botão é clicado."""
    filtros = [f"{nome}: {', '.join(map(str, v))}" for nome, v in (("Rede", redes), ("Diretoria", diretorias),
               ("Município", municipios), ("Faixa", [ROTULO[f] for f in faixas])) if v]
    if busca.strip():
        filtros.append(f"Busca: {busca.strip()}")
    contagem = "; ".join(f"{ROTULO[f]}: {fmt_i((df['faixa'] == f).sum())}" for f in FAIXAS)
    pdf_cols = ["Ordem no filtro", "Posição em SP", "Código INEP", "Faixa", "Escola", "Município", "Diretoria",
                f"Abandono {ano_ref} (%)", "Distorção idade-série (%)", "Alunos por turma"]
    # ponytail: limite de linhas porque o fpdf2 leva ~3 s a cada 1.000; a lista completa vai no CSV
    texto = tabela[pdf_cols].head(LIMITE_PDF).assign(**{c: tabela[c].head(LIMITE_PDF).map(fmt) for c in pdf_cols[7:]}).fillna("—")
    corte = f" O PDF traz as {fmt_i(LIMITE_PDF)} primeiras; a lista completa está no CSV." if len(tabela) > LIMITE_PDF else ""
    return pdf_tabela(f"Alerta precoce de evasão escolar - prioridades para {ano_alvo}", [
        f"Gerado em {date.today():%d/%m/%Y}, com indicadores do INEP de {ano_ref}. Filtros: {' | '.join(filtros) or 'nenhum'}.",
        f"{fmt_i(len(df))} escolas ({contagem}).{corte}",
        "Estimativa estatística por escola, para priorizar apoio: não avalia a escola nem prevê o que acontecerá com alunos. "
        "Uso restrito à equipe; não divulgar como ranking de escolas.",
    ], texto, larguras=(16, 15, 18, 22, 62, 30, 34, 20, 22, 18))


c1, c2 = st.columns(2)
c1.download_button("Baixar a lista filtrada (CSV)",
                   tabela.round(1).to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig"),
                   "escolas_filtradas.csv", "text/csv", key="baixar_lista")
c2.download_button("Baixar a lista filtrada (PDF)", _pdf, "escolas_filtradas.pdf", "application/pdf", key="baixar_pdf")

# ---- detalhe da escola --------------------------------------------------------------------------
st.subheader("Por que esta escola está nesta faixa?")
if sem_faixa.empty:
    st.info("Nenhuma escola corresponde aos filtros.")
    st.stop()

sem_faixa = sem_faixa.sort_values("posicao")
rotulos = dict(zip(sem_faixa["id_escola"], sem_faixa["nome_escola"] + " — " + sem_faixa["nome_municipio"]))
escolhida = st.selectbox("Escola (digite para buscar)", list(rotulos), format_func=rotulos.get, key="escola_detalhe",
                         help="Lista as escolas dos filtros de rede, diretoria, município e busca, de todas as faixas.")
linha = pred.set_index("id_escola").loc[escolhida]
faixa = linha["faixa"]

if linha["dados_incompletos"]:
    st.warning(f"Esta escola não tem taxa de abandono informada em {ano_ref}; a previsão é menos confiável.")
c1, c2 = st.columns(2)
c1.metric("Faixa de risco", rotulo_faixa(faixa))
c2.metric("Posição no ranking", f"{fmt_i(linha['posicao'])} de {fmt_i(len(pred))}")
taxa_faixa = validacao[faixa]["taxa_alto_risco"]
if taxa_faixa is not None:
    st.write(f"Nas validações do modelo, **{fmt(100 * taxa_faixa, 0)}%** das escolas na faixa "
             f"**{rotulo_faixa(faixa)}** ficaram entre as 20% com maior abandono no ano seguinte "
             f"(média geral: {fmt(100 * taxa_base, 0)}%).")

indicadores = {
    "taxa_abandono_ef": (f"Abandono em {ano_ref} (%)", 1), "tdi_ef": ("Distorção idade-série (%)", 1),
    "delta_taxa_abandono_ef": ("Variação do abandono (p.p.)", 1), "atu_ef": ("Alunos por turma", 1),
}
linhas = []
for coluna, (nome, casas) in indicadores.items():
    valor = linha[coluna]
    maior_que = None if valor != valor else 100 * (pred[coluna] < valor).mean()
    linhas.append({"Indicador": nome, "Esta escola": fmt(valor, casas), "Mediana das escolas": fmt(pred[coluna].median(), casas),
                   "Maior que (% das escolas)": fmt(maior_que, 0)})
st.dataframe(linhas, hide_index=True)
st.write(explicacao_escola(linha, pred, ano_ref))
st.caption("No modelo como um todo, abandono anterior e distorção idade-série são os indicadores que mais pesam (ver a página "
           "\"Sobre os dados e limitações\"). Isto mostra onde o risco se concentra, não o que o causa.")

opcoes = {"Abandono (%)": "taxa_abandono_ef", "Distorção idade-série (%)": "tdi_ef", "Alunos por turma": "atu_ef"}
escolha = st.radio("Evolução ao longo dos anos", list(opcoes), horizontal=True, key="indicador_grafico")
coluna = opcoes[escolha]
serie = hist[hist["id_escola"] == escolhida].sort_values("ano")
# compara com a mesma rede: as municipais (abandono quase sempre 0) puxariam a mediana geral para zero
mesma_rede = hist[hist["id_escola"].isin(pred.loc[pred["rede"] == linha["rede"], "id_escola"])]
mediana = mesma_rede.groupby("ano")[coluna].median()
figura = go.Figure()
figura.add_trace(go.Scatter(x=mediana.index, y=mediana.values, mode="lines", name=f"Mediana da rede {linha['rede']}",
                            line=dict(color="#898781", width=2, dash="dash")))
figura.add_trace(go.Scatter(x=serie["ano"], y=serie[coluna], mode="lines+markers", name="Esta escola",
                            line=dict(color="#2a78d6", width=2),
                            marker=dict(size=8, color="#2a78d6", line=dict(color="rgba(255,255,255,0.9)", width=2))))
figura.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                     legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     xaxis=dict(dtick=1, showgrid=False), yaxis=dict(title=escolha, gridcolor="rgba(128,128,128,0.25)"))
st.plotly_chart(figura)

st.divider()
st.caption("Protótipo em avaliação. Modelo treinado com dados públicos do INEP, sem validação em campo. "
           "Use como apoio à decisão, não como decisão.")
