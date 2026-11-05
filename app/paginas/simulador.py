"""Simulador das regras de faltas do CEU (Módulo 3 do enunciado do PI).

Aplica os gatilhos reativos hoje em uso (5, 20%, 10 dias) a uma turma FICTÍCIA e editável, para comparar
com o alerta preventivo por escola do Módulo 2: "calcular o impacto preventivo antes de atingir os
limites formais". Nenhum dado de aluno real é usado ou armazenado.
"""

import numpy as np
import pandas as pd
import streamlit as st

from nucleo import SITUACAO_CEU, carregar_predicoes, fmt, fmt_i, rotulo_faixa, situacao_ceu

st.title("Simulador das regras de faltas do CEU")
st.warning("**Turma fictícia, só para demonstração.** Os dados abaixo são gerados ou digitados por você; "
          "não use nomes nem informações de alunos reais.")
st.caption("Hoje o acompanhamento de faltas dispara em 3 momentos: 5 dias consecutivos (contato com a família), "
          "20% de faltas acumuladas no período (Busca Ativa Escolar) e 10 dias consecutivos (perda de vaga). "
          "Este simulador aplica essas regras a uma turma que você edita, para visualizar o impacto **antes** de "
          "um aluno atingir os limites.")

# contexto opcional: liga o alerta preventivo da escola ao gatilho reativo por aluno
pred = carregar_predicoes()
st.subheader("Contexto (opcional): vincular a uma escola real")
rotulos = dict(zip(pred["id_escola"], pred["nome_escola"] + " — " + pred["nome_municipio"]))
escolhida = st.selectbox("Escola", [None, *rotulos], format_func=lambda i: "Nenhuma" if i is None else rotulos[i],
                         key="sim_escola")
if escolhida is not None:
    linha = pred.set_index("id_escola").loc[escolhida]
    st.info(f"O alerta precoce (Módulo 2) estima esta escola em **{rotulo_faixa(linha['faixa'])}** para "
            f"{int(linha['ano_alvo'])}, com base em indicadores da escola inteira — **antes** de qualquer aluno "
            "atingir os limites de falta abaixo. A turma simulada a seguir é independente dessa estimativa.")

st.subheader("Turma simulada")
c1, c2, c3 = st.columns(3)
dias_letivos = c1.number_input("Dias letivos no período considerado", min_value=1, max_value=250, value=50, key="sim_dias")
n_alunos = c2.slider("Número de alunos", 1, 40, 25, key="sim_n")
gerar = c3.button("Gerar turma fictícia", key="sim_gerar")

# a turma fica no estado da sessão para preservar as edições entre execuções da página
CHAVE = "sim_turma_dados"
if gerar or CHAVE not in st.session_state:
    rng = np.random.default_rng(None if gerar else 42)  # com o botão, uma turma nova a cada clique
    consecutivas = rng.choice([0, 0, 0, 1, 2, 3, 5, 6, 8, 11], size=n_alunos)
    acumuladas = np.clip(consecutivas + rng.integers(0, 6, size=n_alunos), 0, dias_letivos)
    st.session_state[CHAVE] = pd.DataFrame({
        "Aluno": [f"Aluno {i + 1}" for i in range(n_alunos)],
        "Faltas consecutivas atuais": consecutivas.astype(int),
        "Faltas acumuladas no período": acumuladas.astype(int),
    })

turma = st.data_editor(
    st.session_state[CHAVE], hide_index=True, num_rows="dynamic", key="sim_editor",
    column_config={
        "Faltas consecutivas atuais": st.column_config.NumberColumn(min_value=0, max_value=dias_letivos),
        "Faltas acumuladas no período": st.column_config.NumberColumn(min_value=0, max_value=dias_letivos),
    })
st.session_state[CHAVE] = turma  # preserva as edições entre reruns (ex.: ao trocar de escola no contexto acima)

# calcula o percentual de faltas e a situação de cada aluno pelas regras do CEU
turma = turma.copy()
turma["% de faltas no período"] = (turma["Faltas acumuladas no período"] / dias_letivos * 100).round(1)
turma["Situação"] = [situacao_ceu(c, p) for c, p in
                     zip(turma["Faltas consecutivas atuais"], turma["% de faltas no período"])]

st.subheader("Resultado da simulação")
# resumo: quantos alunos em cada situação
contagem = turma["Situação"].value_counts()
colunas = st.columns(4)
for coluna, rotulo in zip(colunas, SITUACAO_CEU.values()):
    coluna.metric(rotulo, fmt_i(contagem.get(rotulo, 0)))

st.dataframe(turma.style.format(fmt, subset=["% de faltas no período"]), hide_index=True, height=420)
st.caption("Regras: 5 dias consecutivos → contato com a família; 20% de faltas acumuladas → Busca Ativa Escolar; "
          "10 dias consecutivos → abandono oficial (perda de vaga). Quando mais de uma regra vale, prevalece a mais grave.")
