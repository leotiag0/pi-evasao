"""Feedback dos avaliadores (base dos testes de usabilidade da Etapa 5).

O formulário grava a resposta, com a versão dos dados e do modelo avaliados, na planilha Google da equipe
(``nucleo.salvar_feedback``); sem planilha configurada, em ``dados_feedback/feedback.csv``. A área da equipe (senha) exporta tudo.
"""

import streamlit as st

from nucleo import (admin_configurado, carregar_meta, ler_feedback, planilha_configurada, salvar_feedback,
                    senha_admin_confere)

# metadados do modelo: entram no registro para saber qual versão foi avaliada
meta = carregar_meta()

st.title("Feedback sobre o protótipo")
st.write("Sua opinião orienta o que vamos melhorar. Leva 2 minutos. Sugestão de tarefas para testar antes de responder:")
st.markdown(
    "1. Ache as escolas de **alto risco** da sua diretoria de ensino (ou do seu município).\n"
    "2. Informe quantas escolas a sua equipe consegue acompanhar e veja quantas delas devem estar de fato em risco.\n"
    "3. Escolha uma escola e explique, com suas palavras, **por que** ela está nessa faixa.\n"
    "4. Baixe a lista filtrada e diga se ela ajudaria a planejar contatos e visitas.")

# formulário de usabilidade (os campos são limpos depois de enviar)
with st.form("form_feedback", clear_on_submit=True):
    perfil = st.selectbox("Seu perfil", ["Gestor(a) escolar", "Orientador(a) pedagógico(a)", "Professor(a)", "Outro"],
                          key="fb_perfil")
    tarefa = st.radio("Conseguiu concluir as tarefas sugeridas?", ["Sim", "Em parte", "Não"], horizontal=True, key="fb_tarefa")
    clareza = st.slider("As informações foram claras? (1 = nada claras, 5 = muito claras)", 1, 5, 3, key="fb_clareza")
    utilidade = st.slider("Isto seria útil no seu trabalho? (1 = nada útil, 5 = muito útil)", 1, 5, 3, key="fb_utilidade")
    confianca = st.slider("O quanto você confiaria neste alerta para priorizar um contato? (1 = nada, 5 = totalmente)",
                          1, 5, 3, key="fb_confianca")
    comentario = st.text_area("O que faltou, confundiu ou você mudaria?", key="fb_comentario")
    st.caption("Por favor, não escreva nomes de pessoas nem dados pessoais.")
    enviado = st.form_submit_button("Enviar feedback")

# grava a resposta junto com a versão dos dados e do modelo avaliados
if enviado:
    salvar_feedback({"perfil": perfil, "tarefa_concluida": tarefa, "clareza": clareza, "utilidade": utilidade,
                     "confianca": confianca, "comentario": comentario.strip(), "modelo": meta.get("modelo"),
                     "fonte_dados": meta.get("fonte_dados"), "versao_dados": meta.get("hash_predicoes")})
    st.success("Obrigado! Seu feedback foi registrado.")

# exportação das respostas, protegida pela senha da equipe
with st.expander("Área da equipe"):
    if not admin_configurado():
        st.info("Desativada: defina `senha_admin` em `.streamlit/secrets.toml` (ou `PI_EVASAO_SENHA_ADMIN`) para exportar as respostas.")
    elif senha_admin_confere(st.text_input("Senha da equipe", type="password", key="senha_admin_digitada")):
        respostas = ler_feedback()
        st.write(f"{len(respostas)} resposta(s) registrada(s).")
        st.dataframe(respostas, hide_index=True)
        if not respostas.empty:
            st.download_button("Baixar respostas (CSV)", respostas.to_csv(index=False).encode("utf-8-sig"),
                               "feedback.csv", "text/csv", key="baixar_feedback")
        if not planilha_configurada():
            st.warning("Planilha não configurada: as respostas estão só no disco do servidor e somem quando o app reinicia.")
