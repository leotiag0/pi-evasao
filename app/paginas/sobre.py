"""Sobre os dados e limitações: o que o painel significa, o quanto ele acerta e o que NÃO deve ser concluído dele."""

import streamlit as st

from nucleo import FAIXAS, aviso_de_fonte, carregar_meta, fmt, fmt_i, rotulo_faixa, taxa_geral_de_alto_risco

# metadados gerados por ``prever``: desempenho, faixas e anos de referência
meta = carregar_meta()
faixas = meta["faixas"]
validacao = faixas["validacao"]
base = taxa_geral_de_alto_risco(validacao)
ano_ref, ano_alvo = int(meta["ano_ref"]), int(meta["ano_alvo"])
teste_ref = meta.get("teste_ano_ref")
periodo_teste = f"{teste_ref}→{teste_ref + 1}" if teste_ref else "o ano mais recente com resultado"
FONTES = {"inep": "planilhas oficiais do INEP por escola (rendimento, distorção idade-série e alunos por turma)",
          "sintetico": "DADOS SINTÉTICOS (fabricados para teste)"}
MODELOS = {"gradient_boosting": "Gradient Boosting"}

st.title("Sobre os dados e limitações")
aviso_de_fonte(meta)

st.header("O que este painel faz")
st.write(f"Estima, para cada escola estadual e municipal de São Paulo, o risco de ela estar entre as **20% com maior taxa de "
         f"abandono em {ano_alvo}**, a partir dos indicadores de {ano_ref}. A ideia é **antecipar** a atenção da equipe "
         "pedagógica: hoje os alertas só disparam depois das faltas (5 dias, 20% ou 10 dias).")

st.header("Como ler as faixas")
st.write("Chamamos de **alto risco** (real) a escola que ficou entre as 20% com maior abandono no ano seguinte. "
         "As faixas abaixo dizem o quanto o modelo acertou em dados passados:")
st.dataframe([
    {"Faixa": rotulo_faixa(f), "O que significa": desc, "Escolas na validação": fmt_i(validacao[f]["n"]),
     "Ficaram em alto risco no ano seguinte": "—" if validacao[f]["taxa_alto_risco"] is None else f"{fmt(100 * validacao[f]['taxa_alto_risco'], 1)}%"}
    for f, desc in zip(FAIXAS, (
        f"Prioridade alta: é o menor corte em que pelo menos {fmt(100 * faixas['precisao_alvo'], 0)}% das escolas acima dele ficaram de fato em alto risco.",
        "O modelo alertaria, mas com menos certeza.",
        "O modelo não alertaria a escola."))
], hide_index=True)
st.caption(f"Média geral: {fmt(100 * base, 1)}% das escolas ficam em alto risco no ano seguinte. Os números vêm de "
           "previsões feitas sobre escolas que o modelo não tinha visto (validação cruzada agrupada por escola).")
st.info("Por decisão de projeto, o painel **não mostra a probabilidade** do modelo: ela seria enganosa (o modelo é ajustado para "
        "não deixar escolas em risco sem alerta, o que infla os valores). Por isso mostramos faixa, posição e taxa observada.")

st.header("O quanto o modelo acerta")
# desempenho no teste fora do tempo, em linguagem simples
teste = meta.get("metricas_avaliacao", {}).get("fora_do_tempo")
if teste:
    alvo = teste["tp"] + teste["fp"]
    total = alvo + teste["tn"] + teste["fn"]
    st.write(f"No teste mais realista (treinar com anos anteriores e prever **{periodo_teste}**, dados que o modelo não viu):")
    st.markdown(
        f"- o modelo alertou **{fmt(100 * alvo / total, 0)}%** das escolas;\n"
        f"- encontrou **{fmt(100 * teste['recall'], 0)}%** das que de fato ficaram em alto risco;\n"
        f"- **{fmt(100 * teste['fp'] / alvo, 0)}%** dos alertas foram falsos (a escola não ficou em alto risco).")
    st.caption("O modelo prioriza não deixar escolas em risco sem alerta, ao custo de alguns alertas a mais. "
               "Descrição técnica completa no relatório do projeto.")
else:
    st.caption("Métricas de avaliação não disponíveis neste conjunto de dados.")

st.header("Dados e modelo")
st.markdown(
    f"- **Fonte dos dados:** {FONTES.get(meta.get('fonte_dados'), meta.get('fonte_dados'))}\n"
    f"- **Base de cálculo:** abandono, distorção idade-série e alunos por turma de {ano_ref} (e a variação de cada um desde o ano anterior); **ano previsto:** {ano_alvo}\n"
    f"- **Escolas no painel:** {fmt_i(meta['n_escolas'])} (ensino fundamental, redes estadual e municipal)\n"
    f"- **Modelo:** {MODELOS.get(meta['modelo'], meta['modelo'])}, treinado com {fmt_i(meta['n_pares'])} pares escola-ano\n"
    f"- **Gerado em:** {meta['gerado_em'].replace('T', ' ')} (versão dos dados: `{meta['hash_predicoes']}`)")

st.header("Limitações")
st.markdown(
    "- **É por escola, não por aluno.** Os dados públicos são agregados; o painel não diz quais alunos vão evadir.\n"
    "- **Risco é relativo:** \"alto risco\" significa estar entre os 20% piores do ano, não superar um patamar fixo.\n"
    "- **Há muitos alertas falsos.** Use a faixa para priorizar a atenção, não como diagnóstico.\n"
    f"- **Testado em um único ano** ({periodo_teste}); o desempenho pode variar em outros anos.\n"
    "- **Não foi validado em campo.** Este protótipo existe justamente para receber a opinião de quem conhece as escolas.\n"
    "- **Importância não é causa.** Abandono anterior e distorção idade-série pesam mais, mas isso mostra onde o risco se "
    "concentra, não o que o provoca.\n"
    "- **Escola sem abandono informado no ano-base** aparece com aviso e tem previsão menos confiável.")

st.header("Uso responsável")
st.write("A lista nomeia escolas reais. Ela deve circular apenas entre quem atua no acompanhamento pedagógico, "
         "e não deve ser usada para ranquear ou expor escolas, professores ou comunidades.")
