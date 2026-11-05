"""Entrada do dashboard de alerta precoce de evasão escolar (protótipo, Etapa 4 do PI IV Univesp).

Rodar localmente:  ``streamlit run app/streamlit_app.py``
Para acrescentar uma função: crie ``paginas/<nome>.py`` e liste-a abaixo (uma página = um arquivo).
"""

import streamlit as st

from nucleo import exigir_acesso

# configuração da página e controle de acesso (senha opcional)
st.set_page_config(page_title="Alerta precoce de evasão escolar", layout="wide")
exigir_acesso()

# menu lateral: uma página por arquivo de ``paginas/``
pagina = st.navigation([
    st.Page("paginas/painel.py", title="Painel geral"),
    st.Page("paginas/alerta.py", title="Alerta precoce", default=True),
    st.Page("paginas/simulador.py", title="Simulador CEU"),
    st.Page("paginas/sobre.py", title="Sobre os dados e limitações"),
    st.Page("paginas/feedback.py", title="Feedback"),
])
pagina.run()
