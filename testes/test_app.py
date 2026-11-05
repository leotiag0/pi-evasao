"""Testes do dashboard (Etapa 4): cada página roda sem erro sobre os dados de um pipeline sintético.

Usa o AppTest do Streamlit (executa as páginas sem navegador). Os arquivos de ``app/dados`` vêm da fixture
``executado`` (pipeline sintético em diretório temporário), então nada depende dos dados reais.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from nucleo import SITUACAO_CEU, ler_feedback, salvar_feedback, situacao_ceu  # noqa: E402


@pytest.fixture()
def ambiente(executado, monkeypatch, tmp_path):
    """Aponta o app para os dados do pipeline de teste e limpa as senhas do ambiente."""
    c, _ = executado
    monkeypatch.setenv("PI_EVASAO_APP_DADOS", str(c.raiz / "app" / "dados"))
    monkeypatch.setenv("PI_EVASAO_APP_FEEDBACK", str(tmp_path / "feedback"))
    for var in ("PI_EVASAO_SENHA", "PI_EVASAO_SENHA_ADMIN", "PI_EVASAO_FEEDBACK_URL", "PI_EVASAO_FEEDBACK_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.syspath_prepend(str(APP))  # as páginas importam `nucleo`
    return {"pred": pd.read_csv(c.raiz / "app" / "dados" / "predicoes.csv.gz"), "feedback": tmp_path / "feedback" / "feedback.csv"}


def _pagina(nome: str) -> AppTest:
    """Executa uma página do Streamlit no modo de teste."""
    return AppTest.from_file(str(APP / "paginas" / nome), default_timeout=60).run()


def _n(valor: int) -> str:
    """Inteiro formatado com ponto de milhar, como o app exibe."""
    return f"{valor:,}".replace(",", ".")


def test_alerta_mostra_o_total_e_as_tres_faixas(ambiente):
    """O Alerta mostra o total de escolas e uma métrica por faixa."""
    at = _pagina("alerta.py")
    assert not at.exception
    pred = ambiente["pred"]
    assert at.metric[0].label == "Escolas exibidas" and at.metric[0].value == _n(len(pred))
    rotulos = ("▲ Prioridade alta", "■ Prioridade média", "● Prioridade baixa")   # forma + texto: a cor nunca aparece sozinha
    for metrica, rotulo, faixa in zip(at.metric[1:4], rotulos, ("Alto", "Médio", "Baixo")):
        assert metrica.label == rotulo
        assert metrica.value == _n(int((pred["faixa"] == faixa).sum()))
    assert any("DADOS SINTÉTICOS" in e.value for e in at.error)      # o aviso de dados fabricados nunca some


def test_alerta_filtra_por_faixa(ambiente):
    """Escolher uma faixa reduz a lista às escolas dela."""
    at = _pagina("alerta.py")
    at.multiselect(key="filtro_faixa").set_value(["Alto"]).run()
    assert not at.exception
    pred = ambiente["pred"]
    assert at.metric[0].value == _n(int((pred["faixa"] == "Alto").sum()))
    assert at.metric[2].value == "0" and at.metric[3].value == "0"   # Médio e Baixo zerados pelo filtro


def test_alerta_sem_resultado_avisa_em_vez_de_quebrar(ambiente):
    """Filtro sem nenhuma escola mostra um aviso, não um erro."""
    at = _pagina("alerta.py")
    at.text_input(key="filtro_busca").set_value("zzz_nao_existe").run()
    assert not at.exception
    assert any("Nenhuma escola" in i.value for i in at.info)


def test_alerta_capacidade_estima_o_risco_entre_as_n_primeiras(ambiente):
    """A estimativa para as N primeiras escolas aparece e muda com N."""
    at = _pagina("alerta.py")
    at.number_input(key="capacidade").set_value(10).run()
    assert not at.exception
    rotulos = [m.label for m in at.metric]
    assert "Em alto risco, entre as 10 primeiras" in rotulos and "De cada 10 escolas em risco neste filtro, a lista alcança" in rotulos


def test_risco_esperado_usa_a_faixa_do_percentil():
    """A taxa esperada de cada escola vem da faixa do seu percentil no ranking."""
    from nucleo import risco_esperado
    r = risco_esperado(pd.Series([0.5, 49.9, 50.0, 100.0]), [0.1, 0.9])
    assert list(r) == [0.1, 0.1, 0.9, 0.9]   # percentil 100 cai na última faixa, não fora dela


def test_alerta_sem_diretorias_esconde_o_filtro(ambiente):
    """O pipeline sintético não baixa as diretorias (ADR 0010): o filtro some em vez de ficar vazio."""
    at = _pagina("alerta.py")
    assert not any(m.key == "filtro_diretoria" for m in at.multiselect)


def test_alerta_e_painel_filtram_por_diretoria(ambiente, monkeypatch, tmp_path):
    """Alerta e Painel filtram por diretoria; usa uma pasta à parte porque o cache do app é por caminho de pasta."""
    import shutil

    nova_pasta = tmp_path / "dados_com_diretoria"
    shutil.copytree(Path(os.environ["PI_EVASAO_APP_DADOS"]), nova_pasta)
    pred = ambiente["pred"].assign(diretoria=lambda d: ["DE Norte" if i % 3 == 0 else "DRE Sul" for i in d.index])
    pred.to_csv(nova_pasta / "predicoes.csv.gz", index=False)
    monkeypatch.setenv("PI_EVASAO_APP_DADOS", str(nova_pasta))

    at = _pagina("alerta.py")
    at.multiselect(key="filtro_diretoria").set_value(["DE Norte"]).run()
    assert not at.exception
    assert at.metric[0].value == _n(int((pred["diretoria"] == "DE Norte").sum()))

    at = _pagina("painel.py")
    at.multiselect(key="painel_diretoria").set_value(["DRE Sul"]).run()
    assert not at.exception


def test_alerta_detalhe_da_escola_tem_grafico_e_tabela_de_indicadores(ambiente):
    """O detalhe da escola traz a tabela de indicadores e o gráfico anual."""
    at = _pagina("alerta.py")
    assert not at.exception and at.selectbox(key="escola_detalhe").value is not None
    assert len(at.dataframe) >= 2          # a lista de escolas e a tabela de indicadores da escola


def test_alerta_detalhe_ignora_o_filtro_de_faixa(ambiente):
    """Com só uma faixa no filtro, o detalhe ainda lista escolas das outras faixas (para comparar)."""
    at = _pagina("alerta.py")
    at.multiselect(key="filtro_faixa").set_value(["Alto"]).run()
    assert not at.exception
    assert len(at.selectbox(key="escola_detalhe").options) == len(ambiente["pred"])


def test_pdf_tabela_aceita_acentos_e_simbolos():
    """O PDF sai válido mesmo com símbolos fora do latin-1 (faixas, travessão)."""
    from nucleo import pdf_tabela
    tabela = pd.DataFrame({"Faixa": ["▲ Prioridade alta", "● Prioridade baixa"], "Escola": ["João — Ribeirão", "Ação"]})
    pdf = pdf_tabela("Título com acentuação", ["Filtros: nenhum ≈ teste"], tabela)
    assert pdf.startswith(b"%PDF") and len(pdf) > 500


def test_socioeducativa_e_explicacao_da_escola():
    """Unidades da Fundação CASA são reconhecidas pelo nome; a explicação cita o indicador que se destaca."""
    from nucleo import e_socioeducativa, explicacao_escola
    nomes = pd.Series(["CENTRO DE ATEND SOCIOEDUC AO ADOLESC", "JULIA DELLA CASA PROFESSORA", None])
    assert list(e_socioeducativa(nomes)) == [True, False, False]
    pred = pd.DataFrame({"taxa_abandono_ef": [0.0, 1.0, 2.0, 9.0], "tdi_ef": [10.0] * 4,
                         "delta_taxa_abandono_ef": [0.0, 0.0, 0.0, float("nan")], "atu_ef": [25.0] * 4})
    texto = explicacao_escola(pred.iloc[3], pred, 2025)
    assert "abandono de 9,0% em 2025" in texto and "distorção" not in texto and "pico isolado" in texto


def test_sobre_explica_faixas_e_limitacoes(ambiente):
    """A página Sobre explica as faixas e lista as limitações."""
    at = _pagina("sobre.py")
    assert not at.exception
    titulos = [h.value for h in at.header]
    assert {"Como ler as faixas", "O quanto o modelo acerta", "Limitações"} <= set(titulos)


def test_feedback_grava_a_resposta(ambiente):
    """Enviar o formulário grava uma linha no CSV de feedback."""
    at = _pagina("feedback.py")
    at.selectbox(key="fb_perfil").select("Professor(a)")
    at.text_area(key="fb_comentario").set_value("Achei claro.")
    at.button[0].click().run()          # botão de envio do formulário
    assert not at.exception and any("registrado" in s.value for s in at.success)
    linha = pd.read_csv(ambiente["feedback"]).iloc[0]
    assert linha["perfil"] == "Professor(a)" and linha["comentario"] == "Achei claro." and linha["fonte_dados"] == "sintetico"


def test_feedback_vai_para_a_planilha_e_cai_no_csv_se_ela_falhar(ambiente, monkeypatch):
    """Com a planilha configurada, a resposta vai para ela; se ela recusar, a resposta não se perde (vai ao CSV)."""
    linhas = []

    class AppsScriptFalso(BaseHTTPRequestHandler):
        """Imita o Apps Script: o POST responde 302 e o resultado vem no GET seguinte."""
        def do_POST(self):
            pedido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if pedido["token"] != "tk":
                resultado = {"erro": "token inválido"}
            elif pedido["acao"] == "gravar":
                if not linhas:                                          # 1ª resposta cria o cabeçalho
                    linhas.append(list(pedido["registro"]))
                linhas.append(list(pedido["registro"].values()))
                resultado = {"ok": True}
            else:
                resultado = {"linhas": linhas}
            self.server.resultado = resultado
            self.send_response(302)
            self.send_header("Location", "/resultado")
            self.end_headers()

        def do_GET(self):
            corpo = json.dumps(self.server.resultado).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def log_message(self, *args):
            pass

    servidor = HTTPServer(("127.0.0.1", 0), AppsScriptFalso)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    monkeypatch.setenv("PI_EVASAO_FEEDBACK_URL", f"http://127.0.0.1:{servidor.server_port}/exec")
    monkeypatch.setenv("PI_EVASAO_FEEDBACK_TOKEN", "tk")
    try:
        assert salvar_feedback({"perfil": "Professor(a)", "clareza": 4}) == "planilha"
        assert not ambiente["feedback"].exists()                      # nada no disco efêmero
        monkeypatch.setenv("PI_EVASAO_FEEDBACK_TOKEN", "errado")
        assert salvar_feedback({"perfil": "Outro", "clareza": 2}) == "csv"   # planilha recusou: CSV guarda
        monkeypatch.setenv("PI_EVASAO_FEEDBACK_TOKEN", "tk")
        assert list(ler_feedback()["perfil"]) == ["Professor(a)", "Outro"]  # a equipe vê as duas
    finally:
        servidor.shutdown()


def test_senha_bloqueia_ate_ser_informada(ambiente, monkeypatch):
    """Com senha configurada, o app só abre depois de informá-la."""
    monkeypatch.setenv("PI_EVASAO_SENHA", "segredo")
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception and len(at.metric) == 0               # nada do painel aparece sem senha
    at.text_input(key="senha_acesso").set_value("errada").run()
    assert any("incorreta" in e.value for e in at.error) and len(at.metric) == 0
    at.text_input(key="senha_acesso").set_value("segredo").run()
    assert not at.exception and len(at.metric) > 0                # com a senha certa, o painel aparece


# ---- Módulo 3: regras de negócio do CEU (lógica pura, sem Streamlit) -----------------------------
def test_situacao_ceu_aplica_a_regra_mais_grave():
    """A situação do aluno segue o gatilho mais grave entre os três."""
    assert situacao_ceu(0, 0) == SITUACAO_CEU["sem_alerta"]
    assert situacao_ceu(4, 19) == SITUACAO_CEU["sem_alerta"]
    assert situacao_ceu(5, 0) == SITUACAO_CEU["contato_familia"]
    assert situacao_ceu(4, 20) == SITUACAO_CEU["busca_ativa"]        # 20% vale mesmo com poucas faltas seguidas
    assert situacao_ceu(10, 0) == SITUACAO_CEU["abandono_oficial"]   # a mais grave, mesmo sem atingir 20%
    assert situacao_ceu(10, 25) == SITUACAO_CEU["abandono_oficial"]  # continua a mais grave


# ---- Módulo 1: Painel geral ------------------------------------------------------------------------
def test_painel_mostra_evolucao_e_ranking(ambiente):
    """O Painel geral mostra a evolução por rede e o ranking de municípios."""
    at = _pagina("painel.py")
    assert not at.exception
    pred = ambiente["pred"]
    assert at.metric[0].value == str(pred["id_escola"].nunique())
    assert len(at.get("plotly_chart")) == 2   # evolução por rede + ranking de municípios
    assert len(at.dataframe) >= 1


def test_painel_filtra_por_municipio(ambiente):
    """O filtro de município restringe as escolas contadas."""
    at = _pagina("painel.py")
    municipio = ambiente["pred"]["nome_municipio"].iloc[0]
    at.multiselect(key="painel_municipio").set_value([municipio]).run()
    assert not at.exception
    esperado = ambiente["pred"][ambiente["pred"]["nome_municipio"] == municipio]["id_escola"].nunique()
    assert at.metric[0].value == str(esperado)


def test_painel_sem_dados_no_filtro_nao_quebra(ambiente):
    """Uma escola cruzada com o município de OUTRA escola: filtro válido, resultado vazio."""
    at = _pagina("painel.py")
    escola = at.multiselect(key="painel_escola").options[0]
    outro_municipio = next(m for m in at.multiselect(key="painel_municipio").options if m not in escola)
    at.multiselect(key="painel_escola").set_value([escola]).run()
    at.multiselect(key="painel_municipio").set_value([outro_municipio]).run()
    assert not at.exception
    assert any("Nenhum dado" in i.value for i in at.info)


# ---- Módulo 3: página do simulador ------------------------------------------------------------------
def test_simulador_turma_padrao_soma_o_total_de_alunos(ambiente):
    """A turma fictícia padrão soma o total de alunos informado."""
    at = _pagina("simulador.py")
    assert not at.exception
    assert sum(int(m.value) for m in at.metric) == at.slider(key="sim_n").value == 25


def test_simulador_gerar_nova_turma_mantem_o_total(ambiente):
    """Gerar outra turma mantém o total de alunos."""
    at = _pagina("simulador.py")
    at.button(key="sim_gerar").click().run()
    assert not at.exception
    assert sum(int(m.value) for m in at.metric) == 25


def test_simulador_contexto_da_escola_mostra_a_faixa(ambiente):
    """Escolher uma escola no simulador mostra a faixa real dela."""
    at = _pagina("simulador.py")
    escola = ambiente["pred"]["id_escola"].iloc[0]
    at.selectbox(key="sim_escola").set_value(escola).run()
    assert not at.exception
    assert any("alerta precoce" in i.value.lower() for i in at.info)
