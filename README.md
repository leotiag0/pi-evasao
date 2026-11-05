# Alerta precoce de evasão escolar

Aplicativo web do Projeto Integrador em Computação IV (PJI410) da Univesp, 2º semestre de 2026.

Estima, para as escolas públicas de ensino fundamental do Estado de São Paulo, quais tendem a ficar entre as 20% com maior
taxa de abandono **no ano seguinte**, para que gestores e orientadores pedagógicos priorizem apoio antes que os gatilhos de
faltas do diário de classe sejam atingidos.

**Aplicativo publicado:** https://pi-evasao-alerta.streamlit.app/

## O que o aplicativo mostra

| Página | O que faz |
|---|---|
| Alerta precoce | Escolas em três faixas de prioridade (alta, média e baixa), com filtros por rede, diretoria de ensino e município, estimativa de acerto para as N primeiras escolas e exportação da lista em CSV e PDF |
| Painel geral | Evolução da taxa de abandono de 2016 a 2025 por rede, diretoria e município |
| Simulador CEU | Aplica os gatilhos de 5 dias consecutivos, 20% de faltas e 10 dias consecutivos a uma turma fictícia |
| Sobre os dados e limitações | Como ler as faixas, desempenho do modelo na validação e o que não concluir do painel |
| Feedback | Formulário de avaliação do protótipo |

A estimativa é feita **por escola**, com indicadores públicos agregados do Inep. O aplicativo não usa dados de alunos e não
exibe a probabilidade bruta do modelo.

## Dados e modelo

- Fonte: Indicadores Educacionais do Inep por escola (taxas de rendimento, distorção idade-série e média de alunos por turma),
  2016 a 2025, e cadastro de escolas da SEDUC-SP (diretoria de ensino).
- Modelo: *gradient boosting* (scikit-learn) treinado com os indicadores de um ano para prever o ano seguinte. No teste com
  2024 para prever 2025, encontrou 90,3% das escolas que tiveram abandono elevado (PR-AUC de 0,677), contra 80,9% da regra de
  ordenar as escolas pelo abandono do ano anterior.
- O aplicativo lê apenas os arquivos já gerados em `app/dados/` (previsões, histórico e metadados do modelo); não treina
  nada nem acessa banco de dados.

## Estrutura

```
app/            aplicativo Streamlit (publicado no Streamlit Community Cloud)
src/pi_evasao/  pipeline: ingestão do Inep, limpeza, banco SQLite, atributos, treino, avaliação e previsão
testes/         testes automatizados (pytest), incluindo um teste contra vazamento de dados
config/         parâmetros do estudo (recorte, semente, percentil, caminhos)
relatorios/     métricas e figuras geradas pela avaliação do modelo
```

## Refazer o pipeline (opcional)

```bash
python -m venv .venv
pip install -e .
python -m pi_evasao.cli pipeline --fonte inep   # 1ª execução baixa ~1,4 GB do Inep (~5 min)
python -m pi_evasao.cli prever                   # atualiza app/dados/
python -m pytest testes                          # testes automatizados
```

O modo `--fonte sintetico` gera dados fabricados apenas para os testes; nenhum resultado vem dele.

## Executar localmente

Requer Python 3.11 ou superior.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

O aplicativo abre em http://localhost:8501. Sem a configuração `senha_app` nos *secrets*, ele abre sem senha.

## Equipe

Cristian Lopes dos Santos Barbosa Germano, Diego Riciardi, Leonardo Tiago de Oliveira, Queila Rosana Teobaldo,
Renata Assunção Portela de Oliveira e Wong Siu Ki.
