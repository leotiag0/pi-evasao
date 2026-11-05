# Engenharia de atributos


- **Fonte:** inep
- **Pares (escola, t -> t+1):** 93.545 de 11.031 escolas
- **Rótulo:** `alto_risco` = abandono em t+1 >= percentil 80% do próprio ano t+1
- **Prevalência total de alto risco:** 21,2%

## Prevalência e limiar por ano-alvo

| Ano-alvo (t+1) | Limiar de abandono (%) | Prevalência de alto risco |
|---:|---:|---:|
| 2017 | 1,10 | 20,3% |
| 2018 | 0,90 | 21,5% |
| 2019 | 0,60 | 21,9% |
| 2020 | 0,10 | 21,7% |
| 2021 | 0,90 | 21,1% |
| 2022 | 0,80 | 20,2% |
| 2023 | 0,80 | 20,3% |
| 2024 | 0,70 | 22,3% |
| 2025 | 1,10 | 21,2% |

*A prevalência pode superar o percentil nominal por empates: as taxas têm uma casa decimal e muitas escolas repetem o mesmo valor no limiar.*

## Variáveis preditoras (todas do ano t)

| Variável | Tipo | % nulos | Média | Desvio | Mín | Máx | Descrição |
|---|---|---:|---:|---:|---:|---:|---|
| `taxa_abandono_ef` | num | 0,1 | 0,52 | 1,34 | 0,00 | 100,00 | Taxa de abandono no ano t (%). Legítima: conhecida no momento da predição. Domina a importância (persistência escolar). |
| `tdi_ef` | num | 0,1 | 7,78 | 10,37 | 0,00 | 100,00 | Taxa de distorção idade-série no ano t (%). |
| `atu_ef` | num | 0,1 | 25,96 | 6,04 | 1,00 | 58,00 | Média de alunos por turma no ano t. |
| `delta_taxa_abandono_ef` | num | 12,0 | -0,04 | 1,46 | -80,00 | 80,00 | DERIVADA: taxa_abandono(t) - taxa_abandono(t-1). NaN se a escola não tem o ano t-1 (ex.: 2016, primeiro ano do painel). |
| `delta_tdi_ef` | num | 12,0 | -0,45 | 3,38 | -100,00 | 100,00 | DERIVADA: variação da distorção idade-série entre t-1 e t. Mesma regra de NaN. |
| `rede` | cat | 0,0 | - | - | - | - | Rede de ensino (estadual | municipal). Níveis: estadual, municipal. |

## Variáveis deliberadamente excluídas de X

| Variável | Motivo |
|---|---|
| `taxa_aprovacao_ef` | Colinearidade exata: aprovação = 100 - reprovação - abandono. Nem é lida da fonte. |
| `id_municipio` | Alta cardinalidade (centenas de códigos): convida o modelo a memorizar municípios em vez de generalizar. |
| `ano` | Convida a memorizar tendência temporal e quebra a validação out-of-time. |
| `id_escola` | Identificador: memorização garantida, generalização nenhuma. Usado só para agrupar a validação por escola. |
| `taxa_abandono_alvo` | É o próprio alvo do ano t+1 (vazamento direto). |
| `limiar_alvo` | Percentil calculado sobre a distribuição do ano t+1 (vazamento indireto). |

NaN estruturais (ex.: deltas de 2020) são tratados no Pipeline do modelo com indicador de ausência, e não removidos.
