# Relatório de qualidade de dados


- **Fonte:** inep
- **Linhas de entrada:** 112.304
- **Linhas após limpeza:** 105.859 (11.268 escolas)

## Etapas que removem linhas

| Etapa | Antes | Depois | Descartadas | Motivo |
|---|---:|---:|---:|---|
| Tipagem e chaves | 112.304 | 112.304 | 0 | Linhas sem id_escola, ano ou id_municipio numéricos |
| Filtro de escopo | 112.304 | 112.304 | 0 | Fora do recorte: redes=['estadual', 'municipal'], anos=2016-2025 |
| Deduplicação | 112.304 | 112.304 | 0 | Mais de uma linha para o mesmo (id_escola, ano); mantida a primeira |
| Escolas sem alvo | 112.304 | 105.859 | 6.445 | Escola sem NENHUM valor de taxa_abandono_ef em todo o período (removida por inteiro) |

## Valores invalidados (viram NaN; a linha é mantida)

Nenhum valor impossível encontrado.

## Nulos por coluna (% das linhas)

| Coluna | Antes | Depois |
|---|---:|---:|
| `taxa_abandono_ef` | 6,63 | 0,95 |
| `tdi_ef` | 6,59 | 0,90 |
| `atu_ef` | 6,59 | 0,90 |

*Nulos remanescentes não são imputados aqui: a imputação ocorre dentro do Pipeline de modelagem, ajustada só no treino (evita vazamento).*
