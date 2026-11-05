# Comparativo do modelo com os pisos


- **Fonte dos dados:** inep  |  **Hash do dataset:** `656f87f3c6464b7d`
- **Limiar de decisão:** maximiza F2 em probabilidades out-of-fold do treino (não é 0,5)
- **Baseline (Dummy):** prevê a prevalência para todas as escolas; com limiar por F2 equivale a alertar em todas. É o piso: um modelo só tem valor se o superar.
- **Baseline (abandono do ano t):** ordena as escolas só pela taxa de abandono do ano de referência (persistência; ADR 0009). É o piso exigente: o que a gestão faria numa planilha, sem modelo. Seu Brier não tem leitura (o escore não é probabilidade).

## Fora do tempo

Treino nos pares com ano de referência < 2024; teste no par 2024 -> 2025. Replica o uso real (prever o futuro).

Treino: 83.205 pares | Teste: 10.340 pares | Prevalência no teste: 21,2%

| Modelo | Acurácia | Precisão | Recall | F1 | F2 | ROC-AUC | PR-AUC | IC 95% da PR-AUC | Brier | Limiar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0,743 | 0,447 | 0,903 | 0,598 | 0,750 | 0,882 | 0,677 | 0,654-0,698 | 0,134 | 0,365 |
| Baseline (abandono do ano t) | 0,709 | 0,406 | 0,809 | 0,541 | 0,675 | 0,808 | 0,564 | 0,543-0,590 | 0,207 | 0,001 |
| Baseline (alerta todas) | 0,212 | 0,212 | 1,000 | 0,350 | 0,574 | 0,500 | 0,212 | 0,203-0,220 | 0,167 | 0,212 |

**Matriz de confusão no limiar escolhido** (linhas = real; colunas = previsto):

| Modelo | Verdadeiro negativo | Falso positivo | Falso negativo | Verdadeiro positivo |
|---|---:|---:|---:|---:|
| Gradient Boosting | 5.698 | 2.448 | 212 | 1.982 |
| Baseline (abandono do ano t) | 5.553 | 2.593 | 420 | 1.774 |
| Baseline (alerta todas) | 0 | 8.146 | 0 | 2.194 |

## O modelo supera os pisos?

Bootstrap (300 reamostragens do teste) sobre a PR-AUC. Contra o acaso: Gradient Boosting tem PR-AUC 0,677 contra 0,212 da prevalência; F2 0,750 contra 0,574. Contra a persistência: 0,677 (IC 0,654-0,698) e o abandono do ano t sozinho, 0,564 (IC 0,543-0,590): o modelo SUPERA a persistência além do ruído.

## Como ler

- **Recall** é a métrica prioritária: falso negativo é escola em risco sem alerta (perde alunos); falso positivo é uma visita a mais (ADR 0004).
- **PR-AUC** compara-se à prevalência (PR-AUC do acaso). **Brier** mede a qualidade da probabilidade; como o modelo usa `class_weight` balanceado, as probabilidades ficam infladas e o Brier tende a ser pior. O dashboard exibe FAIXAS de risco, nunca a probabilidade.
- O regime **fora do tempo** replica o uso real (prever o ano seguinte) e é o número a reportar como desempenho esperado.
