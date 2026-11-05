"""Avaliação do modelo, tabela comparativa e figuras (Etapa 3 do PI).

Papel no pipeline: transforma o que ``treinar`` guardou (probabilidades no teste, limiares) em evidências
para o relatório: ``relatorios/metricas/comparativo.{csv,md}``, duas figuras em ``relatorios/figuras/``
(curva precisão-recall e importância das variáveis; PNG, 300 dpi, rótulos em pt-BR) e as legendas em ``figuras.md``.

Métricas (ADR 0004): acurácia, precisão, recall, F1, F2, ROC-AUC, PR-AUC, Brier e matriz de confusão. PR-AUC é a
métrica informativa sob desbalanceamento; o baseline ``Dummy`` define o piso (PR-AUC do acaso = prevalência) e a
persistência (ADR 0009), o piso exigente. A importância das variáveis é por PERMUTAÇÃO no teste (queda de PR-AUC
ao embaralhar cada coluna).

Estilo das figuras: grade de 1px recessiva, texto em tons de tinta (nunca na cor da série), ponto de operação com
anel de 2px. A tabela ``comparativo.md`` é a visão alternativa dos mesmos números.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from ..config import Config  # noqa: E402
from ..formato import i as _i, n as _n  # noqa: E402
from . import persistencia  # noqa: E402
from .treino import MODELO_FINAL, REGIME_TEMPORAL, eh_baseline  # noqa: E402

SUPERFICIE, TINTA, TINTA2, MUDO, GRADE, EIXO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AZUL, VERDE = "#2a78d6", "#1baf7a"
COR = {MODELO_FINAL: VERDE, "baseline_dummy": MUDO, "baseline_persistencia": TINTA2}
ROTULO = {MODELO_FINAL: "Gradient Boosting", "baseline_dummy": "Baseline (alerta todas)",
          "baseline_persistencia": "Baseline (abandono do ano t)"}


def _tick(v, _pos=None) -> str:
    """Rótulo de eixo em pt-BR sem zeros à direita (0,2 / 0,015 / 1)."""
    s = f"{v:.4f}".rstrip("0").rstrip(".").replace(".", ",")
    return "0" if s in ("", "-0") else s


def calcular_metricas(y, proba, limiar: float) -> dict:
    """Métricas de classificação no limiar escolhido, mais a matriz de confusão."""
    y = np.asarray(y)
    # classe prevista pelo limiar de decisão; as métricas de ranking (AUC) usam a probabilidade
    pred = (proba >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "limiar": float(limiar),
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": fbeta_score(y, pred, beta=1, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def taxa_por_percentil(y, proba, n_faixas: int = 50) -> list[float]:
    """Taxa observada de alto risco em cada faixa de ``100/n_faixas``% do ranking (índice 0 = escolas de
    MENOR escore). O dashboard usa isto para estimar quantas das N primeiras escolas de uma lista estarão
    de fato em alto risco (precisão entre as N primeiras), valendo também para listas filtradas.

    50 faixas: no teste real (~10 mil escolas) são ~200 escolas por faixa, o que distingue o topo
    sem o ruído de faixas de ~100 escolas; com 20 faixas, qualquer N abaixo de ~500 escolas dava a mesma taxa."""
    pct = pd.Series(proba).rank(pct=True).to_numpy()
    faixa = np.minimum((pct * n_faixas).astype(int), n_faixas - 1)
    y = np.asarray(y)
    return [float(y[faixa == f].mean()) for f in range(n_faixas)]


def ic_bootstrap_pr_auc(y, proba, n: int = 300, seed: int = 0) -> tuple[float, float]:
    """IC 95% percentílico da PR-AUC por bootstrap das linhas do teste (um par por escola no fora-do-tempo).

    Serve para dizer se a diferença entre modelos é maior que o ruído amostral."""
    y, proba = np.asarray(y), np.asarray(proba)
    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            valores.append(average_precision_score(y[i], proba[i]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


# ---------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------
def _novo_eixo(titulo: str, subtitulo: str, figsize=(6.4, 4.7)):
    """Cria uma figura com o estilo padrão do projeto (título, subtítulo, grade leve)."""
    # estilo comum a todas as figuras
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": EIXO, "axes.labelcolor": TINTA2, "xtick.color": MUDO, "ytick.color": MUDO,
        "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    })
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=0.82, left=0.12, right=0.96, bottom=0.13)
    fig.text(0.04, 0.955, titulo, fontsize=11.5, fontweight="bold", color=TINTA, va="top")
    fig.text(0.04, 0.90, subtitulo, fontsize=8.5, color=TINTA2, va="top")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(True, color=GRADE, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_tick))
    return fig, ax


def _salvar(fig, caminho) -> None:
    """Salva a figura em PNG a 300 dpi e a fecha."""
    fig.savefig(caminho, dpi=300)
    plt.close(fig)


def _fig_pr(modelos, y, prevalencia, caminho, subtitulo):
    """Curva precisão-recall do modelo e da persistência, com o ponto de operação do limiar."""
    fig, ax = _novo_eixo("Curva precisão-recall - fora do tempo", subtitulo, (6.4, 4.9))
    ax.axhline(prevalencia, color=MUDO, linewidth=1.2, linestyle="--",
               label=f"Baseline: prevalência ({_n(prevalencia * 100, 1)}%)")
    for nome, r in modelos.items():
        if nome == "baseline_dummy":
            continue
        # uma curva por modelo (o piso da prevalência é a linha tracejada)
        prec, rec, _ = precision_recall_curve(y, r["proba_teste"])
        ax.plot(rec, prec, color=COR[nome], linewidth=2, solid_capstyle="round",
                label=f"{ROTULO[nome]} (PR-AUC {_n(r['metricas']['pr_auc'])})")
        m = r["metricas"]  # ponto de operação no limiar F2, com anel de 2px na cor da superfície
        ax.plot(m["recall"], m["precisao"], marker="o", markersize=8, color=COR[nome],
                markeredgecolor=SUPERFICIE, markeredgewidth=2, linestyle="none")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precisão")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TINTA2)
    ax.text(0.01, 0.02, "Círculos: ponto de operação no limiar escolhido por F2", fontsize=7.5,
            color=MUDO, transform=ax.transAxes)
    _salvar(fig, caminho)


def _fig_importancia(imp: pd.DataFrame, caminho, subtitulo):
    """Barras da importância por permutação, com o desvio das repetições."""
    # barras de baixo para cima: a variável mais importante fica no topo
    topo = imp.iloc[::-1]
    fig, ax = _novo_eixo(f"Importância por permutação - {ROTULO[MODELO_FINAL]}", f"Fora do tempo | {subtitulo}", (6.6, 4.0))
    fig.subplots_adjust(left=0.33, right=0.93)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRADE, linewidth=1)
    y = np.arange(len(topo))
    ax.barh(y, topo["media"], xerr=topo["desvio"], height=0.5, color=AZUL,
            error_kw={"ecolor": MUDO, "elinewidth": 1, "capsize": 0})
    alcance = (topo["media"] + topo["desvio"]).max()
    for yi, v, d in zip(y, topo["media"], topo["desvio"]):  # rótulo depois do traço do desvio, sem cruzá-lo
        ax.text(max(v, 0) + d + alcance * 0.02, yi, _n(v), va="center", fontsize=8, color=TINTA2)
    ax.set_yticks(y, topo["variavel"])
    ax.tick_params(axis="y", length=0, labelsize=8.5, labelcolor=TINTA2)
    ax.set_xlabel("Queda média da PR-AUC ao embaralhar a variável (traço: desvio em 5 repetições)")
    ax.set_xlim(0, alcance * 1.2)
    _salvar(fig, caminho)


# ---------------------------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------------------------
def avaliar_resultados(cfg: Config, res: dict) -> dict:
    """Consome o dict salvo por ``treinar`` e gera tabela, figuras e legendas."""
    fonte = res["fonte"]
    pasta_fig, pasta_met = cfg.caminho("figuras"), cfg.caminho("metricas")
    dados = res["regimes"][REGIME_TEMPORAL]
    info, modelos = dados["info"], dados["modelos"]
    sub = f"n teste = {_i(info['n_teste'])} | prevalência {_n(info['prevalencia_teste'] * 100, 1)}%"

    linhas = []
    for nome, r in modelos.items():
        # métricas no limiar de cada modelo, com IC 95% da PR-AUC por bootstrap
        r["metricas"] = calcular_metricas(r["y_teste"], r["proba_teste"], r["limiar"])
        r["metricas"]["pr_auc_ic_baixo"], r["metricas"]["pr_auc_ic_alto"] = ic_bootstrap_pr_auc(
            r["y_teste"], r["proba_teste"], seed=cfg.seed)
        linhas.append({"modelo": nome, "cv_pr_auc": r["cv_pr_auc"], "n_treino": r["n_treino"],
                       "n_teste": r["n_teste"], **r["metricas"]})
    rb = modelos[MODELO_FINAL]
    y = rb["y_teste"]

    # importância por permutação no teste: queda da PR-AUC ao embaralhar cada coluna
    imp = permutation_importance(rb["pipeline"], rb["X_teste"], y, scoring="average_precision",
                                 n_repeats=5, random_state=cfg.seed, n_jobs=1)
    tab = (pd.DataFrame({"variavel": rb["X_teste"].columns, "media": imp.importances_mean,
                         "desvio": imp.importances_std}).sort_values("media", ascending=False))
    tab.to_csv(pasta_met / "importancia_fora_do_tempo.csv", index=False, encoding="utf-8")
    _fig_pr(modelos, y, info["prevalencia_teste"], pasta_fig / "pr_fora_do_tempo.png", sub)
    _fig_importancia(tab, pasta_fig / "importancia_fora_do_tempo.png", sub)

    df = pd.DataFrame(linhas)
    df.to_csv(pasta_met / "comparativo.csv", index=False, encoding="utf-8")
    (pasta_met / "comparativo.md").write_text(_md_comparativo(df, res, cfg), encoding="utf-8")
    (pasta_fig / "figuras.md").write_text(_md_figuras(res, rb["metricas"], tab, info), encoding="utf-8")

    # as métricas e a curva de precisão por posição vão para os metadados, que o dashboard lê
    persistencia.atualizar_meta(cfg, metricas_avaliacao={REGIME_TEMPORAL: rb["metricas"]}, modelo_escolhido=MODELO_FINAL,
                                taxa_por_percentil=taxa_por_percentil(y, rb["proba_teste"]))
    return {"fonte": fonte, "modelo": MODELO_FINAL,
            "pr_auc_fora_do_tempo": round(rb["metricas"]["pr_auc"], 4), "recall_fora_do_tempo": round(rb["metricas"]["recall"], 4),
            "pr_auc_persistencia": round(modelos["baseline_persistencia"]["metricas"]["pr_auc"], 4),
            "possivel_vazamento": bool(rb["metricas"]["pr_auc"] > 0.95)}


def _md_comparativo(df: pd.DataFrame, res: dict, cfg: Config) -> str:
    """Texto de ``comparativo.md``: tabela, matriz de confusão e veredito contra os pisos."""
    aviso = ("> **AVISO:** dados SINTÉTICOS. Estes números medem a capacidade do modelo de recuperar o "
             "mecanismo embutido no gerador, NÃO o risco de escolas reais. Não citar como resultado empírico.\n\n"
             if res["fonte"] == "sintetico" else "")
    info = res["regimes"][REGIME_TEMPORAL]["info"]
    # ordena pela PR-AUC; os pisos aparecem na mesma tabela
    d = df.sort_values("pr_auc", ascending=False)
    L = ["# Comparativo do modelo com os pisos", "", aviso,
         f"- **Fonte dos dados:** {res['fonte']}  |  **Hash do dataset:** `{res['hash_dataset']}`",
         f"- **Limiar de decisão:** maximiza F{cfg.fbeta:g} em probabilidades out-of-fold do treino (não é 0,5)",
         "- **Baseline (Dummy):** prevê a prevalência para todas as escolas; com limiar por F2 equivale a alertar em todas. É o piso: um modelo só tem valor se o superar.",
         "- **Baseline (abandono do ano t):** ordena as escolas só pela taxa de abandono do ano de referência (persistência; ADR 0009). "
         "É o piso exigente: o que a gestão faria numa planilha, sem modelo. Seu Brier não tem leitura (o escore não é probabilidade).", ""]
    if (df[~df.modelo.map(eh_baseline)].pr_auc > 0.95).any():
        L += ["> **ATENÇÃO:** PR-AUC > 0,95. Em dados reais isso indicaria vazamento; investigar antes de qualquer conclusão.", ""]
    colunas = ["acuracia", "precisao", "recall", "f1", "f2", "roc_auc", "pr_auc"]
    cab = ["Modelo", "Acurácia", "Precisão", "Recall", "F1", "F2", "ROC-AUC", "PR-AUC", "IC 95% da PR-AUC", "Brier", "Limiar"]
    L += ["## Fora do tempo", "", info["descricao"], "",
          f"Treino: {_i(info['n_treino'])} pares | Teste: {_i(info['n_teste'])} pares | Prevalência no teste: "
          f"{_n(info['prevalencia_teste'] * 100, 1)}%", "",
          "| " + " | ".join(cab) + " |", "|---|" + "---:|" * (len(cab) - 1)]
    for _, r in d.iterrows():
        L.append(f"| {ROTULO[r['modelo']]} | " + " | ".join(_n(r[c]) for c in colunas)
                 + f" | {_n(r['pr_auc_ic_baixo'])}-{_n(r['pr_auc_ic_alto'])} | {_n(r['brier'])} | {_n(r['limiar'])} |")
    m = d[d.modelo == MODELO_FINAL].iloc[0]
    base = d[d.modelo == "baseline_dummy"].iloc[0]
    pers = d[d.modelo == "baseline_persistencia"].iloc[0]
    L += ["", "**Matriz de confusão no limiar escolhido** (linhas = real; colunas = previsto):", "",
          "| Modelo | Verdadeiro negativo | Falso positivo | Falso negativo | Verdadeiro positivo |", "|---|---:|---:|---:|---:|"]
    L += [f"| {ROTULO[r['modelo']]} | {_i(r['tn'])} | {_i(r['fp'])} | {_i(r['fn'])} | {_i(r['tp'])} |" for _, r in d.iterrows()]
    # veredito: o modelo supera a persistência se o IC dele fica acima do IC dela
    L += ["", "## O modelo supera os pisos?", "",
          f"Bootstrap (300 reamostragens do teste) sobre a PR-AUC. Contra o acaso: {ROTULO[MODELO_FINAL]} tem PR-AUC {_n(m['pr_auc'])} "
          f"contra {_n(base['pr_auc'])} da prevalência; F2 {_n(m['f2'])} contra {_n(base['f2'])}. Contra a persistência: "
          f"{_n(m['pr_auc'])} (IC {_n(m['pr_auc_ic_baixo'])}-{_n(m['pr_auc_ic_alto'])}) e o abandono do ano t sozinho, "
          f"{_n(pers['pr_auc'])} (IC {_n(pers['pr_auc_ic_baixo'])}-{_n(pers['pr_auc_ic_alto'])}): "
          + ("o modelo SUPERA a persistência além do ruído." if m["pr_auc_ic_baixo"] > pers["pr_auc_ic_alto"]
             else "os ICs se sobrepõem, então o modelo NÃO se distingue de ordenar pelo abandono do ano passado."),
          "", "## Como ler", "",
          "- **Recall** é a métrica prioritária: falso negativo é escola em risco sem alerta (perde alunos); falso positivo é uma visita a mais (ADR 0004).",
          "- **PR-AUC** compara-se à prevalência (PR-AUC do acaso). **Brier** mede a qualidade da probabilidade; como o modelo usa `class_weight` balanceado, "
          "as probabilidades ficam infladas e o Brier tende a ser pior. O dashboard exibe FAIXAS de risco, nunca a probabilidade.",
          "- O regime **fora do tempo** replica o uso real (prever o ano seguinte) e é o número a reportar como desempenho esperado.", ""]
    return "\n".join(L)


def _md_figuras(res: dict, m: dict, tab: pd.DataFrame, info: dict) -> str:
    """Texto de ``figuras.md``: legenda pronta de cada figura."""
    L = ["# Legendas das figuras", "",
         f"Geradas automaticamente. Fonte dos dados: **{res['fonte']}**. Arquivos PNG a 300 dpi neste diretório.", ""]
    if res["fonte"] == "sintetico":
        L += ["> **AVISO:** figuras calculadas sobre dados SINTÉTICOS; devem ser rotuladas como tal no relatório.", ""]
    # uma legenda por figura, com o que mostra e o ponto de operação
    itens = [
        ("pr_fora_do_tempo.png", f"Curva precisão-recall do modelo e da persistência, fora do tempo. A linha tracejada é a prevalência "
                                 f"({_n(info['prevalencia_teste'] * 100, 1)}%); os círculos marcam o ponto de operação no limiar por F2 "
                                 f"(recall {_n(m['recall'])}, precisão {_n(m['precisao'])})."),
        ("importancia_fora_do_tempo.png", f"Importância por permutação do {ROTULO[MODELO_FINAL]}, fora do tempo. Variável mais importante: "
                                          f"`{tab.iloc[0]['variavel']}` (queda de PR-AUC {_n(tab.iloc[0]['media'])})."),
    ]
    for n, (arq, legenda) in enumerate(itens, start=1):
        L += [f"**Figura {n}** - {legenda}", f"Arquivo: `{arq}`", "Fonte: elaborado pelos autores (2026), com dados " + res["fonte"] + ".", ""]
    return "\n".join(L)
