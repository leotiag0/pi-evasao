"""Interface de linha de comando (argparse da biblioteca padrão).

Papel no pipeline: cada subcomando executa UMA fase e é decorado com ``registrar_execucao``,
que grava um registro datado em ``docs/execucoes/`` (requisito de documentação do grupo).

    pi-evasao ingerir   --fonte inep|sintetico
    pi-evasao etl
    pi-evasao features
    pi-evasao treinar
    pi-evasao avaliar
    pi-evasao prever                              # Etapa 4: alimenta o dashboard
    pi-evasao pipeline  --fonte inep              # encadeia tudo

Equivalência com as Etapas do PI: ingerir+etl = Etapa 1; features = Etapa 2; treinar+avaliar = Etapa 3.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import joblib
import pandas as pd

from .config import Config, carregar_config
from .erros import ErroAcionavel
from .etl import banco
from .etl.limpeza import limpar, relatorio_markdown
from .features.engenharia import FEATURES_CAT, FEATURES_NUM, construir_dataset, markdown_features, separar_xy
from .ingestao import obter_fonte
from .ingestao.base import markdown_dicionario
from .ingestao.diretorias import carregar_fontes as carregar_diretorias
from .log import hash_dataframe, obter_logger, registrar_execucao
from .modelagem import persistencia
from .modelagem.avaliacao import avaliar_resultados
from .modelagem.faixas import cortes_faixas, validacao_por_faixa
from .modelagem.predicao import gerar_predicoes
from .modelagem.treino import MODELO_FINAL, MODELOS, REGIME_TEMPORAL, indices_temporal, treinar_final, treinar_regime

ARQ_BRUTO = "painel_bruto.csv"
ARQ_RESULTADO = "resultado_treino.joblib"


def _escrever(caminho: Path, texto: str) -> None:
    """Grava um texto UTF-8, criando as pastas que faltam."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8")


# ---- Etapa 1 ----------------------------------------------------------------------------------
@registrar_execucao("ingerir")
def cmd_ingerir(cfg: Config, fonte: str | None = None, n_escolas: int | None = None) -> dict:
    """Etapa 1: obtém o painel bruto da fonte escolhida e grava o CSV e o dicionário de dados."""
    fonte = fonte or cfg.fonte_padrao
    if n_escolas:
        cfg = dataclasses.replace(cfg, n_escolas_sintetico=n_escolas)
    df = obter_fonte(fonte, cfg).carregar(cfg.anos, cfg.uf, cfg.redes)
    destino = cfg.caminho("bruto") / ARQ_BRUTO
    df.to_csv(destino, index=False, encoding="utf-8")
    _escrever(cfg.caminho("docs") / "dicionario_dados.md", markdown_dicionario(df, fonte))
    return {"fonte": fonte, "linhas": len(df), "escolas": df["id_escola"].nunique(),
            "anos": f"{df['ano'].min()}-{df['ano'].max()}", "arquivo": str(destino.relative_to(cfg.raiz)),
            "hash_dataset": hash_dataframe(df)}


@registrar_execucao("etl")
def cmd_etl(cfg: Config) -> dict:
    """Etapa 1: limpa o painel bruto, grava no SQLite e escreve o relatório de qualidade."""
    arq = cfg.caminho("bruto") / ARQ_BRUTO
    if not arq.exists():
        raise ErroAcionavel("Painel bruto não encontrado. Rode 'pi-evasao ingerir' primeiro.")
    # lê o painel bruto, limpa e guarda o resultado no banco com o relatório de qualidade
    bruto = pd.read_csv(arq)
    fonte = str(bruto["fonte"].iloc[0])
    # a limpeza devolve o painel e um relatório de tudo o que foi alterado
    limpo, rel = limpar(bruto, cfg)
    # o hash identifica a versão exata dos dados usada nas fases seguintes
    h = hash_dataframe(limpo)
    with banco.conectar(cfg.caminho("banco")) as con:
        banco.gravar_painel(con, limpo)
        banco.registrar_execucao(con, fonte, rel["linhas_entrada"], rel["linhas_saida"], rel["escolas"], h)
    _escrever(cfg.caminho("metricas") / "qualidade.md", relatorio_markdown(rel, fonte))
    return {"fonte": fonte, "linhas_entrada": rel["linhas_entrada"], "linhas_saida": rel["linhas_saida"],
            "escolas": rel["escolas"], "valores_invalidados": sum(v["valores"] for v in rel["valores"]),
            "hash_dataset": h}


# ---- Etapa 2 ----------------------------------------------------------------------------------
@registrar_execucao("features")
def cmd_features(cfg: Config) -> dict:
    """Etapa 2: monta o dataset de pares (escola, t -> t+1) com atributos e rótulo."""
    with banco.conectar(cfg.caminho("banco")) as con:
        painel = banco.ler_painel(con)
        if painel.empty:
            raise ErroAcionavel("Banco vazio. Rode 'pi-evasao etl' primeiro.")
        fonte = str(painel["fonte"].iloc[0])
        ds, resumo = construir_dataset(painel, cfg)
        banco.gravar_tabela(con, "dataset_modelagem", ds)
    _escrever(cfg.caminho("metricas") / "features.md", markdown_features(ds, resumo, cfg, fonte))
    return {"fonte": fonte, "linhas_painel": len(painel), "pares_t_t1": resumo["pares"],
            "escolas": resumo["escolas"], "prevalencia_alto_risco": resumo["prevalencia_total"],
            "variaveis": len(FEATURES_NUM) + len(FEATURES_CAT), "hash_dataset": hash_dataframe(ds)}


# ---- Etapa 3 ----------------------------------------------------------------------------------
@registrar_execucao("treinar")
def cmd_treinar(cfg: Config) -> dict:
    """Etapa 3: treina o modelo e os pisos, valida fora do tempo e serializa o modelo final."""
    with banco.conectar(cfg.caminho("banco")) as con:
        ds = banco.ler_tabela(con, "dataset_modelagem")
    if ds.empty:
        raise ErroAcionavel("dataset_modelagem vazio. Rode 'pi-evasao features' primeiro.")
    fonte, h = str(ds["fonte"].iloc[0]), hash_dataframe(ds)
    # X só tem colunas do ano t; o rótulo e os metadados do par ficam à parte
    X, y, meta = separar_xy(ds)
    grupos = meta["id_escola"]

    # teste fora do tempo: treina nos anos anteriores ao corte e testa no par do corte
    tr, te = indices_temporal(meta, cfg)
    modelos = treinar_regime(X, y, grupos, tr, te, cfg)
    regimes = {REGIME_TEMPORAL: {"modelos": modelos, "info": {
        "n_treino": len(tr), "n_teste": len(te), "prevalencia_teste": float(y.iloc[te].mean()),
        "descricao": f"Treino nos pares com ano de referência < {cfg.ano_ref_teste_temporal}; teste no par "
                     f"{cfg.ano_ref_teste_temporal} -> {cfg.ano_ref_teste_temporal + 1}. Replica o uso real (prever o futuro)."}}}
    melhor = MODELO_FINAL

    # o modelo final é reajustado com todos os pares; os escores out-of-fold definem o limiar e as faixas
    final, limiar, usadas, oof = treinar_final(X, y, grupos, melhor, cfg)
    cortes = cortes_faixas(y, oof, limiar, cfg.precisao_vermelho)
    excluidas = [c for c in FEATURES_NUM + FEATURES_CAT if c not in usadas]
    # o modelo vai com metadados (limiar, colunas, faixas, hash), que a previsão e o dashboard usam
    persistencia.salvar_modelo(cfg, final, {
        "modelo": melhor, "limiar_decisao": limiar, "fbeta": cfg.fbeta,
        "colunas_numericas": [c for c in FEATURES_NUM if c in usadas],
        "colunas_categoricas": [c for c in FEATURES_CAT if c in usadas],
        "variaveis_excluidas_sem_variacao": excluidas,
        "faixas": {**cortes, "validacao": validacao_por_faixa(y, oof, cortes)},
        "alvo": f"alto_risco: abandono em t+1 >= percentil {cfg.percentil_risco:.0%} do proprio ano t+1",
        "fonte_dados": fonte, "hash_dataset": h, "n_pares": len(ds), "seed": cfg.seed,
        "aviso": "MODELO TREINADO COM DADOS SINTETICOS - nao usar para decisoes sobre escolas reais"
        if fonte == "sintetico" else "",
    })
    # o que a avaliação precisa (probabilidades do teste, limiares) fica em arquivo para a fase seguinte
    joblib.dump({"regimes": regimes, "melhor_modelo": melhor, "fonte": fonte, "hash_dataset": h},
                cfg.caminho("processado") / ARQ_RESULTADO)
    return {"fonte": fonte, "pares": len(ds), "modelos_treinados": len(MODELOS), "modelo": melhor,
            "cv_pr_auc": round(modelos[melhor]["cv_pr_auc"], 4), "limiar_final": round(limiar, 4), "variaveis_excluidas": ", ".join(excluidas) or "nenhuma",
            "hash_dataset": h}


@registrar_execucao("avaliar")
def cmd_avaliar(cfg: Config) -> dict:
    """Etapa 3: calcula as métricas do teste, escreve a tabela comparativa e as figuras."""
    arq = cfg.caminho("processado") / ARQ_RESULTADO
    if not arq.exists():
        raise ErroAcionavel("Resultados de treino não encontrados. Rode 'pi-evasao treinar' primeiro.")
    return avaliar_resultados(cfg, joblib.load(arq))


# ---- Etapa 4 ----------------------------------------------------------------------------------
@registrar_execucao("prever")
def cmd_prever(cfg: Config) -> dict:
    """Risco de cada escola para o ano seguinte ao último do painel; exporta os arquivos do dashboard."""
    diretorias = None
    if persistencia.ler_meta(cfg).get("fonte_dados") != "sintetico":  # sintético não tem escolas reais
        try:
            diretorias = carregar_diretorias(cfg)
        except ErroAcionavel as e:  # sem diretorias o app só esconde o filtro
            obter_logger("prever").warning("Diretorias de ensino indisponíveis (%s).", e)
    return gerar_predicoes(cfg, diretorias)


@registrar_execucao("pipeline")
def cmd_pipeline(cfg: Config, fonte: str | None = None, n_escolas: int | None = None) -> dict:
    """Encadeia todas as fases, da ingestão à previsão para o dashboard."""
    r1 = cmd_ingerir(cfg, fonte=fonte, n_escolas=n_escolas)
    r2 = cmd_etl(cfg)
    r3 = cmd_features(cfg)
    r4 = cmd_treinar(cfg)
    r5 = cmd_avaliar(cfg)
    r6 = cmd_prever(cfg)
    return {"fonte": r1["fonte"], "linhas_ingeridas": r1["linhas"], "linhas_limpas": r2["linhas_saida"],
            "pares_t_t1": r3["pares_t_t1"], "modelo": r4["modelo"],
            "pr_auc_fora_do_tempo": r5["pr_auc_fora_do_tempo"], "recall_fora_do_tempo": r5["recall_fora_do_tempo"],
            "possivel_vazamento": r5["possivel_vazamento"], "escolas_previstas": r6["escolas"]}


# ---- entrada ----------------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    """Monta os subcomandos da linha de comando."""
    p = argparse.ArgumentParser(prog="pi-evasao", description=__doc__.split("\n")[0])
    p.add_argument("--config", help="caminho de um config.yaml alternativo")
    sub = p.add_subparsers(dest="comando", required=True)
    # comandos que aceitam escolher a fonte de dados
    for nome, ajuda in (("ingerir", "Etapa 1: obtém o painel bruto"), ("pipeline", "executa todas as fases em sequência")):
        s = sub.add_parser(nome, help=ajuda)
        s.add_argument("--fonte", choices=["sintetico", "inep"], default=None)
        s.add_argument("--n-escolas", type=int, help="(sintético) número de escolas a gerar")
    # comandos sem argumentos além da configuração
    for nome, ajuda in (("etl", "Etapa 1: limpeza e banco SQLite"), ("features", "Etapa 2: engenharia de atributos"),
                        ("treinar", "Etapa 3: treino e validação"),
                        ("avaliar", "Etapa 3: métricas, tabela e figuras"),
                        ("prever", "Etapa 4: risco por escola para o ano seguinte (alimenta o dashboard)")):
        sub.add_parser(nome, help=ajuda)
    return p


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada: executa o subcomando e devolve o código de saída (2 para erro acionável)."""
    for fluxo in (sys.stdout, sys.stderr):  # console do Windows usa cp1252 por padrão
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    cfg = carregar_config(args.config)
    try:
        # ingerir e pipeline aceitam opções; os demais só recebem a configuração
        if args.comando == "ingerir":
            resumo = cmd_ingerir(cfg, fonte=args.fonte, n_escolas=args.n_escolas)
        elif args.comando == "pipeline":
            resumo = cmd_pipeline(cfg, fonte=args.fonte, n_escolas=args.n_escolas)
        else:
            resumo = {"etl": cmd_etl, "features": cmd_features, "treinar": cmd_treinar,
                      "avaliar": cmd_avaliar, "prever": cmd_prever}[args.comando](cfg)
    # erro esperado: mostra só a mensagem (sem traceback) e sai com código 2
    except ErroAcionavel as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        return 2
    print("\nResumo:")
    for k, v in resumo.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
