"""Diretoria de ensino por escola (ADR 0010): leitura das duas fontes e regra de prioridade."""

import pandas as pd

from pi_evasao.ingestao.diretorias import atribuir_diretoria, ler_seduc, ler_sme

# Recortes no layout real dos arquivos (separador ";", latin1), com colunas a mais que devem ser ignoradas.
SEDUC = ("NOMEDEP;DE;MUN;CD_IBGE;COD_ESC;CODESCMEC;NOMESC\n"
         "ESTADUAL - SE;LESTE 1;SAO PAULO;3550308;3256;35003256;PEDRO TAQUES\n"
         "ESTADUAL - SE;LESTE 2;SAO PAULO;3550308;1;35000001;OUTRA\n"
         "ESTADUAL - SE;ADAMANTINA;ADAMANTINA;3500105;30806;35030806;HELEN KELLER\n")
SME = ("DRE;CODESC;DIRETORIA;CODINEP\n"
       "IP;1;ITAIM PAULISTA;35271160\n"
       "IP;2;ITAIM PAULISTA;\n")  # escola conveniada sem código INEP: descartada


def test_prioridade_escola_depois_municipio(tmp_path):
    """A diretoria da escola vale mais que a do município."""
    (tmp_path / "seduc.csv").write_text(SEDUC, encoding="latin1")
    (tmp_path / "sme.csv").write_text(SME, encoding="latin1")
    seduc, sme = ler_seduc(tmp_path / "seduc.csv"), ler_sme(tmp_path / "sme.csv")
    assert len(sme) == 1

    escolas = pd.DataFrame({
        "id_escola":    [35003256, 35271160, 35999999, 35888888],
        "id_municipio": [3550308,  3550308,  3500105,  3550308],
    })
    d = atribuir_diretoria(escolas, seduc, sme)
    assert d[0] == "DE Leste 1"          # estadual: código na SEDUC
    assert d[1] == "DRE Itaim Paulista"  # municipal da capital: código na SME
    assert d[2] == "DE Adamantina"       # interior: DE única do município
    assert pd.isna(d[3])                 # capital tem várias DEs: sem código, não há como decidir
