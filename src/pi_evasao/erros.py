"""Exceções do projeto.

``ErroAcionavel`` é o erro que a CLI mostra SEM traceback: a mensagem já diz ao usuário
o que fazer (ex.: "rode 'pi-evasao etl' primeiro").
"""


class ErroAcionavel(Exception):
    """Falha esperada cuja mensagem instrui o usuário sobre como corrigi-la."""
