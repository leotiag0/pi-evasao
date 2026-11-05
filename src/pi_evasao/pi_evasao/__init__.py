"""PI IV Univesp (PJI410, 2026S2) - Plataforma de alerta precoce de evasão escolar.

Pacote organizado pelas Etapas do PI:

- ``ingestao``   -> Etapa 1: obtenção dos dados (planilhas do INEP / sintético)
- ``etl``        -> Etapa 1: limpeza e banco relacional (SQLite)
- ``features``   -> Etapa 2: engenharia de atributos
- ``modelagem``  -> Etapa 3: treino, avaliação e serialização do modelo
"""

__version__ = "0.1.0"
