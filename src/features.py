"""Engenharia de features e construção do rótulo RUL.

Definição do alvo (conforme o problema C-MAPSS):
    RUL (Remaining Useful Life) = (último ciclo do motor) − (ciclo atual)

No treino, cada motor é operado ATÉ A FALHA, então o último ciclo de cada
motor tem RUL = 0. No teste, a série é truncada antes da falha e o RUL
verdadeiro da última leitura vem num arquivo à parte (RUL_FD001.txt).
"""

from __future__ import annotations

import pandas as pd

from .data import OP_SETTINGS, SENSORS


def adicionar_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna `RUL` ao DataFrame de treino.

    Para cada motor: RUL = ciclo máximo daquele motor − ciclo atual.
    """
    ciclo_max = df.groupby("unit_number")["time_in_cycles"].transform("max")
    out = df.copy()
    out["RUL"] = ciclo_max - out["time_in_cycles"]
    return out


def colunas_constantes(df: pd.DataFrame, limiar_std: float = 1e-6) -> list[str]:
    """Identifica sensores/ajustes praticamente CONSTANTES (sem informação).

    No FD001 há uma única condição de operação, então os ajustes operacionais
    e vários sensores não variam — não ajudam o modelo e devem sair.
    """
    candidatas = OP_SETTINGS + SENSORS
    desvios = df[candidatas].std(numeric_only=True)
    return sorted(desvios[desvios <= limiar_std].index.tolist())


def features_uteis(df: pd.DataFrame, descartar: list[str] | None = None) -> list[str]:
    """Lista das colunas de entrada do modelo (sensores/ajustes informativos).

    Mantém apenas colunas que existem no DataFrame e não foram descartadas.
    """
    remover = set(descartar or [])
    return [c for c in (*SENSORS, *OP_SETTINGS) if c in df.columns and c not in remover]


def ultima_leitura(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Para o conjunto de teste: a ÚLTIMA leitura (maior ciclo) de cada motor.

    É o ponto em que queremos prever a RUL e comparar com RUL_FD001.txt.
    """
    idx = df.groupby("unit_number")["time_in_cycles"].idxmax()
    return df.loc[idx, colunas].reset_index(drop=True)
