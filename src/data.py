"""Carregamento dos dados NASA C-MAPSS (subconjunto FD001).

Os arquivos vêm SEM cabeçalho e são separados por espaços. Cada linha é uma
leitura (um ciclo de operação) de um motor:

    unit_number  time_in_cycles  op_setting_1..3  sensor_1..21   (26 colunas)

O nome de cada coluna é definido aqui, no carregamento.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 3 ajustes operacionais + 21 sensores = as 24 variáveis medidas.
OP_SETTINGS: list[str] = [f"op_setting_{i}" for i in range(1, 4)]
SENSORS: list[str] = [f"sensor_{i}" for i in range(1, 22)]
COLUNAS: list[str] = ["unit_number", "time_in_cycles", *OP_SETTINGS, *SENSORS]


def carregar_serie(caminho: str | Path) -> pd.DataFrame:
    """Lê um arquivo train/test do C-MAPSS e devolve um DataFrame nomeado.

    O separador é um ou mais espaços (`\\s+`); a última coluna pode vir com
    espaço sobrando no arquivo original — o pandas ignora isso.
    """
    return pd.read_csv(caminho, sep=r"\s+", header=None, names=COLUNAS)


def carregar_rul(caminho: str | Path) -> pd.Series:
    """Lê o RUL verdadeiro do conjunto de teste (um valor por motor)."""
    return pd.read_csv(caminho, sep=r"\s+", header=None, names=["RUL"])["RUL"]
