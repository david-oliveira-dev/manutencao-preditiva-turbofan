"""Fixtures compartilhadas.

Os testes NÃO dependem do dataset C-MAPSS real (que não é versionado): montam
DataFrames pequenos com o mesmo formato de coluna, o suficiente para exercitar
a lógica de features e de avaliação.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import COLUNAS, OP_SETTINGS, SENSORS


@pytest.fixture
def df_treino() -> pd.DataFrame:
    """Dois motores operados até a falha, no formato do train_FD001.

    Motor 1: 3 ciclos (1, 2, 3) → RUL esperada 2, 1, 0.
    Motor 2: 2 ciclos (1, 2)    → RUL esperada 1, 0.

    `op_setting_1` é constante de propósito (sem informação, deve ser
    descartada); `sensor_1` varia (deve ser mantido).
    """
    linhas = [
        # unit, ciclo
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
    ]
    dados = []
    for i, (unit, ciclo) in enumerate(linhas):
        registro: dict[str, float] = {"unit_number": unit, "time_in_cycles": ciclo}
        for s in OP_SETTINGS:
            registro[s] = 0.0  # constantes: sem variação
        for j, s in enumerate(SENSORS):
            registro[s] = float(i + j)  # variam entre as linhas
        dados.append(registro)
    return pd.DataFrame(dados, columns=COLUNAS)
