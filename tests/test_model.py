"""Testes da função de avaliação (RMSE/MAE em ciclos)."""

from __future__ import annotations

import numpy as np

from src.model import avaliar


def test_previsao_perfeita_zera_o_erro() -> None:
    y = np.array([10.0, 20.0, 30.0])
    metricas = avaliar(y, y.copy())
    assert metricas["RMSE"] == 0.0
    assert metricas["MAE"] == 0.0


def test_devolve_rmse_e_mae() -> None:
    metricas = avaliar(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert set(metricas) == {"RMSE", "MAE"}
    assert metricas["MAE"] == 2.0


def test_rul_prevista_negativa_e_cortada_em_zero() -> None:
    """Vida útil restante negativa não existe: o clip em 0 deve valer."""
    y_real = np.array([0.0, 0.0])
    com_negativo = avaliar(y_real, np.array([-50.0, -10.0]))
    cortado = avaliar(y_real, np.array([0.0, 0.0]))
    assert com_negativo == cortado


def test_rmse_penaliza_mais_que_mae_com_erro_concentrado() -> None:
    """Um erro grande isolado eleva o RMSE acima do MAE — é o caso do FD001."""
    y_real = np.array([50.0, 50.0, 50.0, 50.0])
    y_prev = np.array([50.0, 50.0, 50.0, 90.0])
    metricas = avaliar(y_real, y_prev)
    assert metricas["RMSE"] > metricas["MAE"]
