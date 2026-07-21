"""Testes da construção do alvo RUL e da seleção de features."""

from __future__ import annotations

import pandas as pd

from src.features import adicionar_rul, colunas_constantes, features_uteis, ultima_leitura


def test_rul_zera_no_ultimo_ciclo_de_cada_motor(df_treino: pd.DataFrame) -> None:
    """No treino todo motor vai até a falha: a última leitura tem RUL = 0."""
    out = adicionar_rul(df_treino)
    ultimos = out.loc[out.groupby("unit_number")["time_in_cycles"].idxmax()]
    assert (ultimos["RUL"] == 0).all()


def test_rul_decresce_de_um_em_um(df_treino: pd.DataFrame) -> None:
    """RUL = ciclo_max − ciclo_atual, então cai 1 a cada ciclo."""
    out = adicionar_rul(df_treino)
    motor_1 = out[out["unit_number"] == 1].sort_values("time_in_cycles")
    assert motor_1["RUL"].tolist() == [2, 1, 0]

    motor_2 = out[out["unit_number"] == 2].sort_values("time_in_cycles")
    assert motor_2["RUL"].tolist() == [1, 0]


def test_rul_nunca_e_negativa(df_treino: pd.DataFrame) -> None:
    out = adicionar_rul(df_treino)
    assert (out["RUL"] >= 0).all()


def test_adicionar_rul_nao_altera_o_dataframe_original(df_treino: pd.DataFrame) -> None:
    """A função copia: quem chamou não deve ver o DataFrame mudar."""
    colunas_antes = list(df_treino.columns)
    adicionar_rul(df_treino)
    assert list(df_treino.columns) == colunas_antes
    assert "RUL" not in df_treino.columns


def test_colunas_constantes_detecta_ajustes_sem_variacao(df_treino: pd.DataFrame) -> None:
    """Os op_settings foram fixados em 0.0 na fixture: devem ser detectados."""
    constantes = colunas_constantes(df_treino)
    assert "op_setting_1" in constantes
    assert "op_setting_2" in constantes
    assert "op_setting_3" in constantes


def test_colunas_constantes_ignora_sensores_que_variam(df_treino: pd.DataFrame) -> None:
    constantes = colunas_constantes(df_treino)
    assert "sensor_1" not in constantes


def test_features_uteis_remove_as_descartadas(df_treino: pd.DataFrame) -> None:
    descartar = colunas_constantes(df_treino)
    feats = features_uteis(df_treino, descartar)
    assert feats, "a lista de features não pode ficar vazia"
    assert not set(feats) & set(descartar)


def test_features_uteis_sem_descarte_devolve_tudo(df_treino: pd.DataFrame) -> None:
    feats = features_uteis(df_treino, None)
    assert len(feats) == 24  # 21 sensores + 3 ajustes operacionais


def test_ultima_leitura_pega_o_maior_ciclo_de_cada_motor(df_treino: pd.DataFrame) -> None:
    """No teste do C-MAPSS a previsão é feita sobre a última leitura de cada motor."""
    out = ultima_leitura(df_treino, ["unit_number", "time_in_cycles", "sensor_1"])
    assert len(out) == 2  # um registro por motor
    esperado = {1: 3, 2: 2}
    for _, linha in out.iterrows():
        assert linha["time_in_cycles"] == esperado[linha["unit_number"]]
