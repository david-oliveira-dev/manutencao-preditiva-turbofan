"""Testes do carregamento dos arquivos brutos do C-MAPSS.

Os arquivos reais não são versionados, então escrevemos arquivos temporários
com o mesmo formato (sem cabeçalho, separado por espaços).
"""

from __future__ import annotations

from pathlib import Path

from src.data import COLUNAS, OP_SETTINGS, SENSORS, carregar_rul, carregar_serie


def test_layout_de_colunas_do_cmapss() -> None:
    """26 colunas: unit, ciclo, 3 ajustes e 21 sensores."""
    assert len(OP_SETTINGS) == 3
    assert len(SENSORS) == 21
    assert len(COLUNAS) == 26
    assert COLUNAS[:2] == ["unit_number", "time_in_cycles"]


def test_carregar_serie_nomeia_as_colunas(tmp_path: Path) -> None:
    arquivo = tmp_path / "train.txt"
    linha = " ".join(str(float(i)) for i in range(26))
    arquivo.write_text(f"{linha}\n{linha}\n")

    df = carregar_serie(arquivo)

    assert list(df.columns) == COLUNAS
    assert len(df) == 2


def test_carregar_serie_tolera_espacos_multiplos(tmp_path: Path) -> None:
    """O arquivo original do C-MAPSS usa espaçamento irregular e sobra no fim."""
    arquivo = tmp_path / "train.txt"
    valores = [str(float(i)) for i in range(26)]
    arquivo.write_text("  ".join(valores) + " \n")

    df = carregar_serie(arquivo)

    assert df.shape == (1, 26)
    assert df["unit_number"].iloc[0] == 0.0


def test_carregar_rul_devolve_uma_serie(tmp_path: Path) -> None:
    arquivo = tmp_path / "RUL.txt"
    arquivo.write_text("112\n98\n69\n")

    rul = carregar_rul(arquivo)

    assert list(rul) == [112, 98, 69]
    assert rul.name == "RUL"
