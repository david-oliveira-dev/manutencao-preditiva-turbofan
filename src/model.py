"""Treino, comparação e avaliação honesta dos modelos de RUL.

Roda como script (`python -m src.model`):
  1. carrega FD001, monta o alvo RUL e seleciona features informativas;
  2. treina um baseline (regressão linear) + Random Forest + Gradient Boosting;
  3. avalia na última leitura de cada motor de teste contra o RUL verdadeiro;
  4. faz um GridSearch pequeno no melhor modelo;
  5. salva as figuras de modelagem em reports/figures/ e imprime as métricas.

Métrica principal: RMSE em CICLOS (quanto o modelo erra, em média, em ciclos
de operação). MAE como apoio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem tela (para salvar PNG em qualquer ambiente)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .data import carregar_rul, carregar_serie
from .features import adicionar_rul, colunas_constantes, features_uteis, ultima_leitura

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
FIGURAS = RAIZ / "reports" / "figures"


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    """Carrega treino/teste/RUL, cria o alvo e seleciona as features úteis."""
    treino = adicionar_rul(carregar_serie(DATA / "train_FD001.txt"))
    teste = carregar_serie(DATA / "test_FD001.txt")
    rul_teste = carregar_rul(DATA / "RUL_FD001.txt")
    descartadas = colunas_constantes(treino)
    feats = features_uteis(treino, descartadas)
    return treino, teste, rul_teste, feats, descartadas


def avaliar(y_real: np.ndarray, y_prev: np.ndarray) -> dict[str, float]:
    """RMSE e MAE em ciclos (RUL prevista nunca é negativa: cortamos em 0)."""
    y_prev = np.clip(y_prev, 0, None)
    return {
        "RMSE": root_mean_squared_error(y_real, y_prev),
        "MAE": mean_absolute_error(y_real, y_prev),
    }


def _grafico_previsto_vs_real(y_real, y_prev, nome_modelo: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_prev, alpha=0.6, edgecolor="none", color="#305496")
    lim = max(y_real.max(), y_prev.max())
    ax.plot([0, lim], [0, lim], "--", color="crimson", label="previsão perfeita")
    ax.set_xlabel("RUL real (ciclos)")
    ax.set_ylabel("RUL prevista (ciclos)")
    ax.set_title(f"Previsto vs. Real — {nome_modelo}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "previsto_vs_real.png", dpi=120)
    plt.close(fig)


def _grafico_residuos(y_real, y_prev, nome_modelo: str) -> None:
    residuos = y_real - np.clip(y_prev, 0, None)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuos, bins=25, color="#305496", edgecolor="white")
    ax.axvline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Resíduo = RUL real − prevista (ciclos)")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição dos resíduos — {nome_modelo}")
    fig.tight_layout()
    fig.savefig(FIGURAS / "residuos.png", dpi=120)
    plt.close(fig)


def _grafico_erro_por_faixa(y_real, y_prev, nome_modelo: str) -> None:
    """Erro absoluto médio por faixa de RUL real — mostra ONDE o modelo erra mais."""
    df = pd.DataFrame({"real": y_real, "abs_err": np.abs(y_real - np.clip(y_prev, 0, None))})
    faixas = pd.cut(df["real"], bins=[0, 50, 100, 150, 200, 1000],
                    labels=["0–50", "50–100", "100–150", "150–200", "200+"])
    mae_faixa = df.groupby(faixas, observed=True)["abs_err"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    mae_faixa.plot.bar(ax=ax, color="#305496")
    ax.set_xlabel("Faixa de RUL real (ciclos)")
    ax.set_ylabel("Erro absoluto médio (ciclos)")
    ax.set_title(f"Onde o modelo erra mais — {nome_modelo}")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURAS / "erro_por_faixa.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    treino, teste, rul_teste, feats, descartadas = preparar_dados()
    print(f"Motores no treino: {treino['unit_number'].nunique()} | leituras: {len(treino)}")
    print(f"Colunas constantes descartadas ({len(descartadas)}): {descartadas}")
    print(f"Features usadas ({len(feats)}): {feats}\n")

    X_treino, y_treino = treino[feats], treino["RUL"]
    X_teste = ultima_leitura(teste, feats)
    y_teste = rul_teste.to_numpy()

    # Padronização: o scaler aprende SÓ no treino (sem vazar o teste).
    scaler = StandardScaler().fit(X_treino)
    Xtr = scaler.transform(X_treino)
    Xte = scaler.transform(X_teste)

    modelos = {
        "Baseline (Linear)": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    resultados: dict[str, dict[str, float]] = {}
    previsoes: dict[str, np.ndarray] = {}
    for nome, modelo in modelos.items():
        modelo.fit(Xtr, y_treino)
        p = modelo.predict(Xte)
        resultados[nome] = avaliar(y_teste, p)
        previsoes[nome] = np.clip(p, 0, None)
        print(f"{nome:20s}  RMSE={resultados[nome]['RMSE']:6.2f}  "
              f"MAE={resultados[nome]['MAE']:6.2f}  (ciclos)")

    # GridSearch pequeno e justificado no Random Forest (modelo principal).
    print("\nGridSearch (Random Forest, 3-fold)...")
    grade = {"n_estimators": [200, 400], "max_depth": [None, 10, 20],
             "min_samples_leaf": [1, 3]}
    busca = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        grade, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    busca.fit(Xtr, y_treino)
    p_rf = busca.best_estimator_.predict(Xte)
    resultados["Random Forest (tuned)"] = avaliar(y_teste, p_rf)
    previsoes["Random Forest (tuned)"] = np.clip(p_rf, 0, None)
    print(f"  melhores params: {busca.best_params_}")
    print(f"  Random Forest (tuned)  RMSE={resultados['Random Forest (tuned)']['RMSE']:6.2f}  "
          f"MAE={resultados['Random Forest (tuned)']['MAE']:6.2f}")

    # Melhor modelo pelo RMSE -> gera as figuras de avaliação.
    melhor = min(resultados, key=lambda k: resultados[k]["RMSE"])
    print(f"\nMelhor modelo: {melhor} (RMSE={resultados[melhor]['RMSE']:.2f} ciclos)")
    yb = previsoes[melhor]
    _grafico_previsto_vs_real(y_teste, yb, melhor)
    _grafico_residuos(y_teste, yb, melhor)
    _grafico_erro_por_faixa(y_teste, yb, melhor)
    print(f"Figuras salvas em {FIGURAS}")

    return resultados


if __name__ == "__main__":
    main()
