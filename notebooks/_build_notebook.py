"""Gera o notebook 01_eda_modelagem.ipynb com nbformat (fonte única, sem JSON na mão).

Rode da raiz do projeto:  python notebooks/_build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(
    "# Manutenção Preditiva de Motores (NASA C-MAPSS / FD001)\n\n"
    "Prevê a **Vida Útil Restante (RUL)** de motores turbofan a partir de leituras "
    "de sensores — um problema de **regressão supervisionada**.\n\n"
    "Fluxo: EDA → engenharia do alvo RUL → modelagem (baseline + ensembles) → "
    "avaliação honesta. A lógica vive no pacote `src/`; aqui mostramos o passo a passo."
))

cells.append(md("## 0. Setup e imports"))
cells.append(code(
    "import sys\n"
    "from pathlib import Path\n\n"
    "# Permite importar o pacote `src` rodando o Jupyter da raiz OU de notebooks/.\n"
    "RAIZ = Path.cwd()\n"
    "if not (RAIZ / 'src').exists():\n"
    "    RAIZ = RAIZ.parent\n"
    "sys.path.insert(0, str(RAIZ))\n\n"
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd\n\n"
    "from src.data import SENSORS, carregar_rul, carregar_serie\n"
    "from src.features import adicionar_rul, colunas_constantes, features_uteis\n\n"
    "DATA = RAIZ / 'data'\n"
    "FIG = RAIZ / 'reports' / 'figures'\n"
    "FIG.mkdir(parents=True, exist_ok=True)\n"
    "print('Setup OK')"
))

cells.append(md(
    "## 1. Leitura e EDA\n\n"
    "Cada linha é um ciclo de operação de um motor. No treino, todo motor roda "
    "**até falhar**."
))
cells.append(code(
    "treino = carregar_serie(DATA / 'train_FD001.txt')\n"
    "print(f'{treino.unit_number.nunique()} motores | {len(treino)} leituras | {treino.shape[1]} colunas')\n"
    "treino.head()"
))

cells.append(md(
    "### 1.1 Distribuição da duração de vida\n"
    "Quantos ciclos cada motor dura até a falha?"
))
cells.append(code(
    "vida = treino.groupby('unit_number').time_in_cycles.max()\n"
    "print(vida.describe()[['min', 'mean', 'max']].round(1).to_dict())\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.hist(vida, bins=25, color='#305496', edgecolor='white')\n"
    "ax.set_xlabel('Ciclos até a falha'); ax.set_ylabel('Nº de motores')\n"
    "ax.set_title('Distribuição da duração de vida dos motores (FD001)')\n"
    "fig.tight_layout(); fig.savefig(FIG / 'distribuicao_vida.png', dpi=120); plt.show()"
))

cells.append(md(
    "### 1.2 Quais sensores DEGRADAM ao longo da vida?\n\n"
    "Adicionamos o alvo RUL e medimos a **correlação de cada sensor com a RUL**. "
    "Sensores com |correlação| alta carregam o sinal de degradação; os de correlação "
    "~0 são constantes/ruído e serão descartados."
))
cells.append(code(
    "treino_rul = adicionar_rul(treino)\n"
    "constantes = colunas_constantes(treino_rul)\n"
    "corr = treino_rul[SENSORS].corrwith(treino_rul.RUL).dropna().sort_values()\n"
    "print('Sensores constantes (descartados):', constantes)\n"
    "print('\\nTop sensores por correlação com a RUL:')\n"
    "print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(8).round(3))"
))
cells.append(code(
    "# Trajetória dos 4 sensores mais informativos para alguns motores.\n"
    "mais_informativos = corr.abs().sort_values(ascending=False).head(4).index.tolist()\n"
    "motores = [1, 24, 50, 100]\n"
    "fig, axs = plt.subplots(2, 2, figsize=(11, 7))\n"
    "for ax, sensor in zip(axs.ravel(), mais_informativos):\n"
    "    for m in motores:\n"
    "        sub = treino_rul[treino_rul.unit_number == m]\n"
    "        ax.plot(sub.time_in_cycles, sub[sensor], label=f'motor {m}', alpha=0.8)\n"
    "    ax.set_title(sensor); ax.set_xlabel('ciclo'); ax.set_ylabel('leitura')\n"
    "axs[0, 0].legend(fontsize=8)\n"
    "fig.suptitle('Trajetória dos sensores que mais degradam')\n"
    "fig.tight_layout(); fig.savefig(FIG / 'trajetoria_sensores.png', dpi=120); plt.show()"
))
cells.append(md(
    "**Leitura:** os sensores acima têm tendência clara (sobem ou descem) conforme o "
    "motor se aproxima da falha — é deles que o modelo extrai o sinal. Os sensores "
    "listados como constantes não variam e não ajudam."
))

cells.append(md(
    "## 2. Modelagem e avaliação\n\n"
    "Reusamos o pipeline do pacote (`src/model.py`): baseline linear + Random Forest "
    "+ Gradient Boosting + um GridSearch pequeno no RF. Métrica em **ciclos**."
))
cells.append(code(
    "from src.model import main as treinar_e_avaliar\n"
    "resultados = treinar_e_avaliar()\n"
    "pd.DataFrame(resultados).T.round(2).sort_values('RMSE')"
))

cells.append(md(
    "### 2.1 Diagnóstico do melhor modelo\n"
    "As figuras abaixo (geradas pelo pipeline) mostram previsto×real, os resíduos e "
    "**onde** o erro se concentra."
))
cells.append(code(
    "from IPython.display import Image, display\n"
    "for nome in ['previsto_vs_real.png', 'residuos.png', 'erro_por_faixa.png']:\n"
    "    display(Image(filename=str(FIG / nome)))"
))
cells.append(md(
    "## 3. Conclusão\n\n"
    "- O erro se concentra nos motores **longe da falha** (RUL alto): com a janela de "
    "sensores ainda \"saudável\", o sinal de degradação é fraco e o modelo tende a "
    "subestimar a RUL.\n"
    "- Próximo passo padrão na literatura: **limitar a RUL** (ex.: clipar em ~125 "
    "ciclos), refletindo que muito antes da falha a RUL não é prevista pelos sensores. "
    "Isso costuma reduzir bastante o RMSE.\n"
    "- Os números do README vêm **desta execução real**, sem ajuste cosmético."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

destino = Path(__file__).resolve().parent / "01_eda_modelagem.ipynb"
nbf.write(nb, destino)
print(f"Notebook escrito em {destino}")
