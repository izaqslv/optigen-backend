# ================================
# VALIDAÇÃO DE MODELO RF (LIMPO)
# ================================

import joblib
import json
import os
import matplotlib.pyplot as plt

from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.v2_legacy.prepare_dataset import prepare_dataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ================================
# CARREGAR ARTEFATOS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "../..", "artifacts")

with open(os.path.join(ARTIFACTS_DIR, "last_run.json")) as f:
    paths = json.load(f)

model = joblib.load(paths["model"])

with open(paths["features"]) as f:
    features = json.load(f)


# ================================
# CARREGAR DADOS
# ================================
path_excel = "data/DadosSedimentation.xlsx"

measurements, fluids_meta = load_and_align_data(path_excel)

X, y, features, df = prepare_dataset(
    measurements=measurements,
    fluids_meta=fluids_meta
)


# ================================
# VALIDAÇÃO E PLOTS
# ================================
plot_dir = os.path.join(ARTIFACTS_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

print("DEBUG df:", df.columns)


for fluid in df["fluid_id"].unique():

    df_fluid = df[df["fluid_id"] == fluid]
    alturas = sorted(df_fluid["altura"].unique())

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    axes = axes.flatten()

    y_true_all = []
    y_pred_all = []

    for i, altura in enumerate(alturas):

        if i >= len(axes):
            break

        df_h = df_fluid[df_fluid["altura"] == altura]

        # features e target alinhados
        X_rf = df_h[features].iloc[5:]
        y_true = df_h["concentracao"].iloc[5:]
        tempo = df_h["tempo"].iloc[5:]

        # predição
        y_pred = model.predict(X_rf)

        # acumular métricas
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        # plot
        ax = axes[i]
        ax.plot(tempo, y_true, label="Real")
        ax.plot(tempo, y_pred, "--", label="Pred")

        ax.set_title(f"Altura = {altura}")
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Concentração")
        ax.grid(True)

    # ================================
    # MÉTRICAS POR FLUIDO
    # ================================
    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)

    print(f"Fluido {fluid} | R2 = {r2:.4f} | MAE = {mae:.4f} | MSE = {mse:.6f}")

    # ================================
    # SALVAR FIGURA
    # ================================
    fig.legend(["Real", "Pred"], loc="upper right")
    fig.suptitle(f"Fluido {fluid}", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"fluid_{fluid}.png"))
    plt.show()




