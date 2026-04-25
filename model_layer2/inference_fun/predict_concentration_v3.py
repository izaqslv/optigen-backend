import os, json, joblib
import pandas as pd

from model_layer2.dataset.dataset_builder_v3 import build_dataset_v3
from model_layer2.utils.estado_v3 import get_estado
from model_layer2.features.build_features_v3 import build_features_v3

# Caminho absoluto para evitar bugs
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts")
)

RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Carregar último run
with open(os.path.join(RUNS_DIR, "last_run.json")) as f:
    paths = json.load(f)

# Reconstruir caminhos completos
MODEL_PATH = os.path.join(MODELS_DIR, paths["model"])
FEATURES_PATH = os.path.join(CONFIG_DIR, paths["features"])

print("📦 Loading model from:", MODEL_PATH)
print("📦 Loading features from:", FEATURES_PATH)

# Carregar artefatos
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)


def predict_concentration_v3(measurements, fluids_meta):
    df = build_dataset_v3(measurements, fluids_meta)
    df = df.sort_values(["fluid_id", "altura", "tempo"]).reset_index(drop=True)

    out = []

    for (fid, h), g in df.groupby(["fluid_id", "altura"]):
        g = g.sort_values("tempo").reset_index(drop=True)

        # condições iniciais (usar real no t0)
        c_prev = g.loc[0, "concentracao"]
        c_prev2 = c_prev

        preds = [c_prev]

        for i in range(1, len(g)):
            row = g.loc[i]

            estado = get_estado(c_prev, c_prev2)
            feats = build_features_v3(row, c_prev, c_prev2, estado)

            X = pd.DataFrame([feats])[FEATURES]  # garante ordem
            y_hat = float(model.predict(X)[0])

            # (opcional) clipping físico
            if y_hat < 0: y_hat = 0.0

            preds.append(y_hat)

            # atualiza memória
            c_prev2 = c_prev
            c_prev = y_hat

        g["pred_concentracao"] = preds
        out.append(g)

    return pd.concat(out).reset_index(drop=True)