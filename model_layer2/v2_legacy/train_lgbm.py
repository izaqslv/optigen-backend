# ================================
# TREINAMENTO LIGHTGBM
# ================================

import json
import os
import joblib

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.v2_legacy.prepare_dataset import prepare_dataset


# ================================
# PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


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
# SPLIT TEMPORAL (IGUAL RF)
# ================================
split = int(0.8 * len(X))

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")


# ================================
# MODELO LGBM
# ================================
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbosity=-1 # remove warnings
)

model.fit(X_train, y_train)


# ================================
# AVALIAÇÃO
# ================================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"LGBM | R2 = {r2:.4f} | MAE = {mae:.4f} | MSE = {mse:.6f}")


# ================================
# SALVAR ARTEFATOS
# ================================
model_path = os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl")
features_path = os.path.join(ARTIFACTS_DIR, "lgbm_features.json")
metrics_path = os.path.join(ARTIFACTS_DIR, "lgbm_metrics.json")

joblib.dump(model, model_path)

with open(features_path, "w") as f:
    json.dump(features, f, indent=4)

metrics = {
    "r2": float(r2),
    "mae": float(mae),
    "mse": float(mse)
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)


# ================================
# SALVAR COMO ÚLTIMO RUN
# ================================
last_run = {
    "model": model_path,
    "features": features_path,
    "metrics": metrics_path,
    "model_type": "lgbm"
}

with open(os.path.join(ARTIFACTS_DIR, "last_run.json"), "w") as f:
    json.dump(last_run, f, indent=4)


print("🚀 LGBM treinado e salvo com sucesso!")