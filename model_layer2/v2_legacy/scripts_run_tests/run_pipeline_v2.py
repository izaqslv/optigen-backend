# ESSE CARA É QUEM CONECTA TUDO!
from datetime import datetime
import json
import os
from model_layer.analysis.data_loader import load_and_align_data
from model_layer.analysis.dataset_builder import build_dataset
from model_layer2.v2_legacy.prepare_dataset import prepare_dataset
from model_layer2.v2_legacy.train_model import train_model
from model_layer2.v2_legacy.evaluate import evaluate
from model_layer2.v2_legacy.save_artifacts import save_artifacts
from model_layer2.v2_legacy.split import split_by_fluid

# =========================
# 1. LOAD DATA
# =========================

path_excel = "data/DadosSedimentation.xlsx"

measurements, fluids_meta = load_and_align_data(path_excel)

df = build_dataset(measurements=measurements, fluids_meta=fluids_meta)

print("📊 Dataset bruto:", df.shape)


# DEBUG
# checar colunas
print("\nColunas disponíveis:")
print(df.columns)
#

# =========================
# 3. PREPARE DATASET
# =========================

X, y, features, df = prepare_dataset(measurements, fluids_meta)

print("✅ Dataset final:", X.shape)

# DEBUG CRÍTICO
print("\n% NaN por coluna:")
print(X.isna().mean().sort_values(ascending=False))

# =========================
# 4. SPLIT
# =========================

X_train, X_test, y_train, y_test = split_by_fluid(df, X, y)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# =========================
# 5. TRAIN
# =========================

model = train_model(X_train, y_train)

# =========================
# 6. EVALUATE
# =========================

metrics = evaluate(model, X_test, y_test)

print("📈 Metrics:", metrics)

# =========================
# 7. SAVE
# =========================

metadata = {
    "n_samples": len(X),
    "n_features": len(features),
    "features": features,
    "model_type": "RandomForestRegressor",
    "target": "concentracao",
    "timestamp": datetime.now().isoformat()
}

paths = save_artifacts(
    model=model,
    metrics=metrics,
    features=features,
    metadata=metadata
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "../..", "artifacts")

last_run_path = os.path.join(ARTIFACTS_DIR, "last_run.json")

with open(last_run_path, "w") as f:
    json.dump(paths, f, indent=4)

print("✅ last_run.json salvo em:", last_run_path)