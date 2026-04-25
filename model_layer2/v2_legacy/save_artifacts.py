import os
import json
import joblib
from datetime import datetime

def save_artifacts(model, metrics, features, metadata):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 🔥 Caminho absoluto
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

    MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
    CONFIG_DIR = os.path.join(ARTIFACTS_DIR, "config")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # =========================
    # salvar modelo
    # =========================
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.pkl")
    joblib.dump(model, model_path)

    # =========================
    # salvar métricas
    # =========================
    metrics_path = os.path.join(CONFIG_DIR, f"metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # =========================
    # salvar features
    # =========================
    features_path = os.path.join(CONFIG_DIR, f"features_{timestamp}.json")
    with open(features_path, "w") as f:
        json.dump(features, f, indent=4)

    # =========================
    # salvar metadata
    # =========================
    metadata_path = os.path.join(CONFIG_DIR, f"metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✔ Artefatos salvos:")
    print("Modelo:", model_path)
    print("Métricas:", metrics_path)
    print("Features:", features_path)
    print("Metadata:", metadata_path)

    return {
        "model": model_path,
        "metrics": metrics_path,
        "features": features_path,
        "metadata": metadata_path
    }
