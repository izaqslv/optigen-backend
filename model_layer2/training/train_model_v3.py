import json
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.training.build_supervised_v3 import build_supervised_v3
from model_layer2.training.rollout_training_v3 import generate_rollout_dataset_v3
from model_layer2.features.feature_list_v3 import FEATURES_V3

# =========================================================
# Diretórios
# =========================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


# =========================================================
# Funções auxiliares
# =========================================================
def validate_features(X, features):
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"❌ Features ausentes no dataset: {missing}")


def check_nan(df, name="dataset"):
    if df.isnull().any().any():
        cols = df.columns[df.isnull().any()]
        raise ValueError(f"❌ NaN detectado em {name}: {list(cols)}")


def log_metrics(stage, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"\n📊 [{stage}]")
    print(f"R2  = {r2:.4f}")
    print(f"MAE = {mae:.6f}")
    print(f"MSE = {mse:.6f}")

    return {"r2": r2, "mae": mae, "mse": mse}


# =========================================================
# Treinamento principal
# =========================================================
def train_model_v3(measurements, fluids_meta, n_rounds=3):

    print("\n🚀 Iniciando treinamento V3...")

    # -----------------------------------------------------
    # 1) Base supervisionada
    # -----------------------------------------------------
    X, y = build_supervised_v3(measurements, fluids_meta)

    print(f"📦 Dataset bruto: {X.shape}")

    # validação básica
    check_nan(X, "X")
    check_nan(y.to_frame(), "y")

    # -----------------------------------------------------
    # 2) Seleção de features
    # -----------------------------------------------------
    features = FEATURES_V3

    validate_features(X, features)

    X = X[features]

    print(f"🎯 Features usadas ({len(features)}):")
    print(features)

    # -----------------------------------------------------
    # 3) Split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📐 Split:")
    print(f"Train: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    # -----------------------------------------------------
    # 4) Modelo
    # -----------------------------------------------------

    # # Antes: modelo original****************************************************************************************
    # model = LGBMRegressor(
    #     n_estimators=400,
    #     learning_rate=0.05,
    #     max_depth=-1, # melhores: de 8 a 12
    #     # min_sample_leaf= 5, # melhores: 3 a 10
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )
    # # model.fit(X_train, y_train)
    # # Fim do modelo original****************************************************************************************

    # Abaixo: modelo otimizado (com hiperparâmetros):
    param_dist = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [-1, 6, 8, 10],
        "num_leaves": [31, 50, 80, 120],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [5, 10, 20, 50]
    }
    base_model = LGBMRegressor(random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    weights = np.where(
        np.abs(X_train["dist_interface"]) < 0.5,
        4.0,  # região da interface
        np.where(
            np.abs(X_train["dist_interface"]) < 2.0,
            2.0,
            1.0
        )
    )
    search.fit(X_train, y_train, **{"sample_weight": weights})

    # Melhor modelo
    model = search.best_estimator_
    print("\n🔥 MELHOR MODELO ENCONTRADO:")
    print(search.best_params_)

    # ==============================
    # 💾 SALVANDO CONFIGURAÇÕES
    # ==============================
    # 🔹 1. Salvar melhores parâmetros
    best_params = search.best_params_

    params_path = os.path.join(CONFIG_DIR, "best_params_v3.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # 🔹 2. Salvar resultados completos da busca
    cv_results = pd.DataFrame(search.cv_results_)
    cv_path = os.path.join(CONFIG_DIR, "cv_results_v3.csv")
    cv_results.to_csv(cv_path, index=False)

    print(f"✅ Parâmetros salvos em: {params_path}")
    print(f"✅ CV results salvos em: {cv_path}")
    # FIM DA OTIMIZAÇÃO HIPERPARAMÉTRICA DO MODELO ACIMA ......................................................

    # avaliação inicial
    y_pred = model.predict(X_test)
    metrics_initial = log_metrics("BASE", y_test, y_pred)

    # -----------------------------------------------------
    # 5) Rollout Training
    # -----------------------------------------------------
    p = 0.5

    for k in range(n_rounds):
        print(f"\n🔁 Rollout {k + 1}/{n_rounds} | p = {p:.2f}")

        Xr, yr = generate_rollout_dataset_v3(
            model,
            measurements,
            fluids_meta,
            p_use_pred=p
        )

        # valida rollout
        validate_features(Xr, features)
        check_nan(Xr, "X_rollout")
        check_nan(yr.to_frame(), "y_rollout")

        Xr = Xr[features]

        # concatenação segura
        X_all = pd.concat([X, Xr], ignore_index=True)
        y_all = pd.concat([y, yr], ignore_index=True)

        model.fit(X_all, y_all)

        # avaliação após rollout
        y_pred = model.predict(X_test)
        log_metrics(f"ROLLOUT_{k+1}", y_test, y_pred)

        p = min(0.9, p + 0.2)

    # -----------------------------------------------------
    # 6) Salvar artefatos (com versionamento)
    # -----------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(MODELS_DIR, f"model_v3_{timestamp}.pkl")
    joblib.dump(model, model_path)

    features_path = os.path.join(CONFIG_DIR, f"features_v3.json")
    with open(features_path, "w") as f:
        json.dump(features, f, indent=4)

    model_file = os.path.basename(model_path)
    features_file = os.path.basename(features_path)
    last = {
        "model": model_file,
        "features": features_file,
        "timestamp": timestamp,
        "metrics_initial": metrics_initial
    }

    last_path = os.path.join(RUNS_DIR, "last_run.json")
    with open(last_path, "w") as f:
        json.dump(last, f, indent=4)

    print("\n✅ Treinamento concluído!")
    print(f"📦 Modelo e outros artefatos salvos em: {model_path}")
    print(f"\n📦 Artefatos salvos:")
    print(f" Model: {model_file}")
    print(f" Features: {features_file}")
    print(f" Run file: {last_path}")

    return model, features, model_path


# =========================================================
# Execução direta
# =========================================================
if __name__ == "__main__":

    print("\n📥 Carregando dados...")

    measurements, fluids_meta = load_and_align_data(
        "data/DadosSedimentation.xlsx"
    )

    train_model_v3(
        measurements=measurements,
        fluids_meta=fluids_meta,
        n_rounds=3
    )