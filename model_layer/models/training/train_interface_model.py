import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from model_layer.models.training.train_model_core import metrics


# =========================================
# 1. PREPARAR DATASET
# =========================================

def prepare_dataset(df):

    df_model = df.groupby("fluid_id").agg({
        "altura_interface": "first",
        "gradiente_altura": "mean",
        "gradiente_local": "mean",
        "c_max": "max",
        "slope_min": "min"
    }).reset_index()

    features = [
        "gradiente_altura",
        "gradiente_local",
        "c_max",
        "slope_min"
    ]

    X = df_model[features]
    y = df_model["altura_interface"]

    return X, y, features, df_model


# =========================================
# 2. SPLIT POR FLUIDO (CRÍTICO!)
# =========================================

def split_data(X, y, df):

    groups = df["fluid_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


# =========================================
# 3. TREINAR MODELO
# =========================================

def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# =========================================
# 4. AVALIAR
# =========================================

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== RESULTADOS ===")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    metrics = {"MAE": float(mae), "R2": float(r2)}

    return y_pred, metrics


# =========================================
# 5. IMPORTÂNCIA DAS FEATURES
# =========================================

def show_feature_importance(model, features):

    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(df_imp)

    return df_imp


# =========================================
# PIPELINE PRINCIPAL
# =========================================

def run_training(df):

    print("🔧 Preparando dataset...")
    X, y, features, df_model = prepare_dataset(df)

    print("📊 Split por fluido...")
    X_train, X_test, y_train, y_test = split_data(X, y, df_model)

    print("🌲 Treinando Random Forest...")
    model = train_model(X_train, y_train)

    print("📉 Avaliando modelo...")
    y_pred, metrics = evaluate(model, X_test, y_test)

    print("🧠 Interpretando modelo...")
    df_imp = show_feature_importance(model, features)

    print("💾 Salvando artefatos...")
    save_model(model)
    save_metrics(metrics)
    save_feature_importance(df_imp)

    return model, df_imp, metrics


#----------------------------------
# Salvar modelo:
def save_model(model):

    os.makedirs("artifacts/models", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = f"artifacts/models/interface_model_{timestamp}.pkl"

    joblib.dump(model, path)

    print(f"💾 Modelo salvo em: {path}")

    return path

#----------------------------------
# Salvar métricas:

#----------------------------------
# Salvar métricas edm json:
def save_metrics(metrics):

    os.makedirs("artifacts/metrics", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = f"artifacts/metrics/metrics_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📊 Métricas salvas em: {path}")

    return path

#----------------------------------
# Salvar importância das features:
def save_feature_importance(df_imp):

    os.makedirs("artifacts/metrics", exist_ok=True)

    path = "artifacts/metrics/feature_importance.csv"

    df_imp.to_csv(path, index=False)

    print(f"🧠 Feature importance salva em: {path}")