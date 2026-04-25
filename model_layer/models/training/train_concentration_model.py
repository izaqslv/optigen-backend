import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json
import joblib
from datetime import datetime


# PREPARAÇÃO DO DATASET:----------------------------------------------------------
def prepare_dataset_concentration(df: pd.DataFrame):
    print("📦 Preparando dataset de concentração...")

    # features principais
    features = [
        "tempo",
        "altura",
        "dist_interface",

        # dinâmica temporal (essencial)
        "dC_dt",
        "dC_dt_smooth",
        "d2C_dt2",
        "c_lag1",

        # física espacial
        "gradiente_local",
        "gradiente_altura",

        # propriedades globais
        "c_max",
        "slope_min",

        # regime
        "estado"
    ]

    # Garantir features válidas: manter apenas colunas existentes
    missing = [f for f in features if f not in df.columns]
    if missing:
        print("\n⚠️ FEATURES AUSENTES:", missing)
    features = [f for f in features if f in df.columns]
    #

    # Verificar presença de NaN
    print("\n📊 % de NaN por coluna:")
    print(df[features].isna().mean().sort_values(ascending=False))

    # remover NaNs importantes
    df_original = df.copy()
    print("Antes:", len(df))
    df = df.dropna(subset=features)
    print("Depois:", len(df))
    print("Perda (%):", 100*(1-len(df)/len(df_original)))

    if len(df) == 0:
        raise ValueError("  Dataset vazio após dropna. Verifique features e NaNs!")

    X = df[features].copy()
    y = df["concentracao"].copy()

    print(f"✔️ Features usadas: {features}")
    print(f"✔ Shape: {X.shape}")

    return X, y, features


# SPLIT CORRETO:------------------------------------------------------------------------------
def split_by_fluid(df, X, y, test_size=0.2):
    print("🔀 Split por fluido...")

    fluid_ids = df["fluid_id"].unique()

    train_ids, test_ids = train_test_split(
        fluid_ids, test_size=test_size, random_state=42
    )

    train_mask = df["fluid_id"].isin(train_ids)
    test_mask = df["fluid_id"].isin(test_ids)

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask]
    )




# TREINAMENTO:-------------------------------------------------------------------------------
def train_model(X_train, y_train):
    print("🌲 Treinando Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# AVALIAÇÃO:--------------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    print("📊 Avaliando modelo...")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== RESULTADOS ===")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")

    return y_pred, mae, r2


# FEATURE IMPORTANCE--------------------------------------------------------------------------------
def get_feature_importance(model, features):
    df_imp = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(df_imp)

    return df_imp


# SALVAR O MODELO + MÉTRICAS-------------------------------------------------------------------
def save_artifacts(model, metrics, df_imp, features, metadata):
    print("💾 Salvando artefatos...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/metrics", exist_ok=True)
    os.makedirs("artifacts/features", exist_ok=True)
    os.makedirs("artifacts/metadata", exist_ok=True)

    #🔹 modelo
    model_path = f"artifacts/models/concentration_model_{timestamp}.pkl"
    joblib.dump(model, model_path)

    # 🔹 métricas
    metrics_path = f"artifacts/metrics/metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 🔹 features importance
    fi_path = f"artifacts/metrics/feature_importance_{timestamp}.csv"
    df_imp.to_csv(fi_path, index=False)

    # 🔹 features usadas (ESSENCIAL)
    features_path = f"artifacts/features/concentration_model_features_{timestamp}.pkl"
    joblib.dump(features, features_path)

    # 🔹 metadata
    metadata_path = f"artifacts/metadata/metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✔ Modelo salvo em: {model_path}")
    print(f"✔ Métricas salvas em: {metrics_path}")
    print(f"✔ Features salva em: {features_path}")
    print(f"✔ Metadata salva em: {metadata_path}")


# V2:
# PIPELINE COMPLETO -----
#Função de treino
def run_training(df):
    # ========================================
    #  FEATURE ENGINEERING ML (V2)
    # ========================================

    df = df.copy()

    # o modelo RF não aceita entradas que nãos sejam número, então precisamos convertger as entradas que não se adequam:
    df["estado"] = df["estado"].map({
        "estavel": 0,
        "queda": 1,
        "transicao": 2
    })

    df["dist_interface"] = df["altura"] - df["altura_interface"]

    df = df.sort_values(["fluid_id", "altura", "tempo"])

    df["c_lag1"] = (
        df.groupby(["fluid_id", "altura"])["concentracao"]
        .shift(1)
    )

    # df = df.dropna(subset=["c_lag1"])

    # ========================================
    # PIPELINE
    # ========================================

    # debug
    print("\nCOLUNAS DISPONÍVEIS:")
    print(df.columns)
    #

    X, y, features = prepare_dataset_concentration(df)
    X_train, X_test, y_train, y_test = split_by_fluid(df, X, y)
    model = train_model(X_train, y_train)
    y_pred, mae, r2 = evaluate(model, X_test, y_test)
    df_imp = get_feature_importance(model, features)

    metrics = {
        "MAE": float(mae),
        "R2": float(r2)
    }

    metadata = {
        "n_samples": len(X),
        "n_features": len(features),
        "features": features,
        "model_type": "RandomForestRegressor",
        "target": "concentracao",
        "timestamp": datetime.now().isoformat()
    }

    save_artifacts(
        model=model,
        metrics=metrics,
        df_imp=df_imp,
        features=features,
        metadata=metadata
    )

    return model, metrics, df_imp, features, metadata


