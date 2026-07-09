import numpy as np, pandas as pd
from module_fluid_engineering.model_layer.dataset.dataset_validator import build_dataset
from module_fluid_engineering.model_layer.utils.estado import get_estado
from module_fluid_engineering.model_layer.features.build_features import build_features
from module_fluid_engineering.model_layer.features.feature_list import FEATURES

def generate_rollout_dataset(model, measurements, fluids_meta, p_use_pred=0.7, seed=42):
    print("Gerando dataset de rollout (generating rollout dataset)...")
    rng = np.random.default_rng(seed)

    df = build_dataset(measurements, fluids_meta)
    df = df.sort_values(["fluid_id","altura","tempo"]).reset_index(drop=True)

    print("Colunas disponíveis:", df.columns.tolist())

    X_rows, y = [], []

    for (fid, h), g in df.groupby(["fluid_id","altura"]):
        g = g.sort_values("tempo").reset_index(drop=True)

        c_prev = g.loc[0, "concentracao"]
        c_prev2 = c_prev

        for i in range(1, len(g)):
            row = g.loc[i]
            y_real = float(row["concentracao"])

            estado = get_estado(c_prev, c_prev2)
            feats = build_features(row, c_prev, c_prev2, estado)

            X = pd.DataFrame([feats])[FEATURES]
            # X = pd.DataFrame([feats])
            y_hat = float(model.predict(X)[0])

            # X_rows.append(feats)
            feats["fluid_id"] = fid  # 🔥 preserva identidade
            X_rows.append(feats)
            y.append(y_real)

            # scheduled sampling:
            if rng.random() < p_use_pred:
                c_next = y_hat
            else:
                c_next = y_real

            c_prev2 = c_prev
            c_prev = c_next

    X = pd.DataFrame(X_rows)
    y = pd.Series(y)
    return X, y