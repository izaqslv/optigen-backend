import numpy as np
import pandas as pd
from model_layer2.dataset.dataset_builder_v3 import build_dataset_v3
from model_layer2.utils.estado_v3 import get_estado
from model_layer2.features.build_features_v3 import build_features_v3
from model_layer2.features.feature_list_v3 import FEATURES_V3

def generate_rollout_dataset_v3(model, measurements, fluids_meta, p_use_pred=0.7, seed=42):
    print("Gerando dataset de rollout (generating rollout dataset)...")
    rng = np.random.default_rng(seed)

    df = build_dataset_v3(measurements, fluids_meta)
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
            feats = build_features_v3(row, c_prev, c_prev2, estado)

            X = pd.DataFrame([feats])[FEATURES_V3]
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