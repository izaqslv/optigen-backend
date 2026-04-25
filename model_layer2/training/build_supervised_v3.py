import pandas as pd
from model_layer2.dataset.dataset_builder_v3 import build_dataset_v3
from model_layer2.utils.estado_v3 import get_estado
from model_layer2.features.build_features_v3 import build_features_v3

def build_supervised_v3(measurements, fluids_meta):
    df = build_dataset_v3(measurements, fluids_meta)
    df = df.sort_values(["fluid_id","altura","tempo"]).reset_index(drop=True)

    X_rows, y = [], []

    for (fid, h), g in df.groupby(["fluid_id","altura"]):
        g = g.sort_values("tempo").reset_index(drop=True)

        # teacher forcing (lags reais) para base inicial
        c_prev = g.loc[0, "concentracao"]
        c_prev2 = c_prev

        for i in range(1, len(g)):
            row = g.loc[i]
            y_real = float(row["concentracao"])

            estado = get_estado(c_prev, c_prev2)
            feats = build_features_v3(row, c_prev, c_prev2, estado)

            print("FEATURES GERADAS")
            print(feats)

            # X_rows.append(feats)
            feats["fluid_id"] = fid  # 🔥 preserva identidade
            X_rows.append(feats)
            y.append(y_real)

            # atualiza com REAL (teacher forcing base)
            c_prev2 = c_prev
            c_prev = y_real

    X = pd.DataFrame(X_rows)
    y = pd.Series(y)

    print("COLUNAS FINAIS")
    print(X.columns.tolist())

    return X, y