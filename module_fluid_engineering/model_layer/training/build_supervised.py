import pandas as pd
from module_fluid_engineering.model_layer.dataset.dataset_validator import build_dataset
from module_fluid_engineering.model_layer.utils.estado import get_estado
from module_fluid_engineering.model_layer.features.build_features import build_features

def build_supervised(measurements, fluids_meta):
    df = build_dataset(measurements, fluids_meta)
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
            feats = build_features(row, c_prev, c_prev2, estado)

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