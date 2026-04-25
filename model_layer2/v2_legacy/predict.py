# ESTE CÓDIGO É UMA FUNÇÃO UNIVERSAL DE PREVISÃO DE SEDIMENTAÇÃO
import joblib
import json
import os
import pandas as pd

from model_layer.analysis.dataset_builder import build_dataset


# ================================
# CARREGAR MODELO
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

with open(os.path.join(ARTIFACTS_DIR, "last_run.json")) as f:
    paths = json.load(f)

model = joblib.load(paths["model"])

with open(paths["features"]) as f:
    FEATURES = json.load(f)


# ================================
# FUNÇÃO CORRIGIDA (SEQUENCIAL)
# ================================
def predict_concentration(measurements, fluids_meta):

    #  build completo (sem cortar ainda)
    df = build_dataset(measurements, fluids_meta)

    results = []

    #  LOOP POR FLUIDO + ALTURA
    for (fluid_id, altura), g in df.groupby(["fluid_id", "altura"]):

        g = g.sort_values("tempo").reset_index(drop=True)

        #  inicialização dos lags (usa valores reais só no começo)
        c_lag1 = g.loc[0, "concentracao"]
        c_lag2 = g.loc[0, "concentracao"]

        preds = []

        for i in range(len(g)):

            row = g.loc[i].copy()

            #  atualizar lags manualmente
            row["c_lag1"] = c_lag1
            row["c_lag2"] = c_lag2

            # montar vetor de features
            X_t = pd.DataFrame([row[FEATURES]])

            # predição
            pred = model.predict(X_t)[0]

            preds.append(pred)

            #  atualizar memória temporal
            c_lag2 = c_lag1
            c_lag1 = pred

        g["pred_concentracao"] = preds
        results.append(g)

    df_result = pd.concat(results).reset_index(drop=True)

    return df_result


# import joblib
# import json
# import os
# import pandas as pd
#
# from model_layer2.features.prepare_dataset import prepare_dataset
#
#
# # ================================
# # CARREGAR MODELO
# # ================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
#
# with open(os.path.join(ARTIFACTS_DIR, "last_run.json")) as f:
#     paths = json.load(f)
#
# model = joblib.load(paths["model"])
#
# with open(paths["features"]) as f:
#     features = json.load(f)
#
#
# # ================================
# # FUNÇÃO PRINCIPAL
# # ================================
# def predict_concentration(measurements, fluids_meta):
#     """
#     Recebe dados brutos e retorna previsão
#     """
#
#     # preparar dataset igual treino
#     X, y, features_local, df = prepare_dataset(
#         measurements=measurements,
#         fluids_meta=fluids_meta
#     )
#
#     # predição
#     y_pred = model.predict(X)
#
#     # anexar no dataframe
#     df_result = df.copy()
#     df_result["pred_concentracao"] = y_pred
#
#     return df_result
