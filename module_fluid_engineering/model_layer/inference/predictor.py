import os, json, joblib, pandas as pd
from module_fluid_engineering.model_layer.dataset.dataset_validator import build_dataset
from module_fluid_engineering.model_layer.inference.engine import run_autoregressive_loop
from core.paths import ARTIFACTS, MODELS, RUNS, CONFIG

# Resolução de caminhos absolutos para evitar erros no Render
BASE_DIR = ARTIFACTS
RUNS_DIR = RUNS
MODELS_DIR = MODELS
CONFIG_DIR = CONFIG

# Carregamento seguro dos artefatos
def load_artifacts():
    with open(os.path.join(RUNS_DIR, "last_run.json")) as f:
        paths = json.load(f)
    model = joblib.load(os.path.join(MODELS_DIR, paths["model"]))
    with open(os.path.join(CONFIG_DIR, paths["features"])) as f:
        features = json.load(f)
    return model, features

model, FEATURES = load_artifacts()

def predict_concentration(measurements, fluids_meta):
    df = build_dataset(measurements, fluids_meta)
    df = df.sort_values(["fluid_id", "altura", "tempo"]).reset_index(drop=True)

    out = []
    for (fid, h), g in df.groupby(["fluid_id", "altura"]):
        g = g.sort_values("tempo").reset_index(drop=True)

        # Executa a lógica centralizada
        g["pred_concentracao"] = run_autoregressive_loop(
            model=model,
            features_list=FEATURES,
            group_df=g,
            initial_concentration=g.loc[0, "concentracao"]
        )
        out.append(g)

    return pd.concat(out).reset_index(drop=True)
