from module_fluid_engineering.model_layer.inference.engine import run_autoregressive_loop
import numpy as np, pandas as pd, os, json, joblib
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from api_layer.security.dependencies import get_current_user
from module_fluid_engineering.model_layer.features.feature_list import FEATURES
from module_fluid_engineering.model_layer.analysis.analysis import analyze_simulation
from pathlib import Path


# =========================================================
# 📌 SCHEMAS
# =========================================================
class FluidInput(BaseModel):
    dens_susp: float = Field(..., example=1.2)
    dens_solids: float = Field(..., example=2.7)
    teor_solids: float = Field(..., example=0.15)
    m: float = Field(..., example=0.8)
    n: float = Field(..., example=0.6)

class SimulateRequest(BaseModel):
    fluido: FluidInput
    altura_total: float = Field(..., example=10.0)
    tempo_max: int = Field(..., example=50)
    n_alturas: int = Field(default=20)


# =========================================================
# 📌 CORE — MOTOR ESPAÇO-TEMPORAL REAL (V3)
# =========================================================
def simulate_concentration(model, input_data):
    H = input_data.altura_total
    n_h = input_data.n_alturas
    alturas = np.linspace(0, H, n_h)
    tempos = np.arange(0, input_data.tempo_max)

    results = []
    for h in alturas:
        # Criamos o grid para esta altura específica
        g = pd.DataFrame([{
            "tempo": t, "altura": h,
            "dens_susp": input_data.fluido.dens_susp,
            "dens_solids": input_data.fluido.dens_solids,
            "teor_solids": input_data.fluido.teor_solids,
            "m": input_data.fluido.m, "n": input_data.fluido.n,
            "dist_interface": H - h,  # Proxy físico original
            "dc_dh": 0.0
        } for t in tempos])

        # O motor garante que a simulação siga a mesma regra da predição
        preds = run_autoregressive_loop(
            model=model,
            features_list=FEATURES,
            group_df=g,
            initial_concentration=input_data.fluido.teor_solids
        )

        for t, cp in zip(tempos, preds):
            results.append({"tempo": int(t), "altura": float(h), "concentracao": float(cp)})

    return pd.DataFrame(results)


# =========================================================
# 📌 CARREGAMENTO DO MODELO
# =========================================================
BASE_DIR = (
        Path(__file__).resolve().parents[2]
        / "module_fluid_engineering"
        / "model_layer"
        / "artifacts"
)

RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(RUNS_DIR, "last_run.json"), "r") as f:
    run_info = json.load(f)

MODEL_PATH = os.path.join(MODELS_DIR, run_info["model"])

print("📦 Modelo carregado:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

# =========================================================
# 📌 ROUTER
# =========================================================
router = APIRouter(
    prefix="/fluid-engineering",
    tags=["Simulation"]
)

# =========================================================
# 📌 ENDPOINT
# =========================================================
@router.post("/simulate")
def simulate_endpoint(
        data: SimulateRequest,
        user: str = Depends(get_current_user)
):
    df = simulate_concentration(model, data)

    return {
        "success": True,
        "data": df.to_dict(orient="records")
    }

# Novo endpoint
@router.post("/analyze")
def analyze_endpoint(
        data: SimulateRequest,
        user: str = Depends(get_current_user)
):
    # 🔹 roda simulação (motor real)
    df = simulate_concentration(model, data)

    # 🔹 roda análise
    analysis = analyze_simulation(df)

    return {
        "success": True,
        "data": analysis
    }