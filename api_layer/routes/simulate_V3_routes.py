import numpy as np
import pandas as pd
import os
import json
import joblib
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from api_layer.security.dependencies import get_current_user
from model_layer2.features.build_features_v3 import build_features_v3, compute_estado
from model_layer2.features.feature_list_v3 import FEATURES_V3
from model_layer2.analysis.analyse_v3 import analyze_simulation


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

def simulate_concentration_v3(model, input_data):

    H = input_data.altura_total
    n_h = input_data.n_alturas

    alturas = np.linspace(0, H, n_h)
    dh = alturas[1] - alturas[0]

    tempos = np.arange(0, input_data.tempo_max)

    # 🔥 estado inicial (perfil uniforme real)
    estado = np.full(n_h, input_data.fluido.teor_solids)

    results = []

    for t in tempos:

        novo_estado = np.zeros_like(estado)

        for i, h in enumerate(alturas):

            c_prev = estado[i]
            c_prev2 = estado[i]  # simplificação inicial

            # =====================================================
            # 🔥 gradiente espacial (dc_dh)
            # =====================================================
            if i == 0:
                dc_dh = (estado[i+1] - c_prev) / dh
            elif i == n_h - 1:
                dc_dh = (c_prev - estado[i-1]) / dh
            else:
                dc_dh = (estado[i+1] - estado[i-1]) / (2 * dh)

            # =====================================================
            # 🔥 estado do modelo (V3 CORRETO)
            # =====================================================
            estado_val = compute_estado(c_prev, c_prev2)

            # =====================================================
            # 🔥 interface (proxy físico simples)
            # =====================================================
            interface = H  # pode evoluir depois
            dist_interface = interface - h

            # =====================================================
            # 🔥 montar linha base
            # =====================================================
            row = {
                "tempo": t,
                "altura": h,
                "dens_susp": input_data.fluido.dens_susp,
                "dens_solids": input_data.fluido.dens_solids,
                "teor_solids": input_data.fluido.teor_solids,
                "m": input_data.fluido.m,
                "n": input_data.fluido.n,
                "dist_interface": dist_interface,
                "dc_dh": dc_dh
            }

            # =====================================================
            # 🔥 features V3 (CONSISTENTE COM TREINO)
            # =====================================================
            feat_dict = build_features_v3(
                row=row,
                c_prev=c_prev,
                c_prev2=c_prev2,
                estado=estado_val
            )

            df_feat = pd.DataFrame([feat_dict])

            # 🔒 garantir ordem correta
            X = df_feat[FEATURES_V3]

            # =====================================================
            # 🔥 predição
            # =====================================================
            c_pred = model.predict(X)[0]

            novo_estado[i] = c_pred

            results.append({
                "tempo": int(t),
                "altura": float(h),
                "concentracao": float(c_pred)
            })

        # 🔁 rollout temporal
        estado = novo_estado.copy()

    return pd.DataFrame(results)


# =========================================================
# 📌 CARREGAMENTO DO MODELO
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_layer2", "artifacts")
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
    prefix="/v3",
    tags=["V3 Simulation"]
)


# =========================================================
# 📌 ENDPOINT
# =========================================================

@router.post("/simulate")
def simulate_endpoint(
    data: SimulateRequest,
    user: str = Depends(get_current_user)
):

    df = simulate_concentration_v3(model, data)

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
    df = simulate_concentration_v3(model, data)

    # 🔹 roda análise
    analysis = analyze_simulation(df)

    return {
        "success": True,
        "data": analysis
    }