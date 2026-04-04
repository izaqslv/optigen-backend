# api_layer/sedimentation.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List

from model_layer.model_loader import load_active_sedimentation_model
from business_layer.sedimentation.prepare_features import build_lstm_sequence

import numpy as np

router = APIRouter(prefix="/sedimentation", tags=["sedimentation"])

class SedimentationInput(BaseModel):
    ROA: float = Field(..., ge=0.0, le=100.0, description="Razão óleo/água")
    adensante: float = Field(..., ge=0.0, le=1.0, description="Presença (0) ou não (1) de adensante")
    dens_susp: float = Field(..., ge=1e-6, le=1e3, description="Densidade do fluido (g/cm3)")
    dens_solids: float = Field(..., ge=1e-6, le=1e3, description="Densidade dos sólidos presentes no fluido (g/cm3)")
    teor_solids: float = Field(..., ge=1e-6, le=1.0, description="Teor inicial de sólidos no fluido (fração)")
    dp_medio: float = Field(..., ge=1e-6, le=1e3, description="Diâmetro médio das partículas no fluido (g/cm3)")
    m: float = Field(..., ge=1e-6, le=1e3, description="Parâmetro reológico m")
    n: float = Field(..., ge=1e-6, le=1e3, description="Parâmetro reológico n")
    height: float = Field(..., ge=1e-6, le=1e3, description="Altura (cm) do ponto de medida da concentração no poço ou coluna experimental")
    times: List[float] = Field(..., description="Tempos em dias")

    class Config:
        schema_extra = {
            "example": {
                "ROA": 0,
                "adensante": 1.0,
                "dens_susp": 1.17,
                "dens_solids": 2.311,
                "teor_solids": 0.127,
                "dp_medio": 25.16,
                "m": 1.344,
                "n": 0.397,
                "height": 23.0,
                "times": [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 60, 70, 80]  # dias
            }
        }

@router.post("/predict", summary="Predict concentration profile for given fluid params")
def predict_sedimentation(data: SedimentationInput):
    """
    Recebe parâmetros do fluido e lista de tempos, e retorna curva prevista de sedimentação.\n\n
    OBS>> Preencha os dados considerando as seguintes unidades: dens_susp (g/cm³), dens_solids (g/cm³), teor_solids (fração), dp_medio D50 (μm), m (reologia), n (reologia), altura (cm), tempo (dia).
    """
    try:
        model = load_active_sedimentation_model()
    except Exception as e:
        # Modelo indisponível: devolve 503 com detalhe
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")

    outputs = []
    for t in data.times:
        X_num = np.array([[
            data.ROA,
            data.dens_susp,
            data.dens_solids,
            data.teor_solids,
            data.dp_medio,
            data.m,
            data.n,
            data.height,
            t
        ]])
        # ESCALAR (APENAS NUMÉRICAS)
        X_num_scaled = model.scaler.transform(X_num)

        # CATEGORIA (NÃO ESCALAR)
        X_cat = np.array([[data.adensante]])

        # CONCATENAR (ORDEM CORRETA)
        X_final = np.hstack([X_num_scaled, X_cat])

        # PREDIÇÃO
        # y = model.model.predict(X_final)

        try:
            y = model.model.predict(X_final)
            # normalizar formato de retorno (compatível com wrappers TF/ONNX)
            val = float(y.reshape(-1)[0]) if hasattr(y, "reshape") else float(y[0][0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        outputs.append({"time": t, "predicted_concentration": val})

    return {
        "model": getattr(model, "model_dir", None),
        "seq_len": getattr(model, "seq_len", None),
        "predictions": outputs
    }



# Criado em 17/12/2025: Criar endpoint /sedimentation/what-if
@router.post("/what-if", summary="What-if real analysis using current sedimentation model")
def sedimentation_what_if(payload: dict):
    """
    What-if analysis:
    - Recebe parâmetros base
    - Recebe variações (delta ou novos valores)
    - Retorna curva base vs curva modificada
    """

    model = load_active_sedimentation_model()

    base_params = payload["base"]
    changes = payload["changes"]
    times = payload["times"]

    # --------
    # Base case
    # --------
    base_results = []
    for t in times:
        X_base = build_lstm_sequence(
            {**base_params, "time": t},
            seq_len=getattr(model, "seq_len", 5)
        )
        y_base = float(model.predict(X_base).reshape(-1)[0])
        base_results.append({"time": t, "concentration": y_base})

    # ----------------
    # Modified case
    # ----------------
    modified_params = base_params.copy()
    modified_params.update(changes)

    modified_results = []
    for t in times:
        X_mod = build_lstm_sequence(
            {**modified_params, "time": t},
            seq_len=getattr(model, "seq_len", 5)
        )
        y_mod = float(model.predict(X_mod).reshape(-1)[0])
        modified_results.append({"time": t, "concentration": y_mod})

    return {
        "base": base_results,
        "modified": modified_results,
        "delta_summary": {
            "mean_delta": sum(
                m["concentration"] - b["concentration"]
                for b, m in zip(base_results, modified_results)
            ) / len(times)
        }
    }

