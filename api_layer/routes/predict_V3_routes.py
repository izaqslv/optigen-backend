from fastapi import APIRouter, Depends
from pydantic import BaseModel
import base64
import matplotlib.pyplot as plt
from io import BytesIO

from api_layer.security.dependencies import get_current_user
from model_layer2.inference_fun.predict_concentration_v3 import predict_concentration_v3
from model_layer.analysis.data_loader import load_and_align_data

router = APIRouter(
    prefix="/v3",
    tags=["V3 Prediction"]
)

# 🔹 carregar UMA vez (simples e eficiente)
measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")


class PredictRequest(BaseModel):
    fluid_id: int


@router.post("/predict")
def predict_v3(data: PredictRequest, user: str = Depends(get_current_user)):

    df = predict_concentration_v3(measurements, fluids_meta)

    df = df[df["fluid_id"] == data.fluid_id]

    return {
        "success": True,
        "data": df.to_dict(orient="records")
    }


@router.post("/predict-plot")
def predict_plot_v3(data: PredictRequest, user: str = Depends(get_current_user)):

    df = predict_concentration_v3(measurements, fluids_meta)
    df = df[df["fluid_id"] == data.fluid_id]

    fig, ax = plt.subplots()

    for h in sorted(df["altura"].unique()):
        df_h = df[df["altura"] == h]

        ax.plot(df_h["tempo"], df_h["pred_concentracao"], label=f"Modelo h={h}")
        ax.scatter(df_h["tempo"], df_h["concentracao"])

    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode()

    return {
        "success": True,
        "image": img_base64
    }