from fastapi import APIRouter, Depends
from pydantic import BaseModel
import base64, matplotlib.pyplot as plt
from io import BytesIO
from api_layer.security.dependencies import get_current_user
from module_fluid_engineering.model_layer.inference.predictor import predict_concentration
from module_fluid_engineering.model_layer.dataset.data_loader import load_and_align_data
from core.paths import EXCEL_DATA

router = APIRouter(
    prefix="/run",
    tags=["Prediction"]
)

# 🔹 carregar UMA vez (simples e eficiente)
# measurements, fluids_meta = load_and_align_data("module_fluid_engineering/data/DadosSedimentation.xlsx")
measurements, fluids_meta = load_and_align_data(EXCEL_DATA)

class PredictRequest(BaseModel):
    fluid_id: int

@router.post("/predict")
def predict_v3(data: PredictRequest, user: str = Depends(get_current_user)):
    df = predict_concentration(measurements, fluids_meta)

    df = df[df["fluid_id"] == data.fluid_id]

    return {
        "success": True,
        "data": df.to_dict(orient="records")
    }

@router.post("/predict-plot")
def predict_plot_v3(data: PredictRequest, user: str = Depends(get_current_user)):
    df = predict_concentration(measurements, fluids_meta)
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