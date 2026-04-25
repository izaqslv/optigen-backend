from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import os
import base64
import sys
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import mean_absolute_error
from model_layer.analysis.data_loader import load_and_align_data
# from model_layer.regimes.test_plot_regimes import fluid_id
# from model_layer2.v2_legacy.predict import predict_concentration

from model_layer2.inference_fun.predict_concentration_v3 import predict_concentration_v3


router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"]
)

@router.get("/predict")
def predict_concentration_endpoint():
    measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")

    df_pred = predict_concentration_v3(
        measurements=measurements,
        fluids_meta=fluids_meta
    )

    df_pred = df_pred[df_pred["fluid_id"] == fluid_id]

    return df_pred.to_dict(orient="records")


@router.get("/predict_plot")
def predict_plot(fluid_id: int):
    # =========================
    # carregar dados
    # =========================
    measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")

    # filtrar (ou não) fluido
    # measurements = measurements[measurements["fluid_id"] == fluid_id]
    measurements = measurements.sort_values(["fluid_id", "altura", "tempo"])

    print(measurements.head(20))

    # predição
    df = predict_concentration_v3(measurements, fluids_meta)
    df = df[df["fluid_id"] == fluid_id]

    # =========================
    # MÉTRICA
    # =========================
    mae = mean_absolute_error(df["concentracao"], df["pred_concentracao"])

    # =========================
    # GRÁFICO
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))

    alturas = sorted(df["altura"].unique())

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(alturas)))

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '>']
    for i, h in enumerate(alturas):
        df_h = df[df["altura"] == h].sort_values("tempo")

        marker = markers[i % len(markers)]

        #  MODELO (linha)
        ax.plot(
            df_h["tempo"],
            df_h["pred_concentracao"],
            color=colors[i],
            linewidth=2,
            label=f"Modelo (h={h})"
        )

        #  EXPERIMENTAL (pontos)
        ax.scatter(
            df_h["tempo"],
            df_h["concentracao"],
            color=colors[i],
            marker=marker,
            edgecolor="black",
            s=40,
            label=f"Exp (h={h})"
        )

    ax.set_title(f"Fluido {fluid_id} — Perfis de Concentração", fontsize=14)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Concentração")

    ax.grid(True, alpha=0.3)

    #  legenda limpa (evitar duplicação)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    #  métrica no gráfico
    ax.text(
        0.02, 0.95,
        f"MAE = {mae:.5f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top"
    )

    # =========================
    # BASE64
    # =========================
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150)
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode()

    plt.close()

    return {
        "image": img_base64,
        "mae": mae
    }



@router.get("/plots")
def generate_and_get_plots():
    """
    Executa run_analysis.py como script externo
    e retorna os gráficos gerados.

     NÃO altera código existente
     Usa pipeline atual validado
     Estrutura robusta (produção)
    """

    try:
        #  1. Executa script existente (SEM alterar nada)
        # OBS: para ver os gráficos com matplotlib: ative abaixo (até a linha indicada)
        # subprocess.run(
        #   ["python", "-m", "model_layer.analysis.run_analysis"],
        #   [sys.executable, "-m", "model_layer.analysis.run_analysis"],
        # Ou, para rodar sem ver os gráficos (idela para quando estiver no frontend), ativar anaixo:
        # cria ambiente isolado para execução:
        env=os.environ.copy()
        # força o matplotlib a rodar sem interface gráfica (HEADLESS)
        env["MPLBACKEND"] = "Agg"
        # executa o script sem abrir janelas de gráfico
        subprocess.run(
            [sys.executable, "-m", "model_layer.analysis.run_analysis"],
            check=True,
            env=env,
            # cwd=os.getcwd()
        )

        #  2. Diretório onde os gráficos são salvos
        output_dir = "model_layer/analysis/outputs/plots"

        if not os.path.exists(output_dir):
            raise Exception("Diretório de saída não encontrado.")

        plots = []

        #  3. Lê imagens geradas
        for file in os.listdir(output_dir):
            if file.endswith(".png"):
                file_path = os.path.join(output_dir, file)

                with open(file_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                plots.append({
                    "filename": file,
                    "image": img_base64
                })

        if not plots:
            raise Exception("Nenhum gráfico foi gerado.")

        return JSONResponse(content={"plots": plots})

    except subprocess.CalledProcessError:
        raise HTTPException(
            status_code=500,
            detail="Erro ao executar run_analysis.py"
        )

    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=str(e)
    #     )

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # 🔥 mostra erro real no terminal

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
