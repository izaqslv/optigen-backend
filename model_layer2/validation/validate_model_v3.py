import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

from sklearn.metrics import r2_score

from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.dataset.dataset_builder_v3 import build_dataset_v3
from model_layer2.inference_fun.predict_concentration_v3 import predict_concentration_v3


# ================================
# CONFIG
# ================================
PLOTS_DIR = os.path.join("model_layer2", "artifacts", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ================================
# HELPERS
# ================================
def check_nan(df, name):
    if df.isnull().any().any():
        raise ValueError(f"❌ NaNs detectados em {name}")


def validate_columns(df, required_cols):
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Coluna ausente: {col}")


# ================================
# MAIN VALIDATION
# ================================
def validate():

    print("\n🚀 INICIANDO VALIDAÇÃO DO MODELO V3\n")

    # ============================
    # 1) LOAD DATA
    # ============================
    measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")

    print("📥 Dados carregados")
    print(f"Measurements: {measurements.shape}")
    print(f"Fluids_meta: {fluids_meta.shape}")

    # ============================
    # 2) VALIDATE DATASET
    # ============================
    print("\n🔍 Validando dataset...")
    df_check = build_dataset_v3(measurements, fluids_meta)

    check_nan(df_check, "dataset_builder_v3")

    print(f"✅ Dataset válido: {df_check.shape}")

    # ============================
    # 3) RUN MODEL (ROLLOUT)
    # ============================
    print("\n🤖 Executando modelo (rollout)...")

    df_pred = predict_concentration_v3(
        measurements=measurements,
        fluids_meta=fluids_meta
    )

    print(f"📊 Resultado do modelo: {df_pred.shape}")

    # ============================
    # 4) VALIDATE OUTPUT
    # ============================
    print("\n🔎 Validando saída do modelo...")

    required_cols = [
        "concentracao",
        "pred_concentracao",
        "altura",
        "tempo",
        "fluid_id"
    ]

    validate_columns(df_pred, required_cols)
    check_nan(df_pred, "df_pred")

    print("✅ Saída do modelo válida")

    # ============================
    # 5) GLOBAL METRICS
    # ============================
    print("\n📊 MÉTRICAS GLOBAIS")

    erro = df_pred["pred_concentracao"] - df_pred["concentracao"]

    mae = np.mean(np.abs(erro))
    rmse = np.sqrt(np.mean(erro**2))
    r2 = r2_score(df_pred["concentracao"], df_pred["pred_concentracao"])

    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R2   = {r2:.6f}")

    # ============================
    # 6) METRICS BY FLUID
    # ============================
    print("\n📊 ERRO POR FLUIDO")

    for fid, df_f in df_pred.groupby("fluid_id"):
        erro_f = df_f["pred_concentracao"] - df_f["concentracao"]
        mae_f = np.mean(np.abs(erro_f))
        rmse_f = np.sqrt(np.mean(erro_f**2))

        print(f"Fluido {fid} | MAE = {mae_f:.6f} | RMSE = {rmse_f:.6f}")

    # ============================
    # 🔬 7) VALIDAÇÃO FÍSICA
    # ============================

    print("\n🔬 VALIDAÇÃO FÍSICA")

    # só executa se existir interface
    if "dist_interface" in df_pred.columns:

        df_pred["erro_abs"] = np.abs(df_pred["pred_concentracao"] - df_pred["concentracao"])

        print("\n📍 ERRO vs INTERFACE")

        bins = [-100, -5, -1, 1, 5, 100]
        labels = ["Muito abaixo", "Abaixo", "Interface", "Acima", "Muito acima"]

        df_pred["zona_interface"] = pd.cut(
            df_pred["dist_interface"],
            bins=bins,
            labels=labels
        )

        erro_por_zona = df_pred.groupby("zona_interface", observed=True)["erro_abs"].mean()

        print(erro_por_zona)

        # gráfico
        plt.figure(figsize=(6, 4))
        erro_por_zona.plot(kind="bar")
        plt.title("Erro vs Região da Interface")
        plt.ylabel("Erro absoluto médio")
        plt.xticks(rotation=30)
        plt.grid(True)

        path = os.path.join(PLOTS_DIR, "erro_vs_interface.png")
        plt.savefig(path)
        plt.close()

        # ========================
        # Interface temporal
        # ========================
        print("\n📉 COMPORTAMENTO DA INTERFACE")

        interface_por_tempo = df_pred.groupby("tempo")["dist_interface"].mean()

        plt.figure(figsize=(6, 4))
        plt.plot(interface_por_tempo.index, interface_por_tempo.values)
        plt.title("Evolução média da interface")
        plt.xlabel("Tempo")
        plt.ylabel("Distância média")
        plt.grid(True)

        path = os.path.join(PLOTS_DIR, "interface_temporal.png")
        plt.savefig(path)
        plt.close()

    else:
        print("⚠️ dist_interface não encontrada — pulando validação física de interface")

    # ============================
    # ERRO POR ESTADO
    # ============================
    if "estado" in df_pred.columns:

        print("\n⚙️ ERRO POR REGIME")

        erro_estado = df_pred.groupby("estado")["erro_abs"].mean()
        print(erro_estado)

        plt.figure(figsize=(6, 4))
        erro_estado.plot(kind="bar")
        plt.title("Erro por regime")
        plt.ylabel("Erro médio")
        plt.grid(True)

        path = os.path.join(PLOTS_DIR, "erro_por_estado.png")
        plt.savefig(path)
        plt.close()

    # ============================
    # 🧠 DIAGNÓSTICO AUTOMÁTICO
    # ============================
    print("\n🧠 DIAGNÓSTICO AUTOMÁTICO")

    if "dist_interface" in df_pred.columns:

        erro_interface = erro_por_zona.get("Interface", np.nan)
        erro_topo = erro_por_zona.get("Muito acima", np.nan)

        if erro_interface > erro_topo:
            print("✔ Interface é região mais difícil (esperado)")
        else:
            print("⚠️ Interface não está sendo bem capturada")

    if "estado" in df_pred.columns:
        if erro_estado.idxmax() == 1:
            print("✔ Regime de transição é o mais difícil (coerente)")
        else:
            print("⚠️ Regimes podem não estar bem definidos")

    # ============================
    # 8) PLOTS ORIGINAIS
    # ============================
    print("\n📈 Gerando gráficos por fluido...")

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

    for fid, df_f in df_pred.groupby("fluid_id"):

        plt.figure(figsize=(8, 5))

        alturas = sorted(df_f["altura"].unique())

        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(len(alturas))]

        height_handles = []

        for i, h in enumerate(alturas):

            g = df_f[df_f["altura"] == h].sort_values("tempo")

            marker = markers[i % len(markers)]

            plt.scatter(
                g["tempo"],
                g["concentracao"],
                color=colors[i],
                marker=marker,
                alpha=0.8
            )

            plt.plot(
                g["tempo"],
                g["pred_concentracao"],
                color=colors[i],
                linewidth=2
            )

            height_handles.append(
                Line2D(
                    [0], [0],
                    color=colors[i],
                    marker=marker,
                    linestyle='None',
                    label=f"h={h}"
                )
            )

        legend1 = plt.legend(
            handles=height_handles,
            title="Alturas",
            loc="upper left",
            bbox_to_anchor=(1.02, 1)
        )
        plt.gca().add_artist(legend1)

        type_handles = [
            Line2D([0], [0], color='black', marker='o', linestyle='None', label='Experimental'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Predito')
        ]

        plt.legend(
            handles=type_handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0)
        )

        plt.title(f"Fluido {fid}")
        plt.xlabel("Tempo")
        plt.ylabel("Concentração")
        plt.grid(True)

        path = os.path.join(PLOTS_DIR, f"fluido_{fid}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    print("✅ Gráficos salvos em:", PLOTS_DIR)

    print("\n🏁 VALIDAÇÃO FINALIZADA COM SUCESSO\n")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    validate()