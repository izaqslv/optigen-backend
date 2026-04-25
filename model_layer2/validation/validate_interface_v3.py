import matplotlib.pyplot as plt
import pandas as pd

from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.dataset.dataset_builder_v3 import build_dataset_v3
from model_layer2.utils.estimate_interface_height_v3 import estimate_interface_height


# 🔹 1. carregar dados
measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")

# 🔹 2. montar dataset
df = build_dataset_v3(measurements=measurements, fluids_meta=fluids_meta)

# criar dist_interface
df["dist_interface"] = df["altura"] - df["h_interface"]

# Teste1
print("\n=== TESTE h_interface ===")
print(df[["fluid_id", "tempo", "altura", "h_interface"]].head(50))

# Teste2
for t in sorted(df["tempo"].unique())[:5]:
    subset = df[df["tempo"] == t]
    print(f"\n======== Tempo = {t} ================")
    print(subset[["altura", "h_interface"]].drop_duplicates().head(10))

# Teste 3
print("\n=== STATS dist_interface ===")
print(df[["dist_interface"]].describe())

# 🔹 3. ordenar
df = df.sort_values(["fluid_id", "altura", "tempo"]).reset_index(drop=True)

print("Colunas:", df.columns.tolist())


# 🔹 4. Escolhendo fluidos e calculando interface e mostrando gráficos
for fluid_id in df["fluid_id"].unique():

    df_f = df[df["fluid_id"] == fluid_id]

    interfaces = []

    tempos = sorted(df_f["tempo"].unique())

    h_prev = None
    for t in tempos:
        df_t = df_f[df_f["tempo"] == t]

        h_interface = estimate_interface_height(df_t, h_prev)

        interfaces.append(h_interface)

        h_prev = h_interface

    plt.figure()
    plt.plot(tempos, interfaces, marker='o')
    plt.title(f"Interface estimada - Fluid {fluid_id}")
    plt.xlabel("Tempo (dia)")
    plt.ylabel("Altura da interface (cm)")
    plt.grid()

plt.show()