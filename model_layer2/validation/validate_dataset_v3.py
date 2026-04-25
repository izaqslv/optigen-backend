import numpy as np
import pandas as pd

from model_layer.analysis.data_loader import load_and_align_data
from model_layer2.dataset.dataset_builder_v3 import build_dataset_v3


def validate_dataset():
    print("\n=== VALIDANDO DATASET V3 ===")

    measurements, fluids_meta = load_and_align_data("data/DadosSedimentation.xlsx")

    df = build_dataset_v3(measurements, fluids_meta)

    print("\n[1] Shape:", df.shape)

    # -------------------------
    # 1. NaN
    # -------------------------
    print("\n[2] Checando NaN...")
    nan_cols = df.columns[df.isnull().any()]
    if len(nan_cols) > 0:
        print("❌ NaN encontrado em:", list(nan_cols))
    else:
        print("✅ Sem NaN")

    # -------------------------
    # 2. Duplicados
    # -------------------------
    print("\n[3] Checando duplicados...")
    dup = df.duplicated(subset=["fluid_id", "altura", "tempo"]).sum()
    if dup > 0:
        print(f"❌ {dup} duplicados encontrados")
    else:
        print("✅ Sem duplicados")

    # -------------------------
    # 3. Tempo monotônico
    # -------------------------
    print("\n[4] Checando monotonicidade do tempo...")
    for (fid, h), g in df.groupby(["fluid_id", "altura"]):
        if not g["tempo"].is_monotonic_increasing:
            print(f"❌ Tempo não monotônico: fluid={fid}, altura={h}")
            return
    print("✅ Tempo OK")

    # -------------------------
    # 4. Interface física
    # -------------------------
    print("\n[5] Checando h_interface...")
    if "h_interface" not in df.columns:
        print("❌ h_interface não encontrado")
    else:
        print(df["h_interface"].describe())

    # -------------------------
    # 5. dist_interface
    # -------------------------
    print("\n[6] Checando dist_interface...")
    if "dist_interface" not in df.columns:
        print("❌ dist_interface não encontrado")
    else:
        print(df["dist_interface"].describe())

    # -------------------------
    # 6. sanity física
    # -------------------------
    print("\n[7] Checando consistência física...")
    sample = df.sample(5)

    for _, row in sample.iterrows():
        calc = row["altura"] - row["h_interface"]
        if not np.isclose(calc, row["dist_interface"], atol=1e-6):
            print("❌ Inconsistência detectada")
            return

    print("✅ Física consistente")

    print("\n🎯 DATASET VALIDADO COM SUCESSO")


if __name__ == "__main__":
    validate_dataset()