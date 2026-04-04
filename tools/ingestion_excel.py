# NewGen_DataLayer_FullProject/api/intelligence_service/ingestion_excel.py

# ================================================================
# 📘 ingestion_excel.py
# ---------------------------------------------------------------
# Responsável por carregar e validar o conjunto de dados de
# sedimentação industrial a partir do Excel "DadosSedimentation.xlsx".
# Faz o merge entre as abas:
#   - fluids_meta (propriedades fixas por fluido)
#   - measurements (leituras de concentração por altura e tempo)
# ================================================================

import os
import pandas as pd

DATA_PATH = "data/DadosSedimentation.xlsx"

def load_excel_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo Excel não encontrado: {DATA_PATH}")

    data = pd.read_excel(DATA_PATH, sheet_name=None)

    if "fluids_meta" not in data or "measurements" not in data:
        raise RuntimeError("A planilha precisa conter abas: 'fluids_meta' e 'measurements'.")

    fluids = data["fluids_meta"]
    meas = data["measurements"]

    if "fluid_id" not in fluids.columns:
        raise RuntimeError("A aba 'fluids_meta' precisa conter a coluna 'fluid_id'.")

    if "fluid_id" not in meas.columns:
        raise RuntimeError("A aba 'measurements' precisa conter a coluna 'fluid_id'.")

    merged = meas.merge(fluids, on="fluid_id", how="left")

    if merged.isnull().sum().sum() > 0:
        print("⚠ Aviso: valores faltantes encontrados no merge. Verifique sua planilha.")

    return merged, fluids, meas


# 🔥 Carrega automaticamente ao importar
merged_data, fluids_meta, measurements = load_excel_dataset()
