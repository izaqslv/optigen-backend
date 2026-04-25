import pandas as pd

def load_and_align_data(path_excel):

    # =========================
    # 1. Ler abas corretamente
    # =========================
    df_meas = pd.read_excel(path_excel, sheet_name="measurements")
    df_meta = pd.read_excel(path_excel, sheet_name="fluids_meta")

    # =========================
    # 2. Padronizar nomes
    # =========================
    df_meas.columns = df_meas.columns.str.strip().str.lower()
    df_meta.columns = df_meta.columns.str.strip().str.lower()

    # =========================
    # 3. Tipagem
    # =========================
    df_meas["tempo"] = pd.to_numeric(df_meas["tempo"])
    df_meas["altura"] = pd.to_numeric(df_meas["altura"])
    df_meas["fluid_id"] = pd.to_numeric(df_meas["fluid_id"])
    df_meas["concentracao"] = pd.to_numeric(df_meas["concentracao"])

    df_meta["fluid_id"] = pd.to_numeric(df_meta["fluid_id"])

    # =========================
    # 4. Merge (ESSENCIAL)
    # =========================
    df = df_meas.merge(df_meta, on="fluid_id", how="left")

    # =========================
    # 5. Ordenação
    # =========================
    df = df.sort_values(by=["fluid_id", "altura", "tempo"])

    # =========================
    # 6. RESET INDEX
    # =========================
    df = df.reset_index(drop=True)

    return df_meas, df_meta
