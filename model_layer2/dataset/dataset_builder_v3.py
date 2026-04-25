import pandas as pd
from model_layer2.utils.estimate_interface_height_v3 import estimate_interface_height
import numpy as np

def validate_measurements(df: pd.DataFrame):
    required_cols = ["tempo", "altura", "fluid_id", "concentracao"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[measurements] Coluna ausente: {col}")

    if df.isnull().any().any():
        raise ValueError("[measurements] Existem valores nulos")

    if (df["tempo"] < 0).any():
        raise ValueError("[measurements] Tempo negativo detectado")

    if (df["altura"] < 0).any():
        raise ValueError("[measurements] Altura negativa detectada")

    if (df["concentracao"] < 0).any():
        raise ValueError("[measurements] Concentração negativa detectada")


def validate_fluids_meta(df: pd.DataFrame):
    required_cols = [
        "fluid_id",
        "dens_susp",
        "dens_solids",
        "teor_solids",
        "m",
        "n"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[fluids_meta] Coluna ausente: {col}")

    if df.isnull().any().any():
        raise ValueError("[fluids_meta] Existem valores nulos")

    if df["fluid_id"].duplicated().any():
        raise ValueError("[fluids_meta] fluid_id duplicado")


def validate_merge(df: pd.DataFrame):
    if df["dens_susp"].isnull().any():
        raise ValueError("[merge] Falha no merge: fluid_id sem correspondência")

    if df.duplicated(subset=["fluid_id", "altura", "tempo"]).any():
        raise ValueError("[merge] Dados duplicados detectados")


def validate_monotonic_time(df: pd.DataFrame):
    for (fid, h), g in df.groupby(["fluid_id", "altura"]):
        if not g["tempo"].is_monotonic_increasing:
            raise ValueError(
                f"[tempo] Série não ordenada para fluid_id={fid}, altura={h}"
            )


def build_dataset_v3(measurements: pd.DataFrame, fluids_meta: pd.DataFrame) -> pd.DataFrame:
    # =========================
    # 1. validação inicial
    # =========================
    validate_measurements(measurements)
    validate_fluids_meta(fluids_meta)

    # =========================
    # 2. merge
    # =========================
    df = measurements.merge(fluids_meta, on="fluid_id", how="left")

    # =========================
    # 3. ordenação
    # =========================
    df = df.sort_values(["fluid_id", "tempo", "altura"])

    # =========================
    # 4. validações pós-merge
    # =========================
    validate_merge(df)
    validate_monotonic_time(df)

    # ==========================
    # 5. interface física (NOVO)
    # ==========================

    df["h_interface"] = np.nan

    for (fid, t), g in df.groupby(["fluid_id", "tempo"]):
        g = g.sort_values("altura")
        h_int = estimate_interface_height(g)
        df.loc[g.index, "h_interface"] = h_int

    # 6. distância até a interface (FEATURE FÍSICA)
    df["dist_interface"] = df["altura"] - df["h_interface"]

    # ==================================================================================================================
    # 7. gradiente espacial     ## add em 21/04/2026 (para melhorar a v3 que já estava razoavelmente satisfatória)
    # ========================== OBS: para voltar à V3 original, basta deletar/comentar esse tópico 7.
    df["dc_dh"] = 0.0

    for (fid, t), g in df.groupby(["fluid_id", "tempo"]):
        g = g.sort_values("altura")

        c = g["concentracao"].values
        h = g["altura"].values

        dc_dh = np.gradient(c, h)

        df.loc[g.index, "dc_dh"] = dc_dh
    # ==================================================================================================================
    return df