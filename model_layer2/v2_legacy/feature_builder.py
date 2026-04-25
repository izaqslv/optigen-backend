import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========================
    # 1. ORDENAÇÃO (ESSENCIAL)
    # =========================
    df = df.sort_values(["fluid_id", "altura", "tempo"])

    # =========================
    # 2. DINÂMICA TEMPORAL
    # =========================
    df["dC_dt"] = df.groupby(["fluid_id", "altura"])["concentracao"].diff()

    df["dC_dt_smooth"] = (
        df.groupby(["fluid_id", "altura"])["dC_dt"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["d2C_dt2"] = df.groupby(["fluid_id", "altura"])["dC_dt"].diff()

    # histórico
    df["c_lag1"] = df.groupby(["fluid_id", "altura"])["concentracao"].shift(1)

    # =========================
    # 3. EVENTOS FÍSICOS
    # =========================
    df["queda_brusca"] = df["dC_dt"] < -0.01
    df["crescimento"] = df["dC_dt"] > 0.001
    df["plateau"] = df["dC_dt"].abs() < 0.001

    # =========================
    # 4. INTERFACE TEMPORAL
    # =========================
    df_interface = compute_interface_temporal(df)

    df = df.merge(
        df_interface,
        on=["fluid_id", "tempo"],
        how="left"
    )
    print(df.columns)

    # Limpeza crítica
    df = df.drop(columns="atltura_interface_x", errors="ignore")
    if "altura_interface_y" in df.columns:
        df = df.rename(columns={"altura_interface_y": "altura_interface"})

    # distância da interface
    df["dist_interface"] = df["altura"] - df["altura_interface"]

    # =========================
    # 5. REGIME FÍSICO (CRÍTICO)
    # =========================
    df["regime"] = np.select(
        [
            df["dist_interface"] < -1,
            df["dist_interface"].between(-1, 1),
            df["dist_interface"] > 1
        ],
        [0, 1, 2]
    )

    # =========================
    # 6. REGIME GLOBAL (ANTIGO)
    # =========================
    df["estado"] = None

    for (f, h), g in df.groupby(["fluid_id", "altura"]):
        estado = classify_regime(g)
        df.loc[g.index, "estado"] = estado

    df["estado"] = df["estado"].map({
        "colapso": 0,
        "colapso_inicial": 1,
        "estavel": 2,
        "crescimento": 3,
        "transicao": 4,
        "indefinido": 5
    })

    # =========================
    # 7. FEATURES ESPACIAIS
    # =========================
    df_grad_local = compute_gradiente_local(df)

    df = df.merge(
        df_grad_local[["fluid_id", "altura", "gradiente_local"]],
        on=["fluid_id", "altura"],
        how="left"
    )

    # =========================
    # 8. LIMPEZA FINAL
    # =========================
    df = df.dropna()

    return df


# FUNÇÕES AUXILIARES====================================================

# Interface temporal
def compute_interface_temporal(df):
    resultados = []

    for fluid_id, g in df.groupby("fluid_id"):
        for t, gt in g.groupby("tempo"):

            perfil = (
                gt.groupby("altura")["concentracao"]
                .mean()
                .sort_index()
            )

            grad = perfil.diff()

            if grad.abs().max() > 0:
                h_interface = grad.abs().idxmax()
            else:
                h_interface = None

            resultados.append({
                "fluid_id": fluid_id,
                "tempo": t,
                "altura_interface": h_interface
            })

    return pd.DataFrame(resultados)

# Regime (antigo)
def classify_regime(g):
    t_col = g["tempo"][g["queda_brusca"]].min()

    if pd.notnull(t_col):
        if t_col < 2:
            return "colapso_inicial"
        else:
            return "colapso"

    if g["plateau"].any():
        return "estavel"

    if g["crescimento"].any():
        return "crescimento"

    return "indefinido"

# Gradiente local
def compute_gradiente_local(df):
    resultados = []

    for fluid_id, g in df.groupby("fluid_id"):

        g_sorted = g.sort_values("altura")

        c_por_altura = (
            g_sorted.groupby("altura")["concentracao"]
            .mean()
            .reset_index()
        )

        c_por_altura["gradiente_local"] = (
            c_por_altura["concentracao"].diff() /
            c_por_altura["altura"].diff()
        )

        c_por_altura["gradiente_local"] = c_por_altura["gradiente_local"].fillna(0)
        c_por_altura["fluid_id"] = fluid_id

        resultados.append(c_por_altura)

    return pd.concat(resultados, ignore_index=True)
