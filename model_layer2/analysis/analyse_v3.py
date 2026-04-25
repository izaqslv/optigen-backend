import numpy as np
import pandas as pd


# =========================================
# 📊 PERFIL EM UM TEMPO ESPECÍFICO
# =========================================
def get_profile_at_time(df: pd.DataFrame, t: int):
    return df[df["tempo"] == t].sort_values("altura")


# =========================================
# 📈 CURVA NO TOPO
# =========================================
def get_top_curve(df: pd.DataFrame):
    h_max = df["altura"].max()
    return df[df["altura"] == h_max].sort_values("tempo")


# =========================================
# 📉 CURVA NO FUNDO
# =========================================
def get_bottom_curve(df: pd.DataFrame):
    h_min = df["altura"].min()
    return df[df["altura"] == h_min].sort_values("tempo")


# =========================================
# 🧠 ESTIMATIVA DE INTERFACE (GRADIENTE)
# =========================================
def estimate_interface(df: pd.DataFrame):
    interface = []

    tempos = sorted(df["tempo"].unique())

    for t in tempos:
        df_t = df[df["tempo"] == t].sort_values("altura")

        # gradiente da concentração
        grad = np.gradient(df_t["concentracao"].values)

        idx = np.argmax(np.abs(grad))
        h_interface = df_t.iloc[idx]["altura"]

        interface.append({
            "tempo": int(t),
            "altura_interface": float(h_interface)
        })

    return pd.DataFrame(interface)


# =========================================
# ⏱️ TEMPO DE CLAREAMENTO DO TOPO
# =========================================
def time_to_clear_top(df: pd.DataFrame, threshold: float = 0.02):
    topo = get_top_curve(df)

    for _, row in topo.iterrows():
        if row["concentracao"] < threshold:
            return int(row["tempo"])

    return None


# =========================================
# 🧪 FUNÇÃO PRINCIPAL DE ANÁLISE
# =========================================
def analyze_simulation(df: pd.DataFrame):

    perfil_t0 = get_profile_at_time(df, t=0)
    curva_topo = get_top_curve(df)
    curva_fundo = get_bottom_curve(df)
    interface = estimate_interface(df)
    tempo_clareamento = time_to_clear_top(df)

    return {
        "perfil_t0": perfil_t0.to_dict(orient="records"),
        "curva_topo": curva_topo.to_dict(orient="records"),
        "curva_fundo": curva_fundo.to_dict(orient="records"),
        "interface": interface.to_dict(orient="records"),
        "tempo_clareamento_topo": tempo_clareamento
    }