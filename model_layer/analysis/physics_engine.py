# A ENGINE vai detectar:
# * Interface de sedimentação
# * tempo de clarificação (por altura)
# * Velocidade da interface
# * Classificação de zonas

import numpy as np
import pandas as pd


# =========================================================
# 🔬 1. DETECÇÃO DE CLARIFICAÇÃO (C → ~0)
# =========================================================
def detect_clarification_time(time, concentration, threshold=0.02):
    """
    Detecta o tempo em que a concentração se aproxima de zero (região clarificada)
    """
    c0 = concentration[0]
    limit = threshold * c0

    for t, c in zip(time, concentration):
        if c <= limit:
            return t

    return None


# =========================================================
# ⚡ 2. TAXA INICIAL (dC/dt)
# =========================================================
def compute_initial_slope(time, concentration):
    if len(time) < 2:
        return 0

    return (concentration[1] - concentration[0]) / (time[1] - time[0])


# =========================================================
# 📉 3. DETECÇÃO DE QUEDA (interface)
# =========================================================
def detect_interface_drop(time, concentration):
    """
    Detecta queda abrupta → passagem da interface
    """
    gradient = np.gradient(concentration, time)

    idx = np.argmin(gradient)  # maior queda

    return {
        "time_interface": time[idx],
        "intensity": gradient[idx]
    }


# =========================================================
# 🧱 4. CLASSIFICAÇÃO DA CURVA
# =========================================================
def classify_curve(concentration, threshold=0.02):
    c0 = concentration[0]
    cf = concentration[-1]

    if cf <= threshold * c0:
        return "Clarificado"

    elif cf >= 0.8 * c0:
        return "Concentrado"

    else:
        return "Transição"


# =========================================================
# 🧠 5. ANÁLISE COMPLETA POR ALTURA
# =========================================================
def analyze_height(time, concentration, height):

    time = np.array(time)
    concentration = np.array(concentration)

    # Ordenação obrigatória (segurança)
    idx = np.argsort(time)
    time = time[idx]
    concentration = concentration[idx]

    t_clear = detect_clarification_time(time, concentration)
    slope0 = compute_initial_slope(time, concentration)
    interface = detect_interface_drop(time, concentration)
    regime = classify_curve(concentration)

    return {
        "height": height,
        "regime": regime,
        "t_clarification": t_clear,
        "initial_slope": slope0,
        "interface_time": interface["time_interface"],
        "interface_intensity": interface["intensity"]
    }


# =========================================================
# 🚀 6. ANÁLISE GLOBAL DO FLUIDO
# =========================================================
def analyze_fluid(df):

    results = []

    for h in sorted(df["altura"].unique()):

        df_h = df[df["altura"] == h]

        time = df_h["tempo"].values
        conc = df_h["concentracao_pred"].values

        res = analyze_height(time, conc, h)
        results.append(res)

    return pd.DataFrame(results)


# =========================================================
# ⚡ 7. VELOCIDADE DA INTERFACE
# =========================================================
def compute_interface_velocity(df_results):
    """
    Calcula velocidade média da interface dh/dt
    """

    df_valid = df_results.dropna(subset=["t_clarification"])

    if len(df_valid) < 2:
        return None

    h = df_valid["height"].values
    t = df_valid["t_clarification"].values

    # regressão linear simples
    coef = np.polyfit(t, h, 1)

    return coef[0]  # velocidade


# =========================================================
# 📊 8. GERAÇÃO DE RELATÓRIO FÍSICO
# =========================================================
def generate_physics_report(df):

    df_results = analyze_fluid(df)

    velocity = compute_interface_velocity(df_results)

    report = {
        "summary": df_results,
        "interface_velocity": velocity
    }

    return report
