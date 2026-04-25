import pandas as pd

# Estrutura principal: pipeline
def build_dataset(measurements: pd.DataFrame, fluids_meta: pd.DataFrame):
    #     Junta os dados experimentais com propriedades dos fluidos
    # =========================
    # 1. Merge base
    # =========================
    df = measurements.merge(fluids_meta, on="fluid_id", how="left")

    # =========================
    # 2. Ordenação
    # =========================
    df = df.sort_values(by=["fluid_id", "altura", "tempo"])
    df = df.reset_index(drop=True)

    # =========================
    # 3. Feature Engineering
    # =========================
    df = add_dynamic_features(df)
    df = add_event_features(df)
    df = add_curve_features(df)

    # =========================
    # 🔥 NOVO BLOCO — REGIMES
    # =========================

    # 1. Classificar regime por grupo
    df["estado"] = None

    for (f, h), g in df.groupby(["fluid_id", "altura"]):
        estado = classify_regime(g)
        df.loc[g.index, "estado"] = estado

    # 2. Calcular altura crítica
    df_altura_critica = compute_altura_critica(df)

    # 3. Merge com dataset principal
    df = df.merge(df_altura_critica, on="fluid_id", how="left")

    # 🔥 (NOVO) gradiente global de altura
    df_grad = compute_gradiente_altura(df)
    df = df.merge(df_grad, on="fluid_id", how="left")

    # 🔥 gradiente local
    df_grad_local = compute_gradiente_local(df)

    # merge por fluido + altura
    df = df.merge(
        df_grad_local[["fluid_id", "altura", "gradiente_local"]],
        on=["fluid_id", "altura"],
        how="left"
    )

    # Perfil vertical
    df_perfil = compute_perfil_vertical(df)

    # Interface de sedimentação
    df_interface = compute_interface_sedimentacao(df)
    df = df.merge(df_interface, on="fluid_id", how="left")

    return df

#==============================================================================================================
# BLOCO 1 - dinâmica
def add_dynamic_features(df):
    df["dC_dt"] = df.groupby(["fluid_id", "altura"])["concentracao"].diff()

    df["dC_dt_smooth"] = (
        df.groupby(["fluid_id", "altura"])["dC_dt"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    df["d2C_dt2"] = df.groupby(["fluid_id", "altura"])["dC_dt"].diff()

    return df

# BLOCO 2 - eventos
def add_event_features(df):

    threshold = -0.01
    # threshold = -0.02 # mais robusto para ruído (evoluir para aqui depois - ajustar com dados)

    df["queda_brusca"] = df["dC_dt"] < threshold
    df["crescimento"] = df["dC_dt"] > 0.001
    df["plateau"] = df["dC_dt"].abs() < 0.001

    def classify_state(row):
        if row["queda_brusca"]:
            return "colapso"
        elif row["plateau"]:
            return "estavel"
        elif row["crescimento"]:
            return "crescimento"
        else:
            return "transicao"

    # df["estado"] = df.apply(classify_state, axis=1)

    return df

# BLOCO 3 - curva
def add_curve_features(df):

    t_colapso = (
        df[df["queda_brusca"]]
        .groupby(["fluid_id", "altura"])["tempo"]
        .min()
        .reset_index(name="t_colapso")
    )

    slope_min = (
        df.groupby(["fluid_id", "altura"])["dC_dt"]
        .min()
        .reset_index(name="slope_min")
    )

    c_max = (
        df.groupby(["fluid_id", "altura"])["concentracao"]
        .max()
        .reset_index(name="c_max")
    )

    features = t_colapso.merge(slope_min, on=["fluid_id", "altura"], how="left")
    features = features.merge(c_max, on=["fluid_id", "altura"], how="left")

    df = df.merge(features, on=["fluid_id", "altura"], how="left")

    return df


# Novo bloco lógico: classificr regimes automaticamente
def classify_regime(g):
    t_col = g["t_colapso"].iloc[0]

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


# Outra função auxiliar
def compute_altura_critica(df):
    resultados = []

    for fluid_id, g in df.groupby("fluid_id"):

        colapsos = g[g["estado"].isin(["colapso", "colapso_inicial"])]

        if not colapsos.empty:
            h_crit = colapsos["altura"].min()
        else:
            h_crit = None

        resultados.append({
            "fluid_id": fluid_id,
            "altura_critica": h_crit
        })

    return pd.DataFrame(resultados)

# Outra função auxiliar: GRADIENTE DE ALTURA (GLOBAL POR FLUIDO)
def compute_gradiente_altura(df) -> pd.DataFrame:
    gradientes = []

    for fluid_id, g in df.groupby("fluid_id"):

        # ordenar por altura
        g_sorted = g.sort_values("altura")

        # média da concentração por altura
        c_por_altura = g_sorted.groupby("altura")["concentracao"].mean()

        # diferença entre alturas (gradiente discreto)
        delta = c_por_altura.diff()

        # tratar NaN inicial
        delta = delta.fillna(0)

        gradientes.append({
            "fluid_id": fluid_id,
            "gradiente_altura": delta.mean()
        })

    return pd.DataFrame(gradientes)


# Outra função auxiliar: GRADIENTE LOCAL DE ALTURA
def compute_gradiente_local(df) -> pd.DataFrame:
    resultados = []

    for fluid_id, g in df.groupby("fluid_id"):

        # ordenar por altura
        g_sorted = g.sort_values("altura")

        # média da concentração por altura
        c_por_altura = g_sorted.groupby("altura")["concentracao"].mean().reset_index()

        # calcular gradiente local (diferença entre alturas consecutivas)
        c_por_altura["gradiente_local"] = c_por_altura["concentracao"].diff() / c_por_altura["altura"].diff()

        # tratar NaN inicial
        c_por_altura["gradiente_local"] = c_por_altura["gradiente_local"].fillna(0)

        c_por_altura["fluid_id"] = fluid_id

        resultados.append(c_por_altura)

    return pd.concat(resultados, ignore_index=True)


# Outra função auxiliar: Computar perfil vertical
def compute_perfil_vertical(df) -> pd.DataFrame:
    return (
        df.groupby(["fluid_id", "altura"])["concentracao"]
        .mean()
        .reset_index(name="c_media")
    )

# Outra função auxiliar: DETECTAR INTERFACE DE SEDIMENTAÇÃO
def compute_interface_sedimentacao(df):
    resultados = []

    for fluid_id, g in df.groupby("fluid_id"):

        # perfil vertical médio
        perfil = (
            g.groupby("altura")["concentracao"]
            .mean()
            .sort_index()
        )

        # gradiente vertical
        grad = perfil.diff()

        # pega maior variação (em módulo)
        if grad.abs().max() > 0:
            h_interface = grad.abs().idxmax()
        else:
            h_interface = None

        resultados.append({
            "fluid_id": fluid_id,
            "altura_interface": h_interface
        })

    return pd.DataFrame(resultados)
