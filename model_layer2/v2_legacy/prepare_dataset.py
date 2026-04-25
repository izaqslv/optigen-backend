from model_layer.analysis.dataset_builder import build_dataset
def prepare_dataset(measurements, fluids_meta):
    df = build_dataset(measurements, fluids_meta)

    features = [
        "tempo",
        "altura",
        "altura_interface",
        "gradiente_altura",
        "gradiente_local",
        "c_lag1",
        "c_lag2",
        # "c_lag3",
        "dC_dt",
        "d2C_dt2",
        "dens_susp",
        "dens_solids",
        "teor_solids",
        "dp_medio",
        "m",
        "n"
    ]
    df["c_lag1"] = df.groupby(["fluid_id", "altura"])["concentracao"].shift(1)
    df["c_lag2"] = df.groupby(["fluid_id", "altura"])["concentracao"].shift(2)
    # df["c_lag3"] = df.groupby(["fluid_id", "altura"])["concentracao"].shift(3)

    df = df.dropna(subset=features + ["concentracao"])

    X = df[features]
    y = df["concentracao"]

    return X, y, features, df
