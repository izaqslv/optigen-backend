def compute_estado(c_prev, c_prev2, threshold=1e-4):
    if c_prev is None or c_prev2 is None:
        return 0
    dc = abs(c_prev - c_prev2)
    if dc > threshold:
        return 1
    else:
        return 2
def build_features_v3(row, c_prev, c_prev2, estado):
    # dinâmica
    dc_dt = c_prev - c_prev2
    # proxy de interface
    dist_interface = row["dist_interface"]
    abs_dist_interface = abs(dist_interface)
    regime_interface = int(dist_interface > 0)
    return {
        "tempo": row["tempo"],
        "altura": row["altura"],
        "c_lag1": c_prev,
        "c_lag2": c_prev2,
        "dc_dt": dc_dt,
        "dc_dh": row["dc_dh"], # OBS: para voltar à V3 original, basta comentar essa linha (ver OBS no dataset_builder_v3)
        "estado": estado,
        "dist_interface": dist_interface,
        "abs_dist_interface": abs_dist_interface,
        "regime_interface": regime_interface,
        # propriedades físicas (SEM .get agora)
        "dens_susp": row["dens_susp"],
        "dens_solids": row["dens_solids"],
        "teor_solids": row["teor_solids"],
        "m": row["m"],
        "n": row["n"],
    }