FEATURES_V3 = [
    "tempo",
    "altura",

    # dinâmica
    "c_lag1",
    "c_lag2",
    "dc_dt",
    "dc_dh", # OBS: para voltar à V3 original, basta comentar essa linha (ver OBS no dataset_builder_v3)

    # regime
    "estado",
    "regime_interface",  # regime físico

    # interface (proxy)
    "dist_interface",  # posição relativa
    "abs_dist_interface",  # distância pura

    # propriedades físicas
    "dens_susp",
    "dens_solids",
    "teor_solids",
    "m",
    "n"
]
