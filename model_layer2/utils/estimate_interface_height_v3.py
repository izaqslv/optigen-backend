import numpy as np

def estimate_interface_height(df_t, h_prev=None):

    df_t = df_t.sort_values("altura")

    c = df_t["concentracao"].values
    h = df_t["altura"].values

    # normalizar
    c_min, c_max = np.min(c), np.max(c)
    if c_max - c_min < 1e-6:
        return h_prev if h_prev is not None else np.nan

    c_norm = (c - c_min) / (c_max - c_min)

    target = 0.5

    # 🔥 encontrar onde cruza o target
    for i in range(len(c_norm) - 1):
        if (c_norm[i] - target) * (c_norm[i+1] - target) < 0:

            # interpolação linear
            h1, h2 = h[i], h[i+1]
            c1, c2 = c_norm[i], c_norm[i+1]

            h_interface = h1 + (target - c1) * (h2 - h1) / (c2 - c1)

            return h_interface

    return h_prev if h_prev is not None else np.nan