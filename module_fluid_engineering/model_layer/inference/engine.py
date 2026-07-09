import pandas as pd, os

# Imports com fallback para garantir funcionamento em diferentes ambientes (Local/Render)
try:
    from module_fluid_engineering.model_layer.utils.estado import get_estado
    from module_fluid_engineering.model_layer.features.build_features import build_features
except ImportError:
    import sys

    sys.path.append(os.getcwd())
    from module_fluid_engineering.model_layer.utils.estado import get_estado
    from module_fluid_engineering.model_layer.features.build_features import build_features


def run_autoregressive_loop(model, features_list, group_df, initial_concentration):
    """
    Executa o loop autoregressivo idêntico ao original.
    Mantém a fidelidade matemática das predições temporais.
    """
    c_prev = float(initial_concentration)
    c_prev2 = c_prev

    preds = [c_prev]  # t0

    # Loop a partir de t1
    for i in range(1, len(group_df)):
        row = group_df.iloc[i]

        # 1. Estado físico (idêntico ao original)
        estado = get_estado(c_prev, c_prev2)

        # 2. Features (idêntico ao original)
        feats = build_features(row, c_prev, c_prev2, estado)

        # 3. Preparação de entrada (garantindo a ordem das colunas do modelo)
        X = pd.DataFrame([feats])[features_list]

        # 4. Predição e Clipping (conforme implementado na sua V3)
        y_hat = float(model.predict(X)[0])
        if y_hat < 0:
            y_hat = 0.0

        preds.append(y_hat)

        # 5. Atualização da memória (rollout)
        c_prev2 = c_prev
        c_prev = y_hat

    return preds
