import matplotlib.pyplot as plt
import numpy as np

import os
output_dir = "model_layer/analysis/outputs/plots"
os.makedirs(output_dir, exist_ok=True)



def plot_by_fluid(df):
    """
    Gera gráficos por fluido contendo todas as alturas
    Compara concentração real vs predita
    """

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '>']
    colors = plt.cm.tab10.colors

    for fluid_id in df["fluid_id"].unique():
        df_f = df[df["fluid_id"] == fluid_id]

        plt.figure(figsize=(10, 6))

        alturas = sorted(df_f["altura"].unique())

        for i, altura in enumerate(alturas):
            df_h = df_f[df_f["altura"] == altura]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            #  Experimental (pontos)
            plt.plot(
                df_h["tempo"],
                df_h["concentracao"],
                marker=marker,
                linestyle='None',
                color=color,
                label=f"Exp - h={altura}",
                markersize=8
            )

            #  Predito (linha)
            plt.plot(
                df_h["tempo"],
                df_h["concentracao_pred"],
                linestyle='-',
                color=color,
                linewidth=2,
                label=f"Pred - h={altura}"
            )

        # RMSE global
        erro = df_f["concentracao_pred"] - df_f["concentracao"]
        rmse = np.sqrt((erro ** 2).mean())

        plt.title(f"Fluido {fluid_id} | RMSE={rmse:.4f}", fontsize=14)
        plt.xlabel(r"Tempo (dia)", fontsize=12)
        plt.ylabel(r"Concentração (v/v)", fontsize=12)

        plt.text(
            0.02, 0.95,
            "OBS: Altura h (cm)",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top'
        )

        plt.legend(ncol=2, fontsize=9)  # legenda organizada
        plt.grid(alpha=0.3)

        # Exportar (opcional)
        plt.savefig(f"{output_dir}/fluido_{fluid_id}.png", dpi=300, bbox_inches='tight')

        # mostrar todos
    plt.show()










# add em 10/04/2026
def plot_experimental_only(df):
    import matplotlib.pyplot as plt
    import os

    output_dir = "model_layer/analysis/outputs/plots_exp_only"
    os.makedirs(output_dir, exist_ok=True)

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '>']
    colors = plt.cm.tab10.colors

    for fluid_id in df["fluid_id"].unique():

        df_f = df[df["fluid_id"] == fluid_id]

        plt.figure(figsize=(10, 6))

        alturas = sorted(df_f["altura"].unique())

        for i, altura in enumerate(alturas):
            df_h = df_f[df_f["altura"] == altura]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            plt.scatter(
                df_h["tempo"],
                df_h["concentracao"],
                color=color,
                marker=marker,
                label=f"Exp - h={altura}",
                s=25  #  maior que antes
            )

        plt.title(f"Fluido {fluid_id} - SOMENTE EXPERIMENTAL", fontsize=14)
        plt.xlabel("Tempo (dia)")
        plt.ylabel("Concentração (v/v)")

        plt.legend(ncol=2, fontsize=8)
        plt.grid(alpha=0.3)

        plt.savefig(f"{output_dir}/fluido_{fluid_id}_exp.png", dpi=300)

        plt.show()
