# business_layer/fluid_engineering/plotter.py

"""
Geração de gráficos (PNG) para perfis de sedimentação C(t) em uma altura fixa.
Integrado ao dataset consolidado.
A função principal é:
    generate_profile_plot_from_dataset(...)
que está alinhada com o uso em api_layer/main_bkp.py.
"""

import io, base64, matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

#=======================================================================================================================
# MAPA DE UNIDADES (global no módulo)
UNITS_MAP = {
    "dens_susp": "g/cm³",
    "dens_solids": "g/cm³",
    "teor_solids": "fração",
    "dp_medio": "µm",
    "ROA": "-",
    "adensante": "-",
    "m": "-",
    "n": "-"
}
def format_metadata_text(meta: dict) -> str:
    lines = []
    for k, v in meta.items():
        unit = UNITS_MAP.get(k, "")
        if unit:
            lines.append(f"{k}: {v} {unit}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
#=======================================================================================================================


def _resolve_profile(
        dataset: Dict[str, Any],
        fluid_id: int,
        height: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Localiza o fluido e o perfil de altura dentro do dataset,
    corrigindo tipos de chave (str/int/float).
    """
    f_key = str(int(fluid_id))
    if f_key not in dataset:
        available = sorted(int(k) for k in dataset.keys())
        raise ValueError(
            f"Fluid ID {fluid_id} not found. Available fluids: {available}"
        )

    fluid = dataset[f_key]
    profiles = fluid.get("profiles", {})

    if not profiles:
        raise ValueError(f"No profiles found for fluid {fluid_id}.")

    h_key = str(float(height))
    if h_key not in profiles:
        available_h = sorted(float(h) for h in profiles.keys())
        raise ValueError(
            f"Height {height} not available for fluid {fluid_id}. "
            f"Available heights: {available_h}"
        )

    profile = profiles[h_key]
    return fluid, profile


def generate_profile_plot_from_dataset(
        dataset: Dict[str, Any],
        fluid_id: int,
        height: float,
        return_png: bool = False,
        save_path: Optional[str] = None,
        show_metadata: bool = False,
):
    """
    Gera gráfico de concentração vs tempo para um fluido e altura.

    Parâmetros:
    - dataset: dict retornado por try_load_dataset()
    - fluid_id: id do fluido (5–10)
    - height: altura de medição (cm)
    - return_png: se True, retorna (png_bytes, metadata)
    - save_path: se informado, salva um arquivo PNG nesse caminho
    - show_metadata: se True, inclui metadata no retorno e também no gráfico (texto lateral)

    Retorno:
    - se return_png=True:
        (png_bytes: bytes, metadata: dict | None)
    - se return_png=False:
        dict com informações básicas e caminho salvo (opcional)
    """
    fluid, profile = _resolve_profile(dataset, fluid_id, height)
    meta = fluid.get("features", {})

    tempo = profile["tempo"]
    conc = profile["concentracao"]

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        ax.plot(tempo, conc, marker="o")
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("Concentration (fraction)")
        ax.set_title(f"Fluid {fluid_id} — Height {height} cm")

        if show_metadata and meta:
            txt = "\n".join([
                f"{k}: {round(v, 3) if isinstance(v, float) else v} {UNITS_MAP.get(k, '')}"
                for k, v in meta.items()
            ])
            ax.text(
                1.02, 0.5, txt,
                transform=ax.transAxes,
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        # buffer em memória
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        png_bytes = buf.getvalue()

        # salvar em disco se solicitado
        if save_path:
            with open(save_path, "wb") as f:
                f.write(png_bytes)

        if return_png:
            return png_bytes, (meta if show_metadata else None)

        # modo legado (não usado hoje, mas deixamos por segurança)
        result = {
            "fluid_id": int(fluid_id),
            "height": float(height),
            "saved_path": save_path,
        }

        if show_metadata:
            result["metadata"] = meta

        return result

    finally:
        plt.close("all")


def generate_curve_plot(tempo, concentracao):
    plt.figure(figsize=(8, 5))

    plt.plot(
        tempo,
        concentracao,
        marker='o',
        linewidth=2,
        label="Modelo IA"
    )

    plt.xlabel("Tempo (dias)")
    plt.ylabel("Concentração")
    plt.title("Perfil de Sedimentação")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()

    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return image_base64


## PLOT COMPARATIVO (RN vs EXPERIMENTAL): 05/04/2026
def generate_comparison_plot(tempo, y_pred, y_exp):
    plt.figure()

    plt.plot(tempo, y_pred, label="Predito (IA)", marker='o')
    plt.scatter(tempo, y_exp, label="Experimental", color='red')

    plt.xlabel("Tempo")
    plt.ylabel("Concentração")
    plt.title("IA vs Experimental")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")