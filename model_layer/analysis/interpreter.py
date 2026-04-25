from model_layer.analysis.physics_engine import generate_physics_report


def generate_report(df):
    """
    Orquestra a análise física e formata saída textual profissional
    """

    physics = generate_physics_report(df)

    df_summary = physics["summary"]
    velocity = physics["interface_velocity"]

    lines = []

    lines.append("=== RELATÓRIO FÍSICO DE SEDIMENTAÇÃO ===\n")

    for _, row in df_summary.iterrows():

        line = (
            f"Altura {row['height']} cm → "
            f"Regime: {row['regime']} | "
            f"T_clarificação: {format_value(row['t_clarification'])} dias | "
            f"T_interface: {format_value(row['interface_time'])} dias"
        )

        lines.append(line)

    lines.append("\n---")

    if velocity:
        lines.append(f"Velocidade média da interface: {velocity:.4f} cm/dia")
    else:
        lines.append("Velocidade da interface não pôde ser determinada")

    return {
        "raw": physics,
        "text": "\n".join(lines)
    }


def format_value(value):
    if value is None or str(value) == "nan":
        return "N/A"
    return f"{value:.2f}"
