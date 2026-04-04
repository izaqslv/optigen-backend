import os
import json
import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
EXCEL_PATH = os.path.join(PROJECT_ROOT, "data", "DadosSedimentation.xlsx")
JSON_PATH = os.path.join(PROJECT_ROOT, "data", "DadosSedimentation.json")


def normalize_decimal(x):
    """Converte string com vírgula para float com ponto."""
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except:
        return x


def load_excel():
    print("📘 Lendo planilha Excel...")
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Arquivo Excel não encontrado: {EXCEL_PATH}")

    df_fluids = pd.read_excel(EXCEL_PATH, sheet_name="fluids_meta")
    df_meas = pd.read_excel(EXCEL_PATH, sheet_name="measurements")

    print(f"➡️ FLUIDS: {df_fluids.shape[0]} linhas")
    print(f"➡️ MEASUREMENTS: {df_meas.shape[0]} linhas")

    return df_fluids, df_meas


def validate_structure(df_fluids, df_meas):
    required_fluids_cols = [
        "fluid_id",
        "ROA",
        "adensante",
        "dens_susp",
        "dens_solids",
        "teor_solids",
        "dp_medio",
        "m",
        "n"
    ]

    required_meas_cols = [
        "fluid_id",
        "altura",
        "tempo",
        "concentracao"
    ]

    for col in required_fluids_cols:
        if col not in df_fluids.columns:
            raise ValueError(f"Coluna ausente na aba FLUIDS: {col}")

    for col in required_meas_cols:
        if col not in df_meas.columns:
            raise ValueError(f"Coluna ausente na aba MEASUREMENTS: {col}")

    print("✔️ Estrutura validada com sucesso!")


def convert_and_merge(df_fluids, df_meas):
    print("🔄 Convertendo formatação...")

    for col in df_fluids.columns:
        df_fluids[col] = df_fluids[col].apply(normalize_decimal)

    for col in ["altura", "tempo", "concentracao"]:
        df_meas[col] = df_meas[col].apply(normalize_decimal)

    print("🔗 Mesclando dados...")

    fluids = {}
    for _, row in df_fluids.iterrows():
        fluid_id = int(row["fluid_id"])
        fluids[fluid_id] = {
            "fluid_id": fluid_id,
            "features": {
                "ROA": row["ROA"],
                "adensante": row["adensante"],
                "dens_susp": row["dens_susp"],
                "dens_solids": row["dens_solids"],
                "teor_solids": row["teor_solids"],
                "dp_medio": row["dp_medio"],
                "m": row["m"],
                "n": row["n"],
            },
            "profiles": {}
        }

    for _, row in df_meas.iterrows():
        fluid_id = int(row["fluid_id"])
        altura = float(row["altura"])
        tempo = float(row["tempo"])
        conc = float(row["concentracao"])

        if altura not in fluids[fluid_id]["profiles"]:
            fluids[fluid_id]["profiles"][altura] = {
                "altura": altura,
                "tempo": [],
                "concentracao": []
            }

        fluids[fluid_id]["profiles"][altura]["tempo"].append(tempo)
        fluids[fluid_id]["profiles"][altura]["concentracao"].append(conc)

    print("📦 Dados consolidados.")
    return fluids


def save_json(fluids):
    print(f"💾 Salvando JSON em: {JSON_PATH}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(fluids, f, indent=4, ensure_ascii=False)

    print("🎉 JSON gerado com sucesso!")


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    print("\n=== OptiGen – Excel → JSON Ingestion Tool ===\n")

    df_fluids, df_meas = load_excel()
    validate_structure(df_fluids, df_meas)
    fluids = convert_and_merge(df_fluids, df_meas)
    save_json(fluids)

    print("\n✅ Finalizado com sucesso!\n")
