import os
import json
import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
EXCEL_PATH = os.path.join(PROJECT_ROOT, "data", "DadosSedimentation_1a15.xlsx")
JSON_PATH = os.path.join(PROJECT_ROOT, "data", "DadosSedimentation_1a15.json")


def normalize_decimal(x):
    """Converte string com vírgula para float com ponto."""
    if isinstance(x, str):
        x = x.replace(",", ".")
    try:
        return float(x)
    except:
        return x


# --------------------------------------------------
# LOAD EXCEL
# --------------------------------------------------

def load_excel():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel não encontrado em: {EXCEL_PATH}")

    print("  Lendo planilha Excel...")

    df_fluids = pd.read_excel(EXCEL_PATH, sheet_name="fluids_meta")
    df_meas = pd.read_excel(EXCEL_PATH, sheet_name="measurements")

    return df_fluids, df_meas


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_structure(df_fluids, df_meas):
    req_fluids = ["fluid_id","ROA","adensante","dens_susp","dens_solids","teor_solids","dp_medio","m","n"]
    req_meas = ["fluid_id","altura","tempo","concentracao"]

    for col in req_fluids:
        if col not in df_fluids.columns:
            raise ValueError(f"Coluna ausente (fluids_meta): {col}")

    for col in req_meas:
        if col not in df_meas.columns:
            raise ValueError(f"Coluna ausente (measurements): {col}")

    print("  Estrutura validada.")


# --------------------------------------------------
# CONVERSION TO OPTIGEN FORMAT
# --------------------------------------------------

def convert_to_optigen_format(df_fluids, df_meas):

    # normaliza números
    for col in df_fluids.columns:
        df_fluids[col] = df_fluids[col].apply(normalize_decimal)

    for col in ["altura", "tempo", "concentracao"]:
        df_meas[col] = df_meas[col].apply(normalize_decimal)

    print("  Normalização concluída.")

    # estrutura final
    dataset = {}

    # cria blocos por fluido
    for _, row in df_fluids.iterrows():
        fid = int(row["fluid_id"])

        dataset[fid] = {
            "fluid_id": fid,
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

    # adiciona medições altitude × tempo × concentração
    for _, row in df_meas.iterrows():
        fid = int(row["fluid_id"])
        altura = float(row["altura"])
        tempo = float(row["tempo"])
        conc = float(row["concentracao"])

        if altura not in dataset[fid]["profiles"]:
            dataset[fid]["profiles"][altura] = {
                "altura": altura,
                "tempo": [],
                "concentracao": []
            }

        dataset[fid]["profiles"][altura]["tempo"].append(tempo)
        dataset[fid]["profiles"][altura]["concentracao"].append(conc)

    print("  Dados convertidos para o modelo OptiGen.")
    return dataset


# --------------------------------------------------
# SAVE JSON
# --------------------------------------------------

def save_json(data):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"  JSON salvo com sucesso em:\n{JSON_PATH}")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    print("\n=== OptiGen – Novo Ingestor Excel → JSON (v2) ===\n")

    df_fluids, df_meas = load_excel()
    validate_structure(df_fluids, df_meas)
    dataset = convert_to_optigen_format(df_fluids, df_meas)
    save_json(dataset)

    print("\n  Finalizado com sucesso!\n")
