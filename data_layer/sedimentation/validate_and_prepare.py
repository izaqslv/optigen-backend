# data_layer/sedimentation/validate_and_prepare.py
"""
data_layer/sedimentation/validate_and_prepare.py
Validador e preparador nível-produto para o dataset de sedimentação (OptiGen).

O que faz:
- Lê CSV ou Parquet de entrada (CSV aceita parâmetro --decimal para vírgula)
- Normaliza nomes de colunas via aliases (case-insensitive)
- Converte unidades heurísticas (height em cm -> m; opção para forçar unidades)
- Converte tipos, trata NaNs e preenche com mediana onde apropriado
- Constrói 'profile_key' (fluid_id__adensante__height)
- Ordena por profile_key + time
- Gera arquivo parquet padronizado e salva schema.json + report_json
- Fail-fast com mensagens úteis quando colunas obrigatórias faltam ou quando há NaNs críticos

Uso:
python -m data_layer.sedimentation.validate_and_prepare \
    --input data/sedimentation_training.csv \
    --output storage/datasets/sedimentation_training.parquet \
    --decimal , \
    --force-height-unit m \
    --force-time-unit min

Saídas:
- parquet salvo em --output
- schema salvo em <output_dir>/schema.json
- relatório salvo em <output_dir>/validation_report_<timestamp>.json
"""
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_and_prepare")

# ---------------------------
# Config: aliases para colunas
# ---------------------------
EXPECTED_COLUMNS = {
    "fluid_id": ["fluid_id", "fluid", "id", "fluidId", "fluido", "fluidid"],
    "ROA": ["ROA", "roa", "rotational_speed"],
    "adensante": ["adensante", "thickener", "additive", "adens"],
    "dens_susp": ["dens_susp", "density_susp", "rho_susp", "densidade_susp", "dens_susp"],
    "dens_solids": ["dens_solids", "density_solids", "rho_solid", "densidade_solidos", "dens_solids"],
    "teor_solids": ["teor_solids", "solid_fraction", "solids_frac", "teor", "teor_solidos"],
    "dp_medio": ["dp_medio", "d_mean", "particle_size", "dp_med", "dp_medio"],
    "m": ["m", "consistency_index", "consistency"],
    "n": ["n", "flow_index", "flow"],
    "height": ["height", "altura", "h", "height_cm", "alt", "altura_cm"],
    "time": ["time", "t", "tempo", "time_min", "minutes", "time_s", "seconds"],
    "concentration": ["concentration", "conc", "c", "concentracao", "concentracao"]
}

NUMERIC_COLUMNS = ["ROA","dens_susp","dens_solids","teor_solids","dp_medio","m","n","height","time","concentration"]

# ---------------------------
# Funções utilitárias
# ---------------------------
def find_column(df_cols, aliases):
    """Retorna o nome real da coluna em df_cols que case com algum alias (case-insensitive)."""
    lower_map = {c.lower(): c for c in df_cols}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None

def map_and_rename_columns(df):
    """Mapeia colunas do dataframe aos nomes padrão EXPECTED_COLUMNS."""
    mapping = {}
    missing = []
    for std_name, aliases in EXPECTED_COLUMNS.items():
        found = find_column(df.columns, aliases)
        if found:
            mapping[found] = std_name
        else:
            missing.append((std_name, aliases))
    # não renomear ainda; verificar faltantes críticos
    return mapping, missing

def enforce_schema(df, mapping, fail_on_missing=True):
    """Renomeia e verifica colunas obrigatórias."""
    df = df.rename(columns=mapping)
    required = list(EXPECTED_COLUMNS.keys())
    missing_final = [c for c in required if c not in df.columns]
    if missing_final:
        msg = f"Colunas obrigatórias faltando após mapeamento: {missing_final}"
        if fail_on_missing:
            raise KeyError(msg)
        else:
            logger.warning(msg)
    return df

def detect_and_fix_decimal_comma(path_in, decimal):
    """Leitura do CSV com parâmetro decimal apropriado."""
    if decimal is not None and decimal != '.':
        df = pd.read_csv(path_in, decimal=decimal, engine="python")
    else:
        df = pd.read_csv(path_in)
    return df

def convert_numeric_types(df):
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def heuristic_fix_units(df, force_height_unit=None, force_time_unit=None):
    """
    Heurísticas para unidades:
    - height: se mediana > 5 -> provavelmente está em cm -> divide por 100
    - time: se mediana muito grande (e.g. > 1000) sugere segundos ou ms ; usamos heurística leve
    Options: force_height_unit in ['m','cm'] ; force_time_unit in ['min','s']
    """
    if "height" in df.columns:
        med_h = float(df["height"].median(skipna=True))
        if force_height_unit:
            if force_height_unit == "cm":
                logger.info("Forçando height em cm -> convertendo para metros (/100)")
                df["height"] = df["height"].astype(float) / 100.0
            elif force_height_unit == "m":
                df["height"] = df["height"].astype(float)
        else:
            if not np.isnan(med_h) and med_h > 5:  # heurística: alturas típicas em metros < ~5
                logger.info("Heurística: mediana de height > 5. Convertendo height de cm -> m (/100).")
                df["height"] = df["height"].astype(float) / 100.0
    if "time" in df.columns:
        med_t = float(df["time"].median(skipna=True))
        if force_time_unit:
            if force_time_unit == "s":
                logger.info("Forçando time em segundos; mantendo valores.")
            elif force_time_unit == "min":
                logger.info("Forçando time em minutos; mantendo valores.")
        else:
            # heurística: se mediana muito baixa e existem fracionários, mantemos; se muito alta (>=1000) pode ser segundos->convert to min
            if not np.isnan(med_t) and med_t >= 1000:
                logger.info("Heurística: mediana de time >= 1000. Possível unidade em seconds/ms. Converta manualmente se necessário.")
    return df

def fill_missing_numeric_with_median(df):
    miss = {}
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            n_missing = int(df[c].isna().sum())
            if n_missing > 0:
                med = float(df[c].median(skipna=True))
                df[c] = df[c].fillna(med)
                miss[c] = {"missing": n_missing, "filled_with_median": med}
    return df, miss

def build_profile_key(df):
    # garante colunas existentes
    for c in ["fluid_id","adensante","height"]:
        if c not in df.columns:
            raise KeyError(f"Coluna necessária para profile_key ausente: {c}")
    df["profile_key"] = df["fluid_id"].astype(str) + "__" + df["adensante"].astype(str) + "__" + df["height"].astype(str)
    return df

def sort_by_profile_time(df):
    if "profile_key" in df.columns and "time" in df.columns:
        df = df.sort_values(["profile_key","time"]).reset_index(drop=True)
    return df

def save_schema(df, path: Path):
    schema = {col: str(df[col].dtype) for col in df.columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    logger.info("Schema salvo em: %s", path)

def save_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Relatório de validação salvo em: %s", path)

# ---------------------------
# Função principal
# ---------------------------
def validate_and_prepare(input_path: Path, output_parquet: Path,
                         decimal=None, force_height_unit=None, force_time_unit=None,
                         fail_on_missing=True, allow_partial=False):
    input_path = Path(input_path)
    output_parquet = Path(output_parquet)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    logger.info("Iniciando validação do dataset: %s", input_path)

    # 1) read
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    if input_path.suffix.lower() in [".csv", ".txt"]:
        df = detect_and_fix_decimal_comma(input_path, decimal)
    elif input_path.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Formato de entrada não suportado. Use .csv ou .parquet")

    orig_cols = list(df.columns)
    logger.info("Colunas originais detectadas: %s", orig_cols)

    # 2) detect and map aliases
    mapping, missing = map_and_rename_columns(df)
    if missing and fail_on_missing:
        # build friendly message with what wasn't found
        missing_names = [m[0] for m in missing]
        logger.warning("Algumas colunas padrão não foram encontradas entre os aliases: %s", missing_names)
        # we'll still try to rename what we found
    df = df.rename(columns=mapping)

    # 3) enforce schema (this will raise if required columns absent)
    req_cols = list(EXPECTED_COLUMNS.keys())
    absent = [c for c in req_cols if c not in df.columns]
    if absent:
        # if allow_partial, warn and proceed, else fail
        msg = f"Colunas obrigatórias ausentes após mapeamento: {absent}"
        if allow_partial:
            logger.warning(msg + " -- seguindo em modo parcial (allow_partial=True).")
        else:
            raise KeyError(msg)

    # 4) convert numeric types
    df = convert_numeric_types(df)

    # 5) unit heuristics
    df = heuristic_fix_units(df, force_height_unit=force_height_unit, force_time_unit=force_time_unit)

    # 6) build profile_key and sort
    df = build_profile_key(df) if "fluid_id" in df.columns and "adensante" in df.columns and "height" in df.columns else df
    df = sort_by_profile_time(df)

    # 7) missing numeric handling (fill medians)
    df, filled_info = fill_missing_numeric_with_median(df)

    # 8) final checks: any NaNs left in numeric columns?
    remaining_nans = {}
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            n = int(df[c].isna().sum())
            if n > 0:
                remaining_nans[c] = n
    if remaining_nans:
        msg = f"Valores numéricos ausentes restantes (após preenchimento): {remaining_nans}"
        if fail_on_missing:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # 9) summary
    report = {
        "timestamp_utc": ts,
        "input_path": str(input_path),
        "output_parquet": str(output_parquet),
        "original_columns": orig_cols,
        "renamed_columns": mapping,
        "columns_after": list(df.columns),
        "num_rows": int(len(df)),
        "num_profiles": int(df["profile_key"].nunique()) if "profile_key" in df.columns else None,
        "filled_with_median": filled_info,
        "remaining_nans": remaining_nans,
    }

    # 10) save parquet and schema & report
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    save_schema(df, output_parquet.parent / "schema.json")
    report_path = output_parquet.parent / f"validation_report_{ts}.json"
    save_report(report, report_path)

    logger.info("Validação concluída com sucesso. Parquet salvo em: %s", output_parquet)
    return df, report

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Validador e preparador do dataset de sedimentação (OptiGen).")
    p.add_argument("--input", "-i", required=True, help="Arquivo CSV (.csv) ou Parquet (.parquet) de entrada")
    p.add_argument("--output", "-o", required=True, help="Parquet padronizado de saída")
    p.add_argument("--decimal", default=None, help="Separador decimal no CSV (ex: , ou .). Use ',' se o CSV vem do Excel PT-BR")
    p.add_argument("--force-height-unit", choices=["m","cm"], default=None, help="Forçar unidade de height (m ou cm).")
    p.add_argument("--force-time-unit", choices=["min","s"], default=None, help="Forçar unidade de time (min ou s).")
    p.add_argument("--allow-partial", action="store_true", help="Permitir que colunas faltantes não causem falha (modo parcial).")
    p.add_argument("--no-fail", action="store_true", help="Não falhar em caso de NaNs remanescentes; apenas logar warnings.")
    return p.parse_args()

def main_cli():
    args = parse_args()
    try:
        df, report = validate_and_prepare(
            input_path=Path(args.input),
            output_parquet=Path(args.output),
            decimal=args.decimal,
            force_height_unit=args.force_height_unit,
            force_time_unit=args.force_time_unit,
            fail_on_missing=(not args.allow_partial and not args.no_fail),
            allow_partial=args.allow_partial
        )
        logger.info("Validador finalizou sem erros. Linhas: %d Perfis: %s", len(df), report.get("num_profiles"))
        logger.info("Relatório: %s", report)
    except Exception as e:
        logger.exception("Validação falhou: %s", e)
        sys.exit(2)

if __name__ == "__main__":
    main_cli()
