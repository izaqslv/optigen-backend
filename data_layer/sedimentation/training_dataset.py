# data_layer/sedimentation/training_dataset.py

"""
Construtor do dataset de treino do OptiGen (sedimentação).

Formato atual do JSON (usado pela API):

{
  "5": {
    "fluid_id": 5,
    "features": { ... },
    "profiles": {
      "0.5": {
        "altura": 0.5,
        "tempo": [...],
        "concentracao": [...]
      },
      "2.0": {
        "altura": 2.0,
        "tempo": [...],
        "concentracao": [...]
      }
    }
  },
  "6": { ... }
}

Este módulo LÊ esse JSON consolidado e gera um DataFrame achatado
onde cada linha representa:

    (fluid_id, height_cm, time_min, concentration, + features físicas do fluido)

Saídas:
    - data/sedimentation_training.csv
    - data/sedimentation_training.parquet  (opcional)

Ele também é tolerante a um formato alternativo "achatado":

[
  {
    "fluid_id": 5,
    "height": 0.5,
    "times": [...],
    "values": [...],
    "features": {...}
  },
  ...
]
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


# -------------------------------------------------------------------------
# Caminhos
# -------------------------------------------------------------------------

# raiz do projeto = dois níveis acima deste arquivo
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INPUT_JSON = DATA_DIR / "DadosSedimentation.json"

OUTPUT_CSV = DATA_DIR / "sedimentation_training.csv"
OUTPUT_PARQUET = DATA_DIR / "sedimentation_training.parquet"


# -------------------------------------------------------------------------
# Carregamento do JSON
# -------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Carrega o JSON bruto."""
    if not path.exists():
        raise FileNotFoundError(f"JSON de entrada não encontrado em: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# -------------------------------------------------------------------------
# Flatten – caso 1: formato achatado (lista de registros simples)
# -------------------------------------------------------------------------

def _flatten_flat_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Formato esperado aqui:

      {
        "fluid_id": 5,
        "height": 0.5,
        "times": [...],
        "values": [...],
        "features": {...}
      }

    Retorna uma lista de dicionários prontos para virar DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        try:
            fluid_id = int(rec["fluid_id"])
            height = float(rec["height"])
            times = rec.get("times")
            values = rec.get("values") or rec.get("concentration")
            features = rec.get("features", {})

            if not isinstance(times, list) or not isinstance(values, list):
                raise ValueError("Campos 'times' e 'values' precisam ser listas.")

            if len(times) != len(values):
                raise ValueError(
                    f"Registro {idx}: tamanho de 'times' ({len(times)}) "
                    f"≠ tamanho de 'values' ({len(values)})"
                )

            for t, v in zip(times, values):
                row = {
                    "fluid_id": fluid_id,
                    "height_cm": float(height),
                    "time_min": float(t),
                    "concentration": float(v),
                }
                # adiciona features do fluido
                for fk, fv in features.items():
                    row[fk] = fv

                rows.append(row)

        except Exception as e:
            raise ValueError(f"Erro ao processar registro (flat) índice {idx}: {e}") from e

    return rows


# -------------------------------------------------------------------------
# Flatten – caso 2: formato OptiGen atual (fluido → profiles)
# -------------------------------------------------------------------------

def _flatten_fluids_profiles(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Formato esperado:

    {
      "5": {
        "fluid_id": 5,
        "features": {...},
        "profiles": {
          "0.5": {
            "altura": 0.5,
            "tempo": [...],
            "concentracao": [...]
          },
          ...
        }
      },
      "6": {...}
    }
    """
    rows: List[Dict[str, Any]] = []

    for key, fluid in data.items():
        if not isinstance(fluid, dict):
            # ignora qualquer lixo inesperado
            continue

        try:
            fluid_id = int(fluid.get("fluid_id", key))
            features = fluid.get("features", {}) or {}
            profiles = fluid.get("profiles", {})

            if not isinstance(profiles, dict):
                raise ValueError("Campo 'profiles' deve ser um dicionário.")

            for h_key, prof in profiles.items():
                if not isinstance(prof, dict):
                    continue

                # altura pode vir em "altura" ou ser inferida da chave
                height = prof.get("altura", None)
                if height is None:
                    try:
                        height = float(h_key)
                    except Exception:
                        raise ValueError(
                            f"Altura não encontrada nem inferível para profile '{h_key}'"
                        )

                times = prof.get("tempo") or prof.get("times")
                values = prof.get("concentracao") or prof.get("values")

                if times is None or values is None:
                    raise ValueError(
                        f"Profile (fluid_id={fluid_id}, height={height}) sem 'tempo' ou 'concentracao'."
                    )

                if not isinstance(times, list) or not isinstance(values, list):
                    raise ValueError("'tempo' e 'concentracao' devem ser listas.")

                if len(times) != len(values):
                    raise ValueError(
                        f"Profile (fluid_id={fluid_id}, height={height}): "
                        f"len(tempo)={len(times)} ≠ len(concentracao)={len(values)}"
                    )

                for t, v in zip(times, values):
                    row = {
                        "fluid_id": fluid_id,
                        "height_cm": float(height),
                        "time_min": float(t),
                        "concentration": float(v),
                    }
                    for fk, fv in features.items():
                        row[fk] = fv

                    rows.append(row)

        except Exception as e:
            raise ValueError(
                f"Erro ao processar fluido chave='{key}' (fluid_id={fluid.get('fluid_id')}): {e}"
            ) from e

    return rows


# -------------------------------------------------------------------------
# Função geral de flatten
# -------------------------------------------------------------------------

def _build_rows_from_any(data: Any) -> List[Dict[str, Any]]:
    """
    Aceita:
      - dict no formato OptiGen (fluido → profiles)
      - dict com chave "records" (lista)
      - list de registros já achatados
    """
    # Caso 1: dict
    if isinstance(data, dict):
        # Se tiver "profiles" dentro de pelo menos um registro, assumimos formato OptiGen
        has_profiles = any(
            isinstance(v, dict) and "profiles" in v for v in data.values()
        )
        if has_profiles:
            return _flatten_fluids_profiles(data)

        # Caso "records": [...]
        if "records" in data and isinstance(data["records"], list):
            return _flatten_flat_records(data["records"])

        # Outro dict genérico: usamos os values como lista
        return _flatten_flat_records(list(data.values()))

    # Caso 2: lista
    if isinstance(data, list):
        return _flatten_flat_records(data)

    raise ValueError("Formato inesperado de JSON: esperado dict ou list.")


# -------------------------------------------------------------------------
# Construção do DataFrame final
# -------------------------------------------------------------------------

def _records_to_dataframe(data: Any) -> pd.DataFrame:
    rows = _build_rows_from_any(data)

    if not rows:
        raise ValueError("Nenhum registro válido encontrado no JSON.")

    df = pd.DataFrame(rows)

    # ordena para ficar organizado
    df.sort_values(by=["fluid_id", "height_cm", "time_min"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# -------------------------------------------------------------------------
# Função principal de construção do dataset
# -------------------------------------------------------------------------

def build_training_dataset(
    input_path: Path = INPUT_JSON,
    output_csv: Path = OUTPUT_CSV,
    output_parquet: Path = OUTPUT_PARQUET,
) -> pd.DataFrame:
    """
    Constrói o dataset de treino a partir do JSON consolidado.

    Retorna o DataFrame gerado e salva em CSV e Parquet.
    """
    print("=== OptiGen | Construção do dataset de treino (sedimentação) ===")
    print(f"📥 Lendo JSON em: {input_path}")

    raw_data = _load_json(input_path)
    df = _records_to_dataframe(raw_data)

    print(f"✅ DataFrame final: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # garante pasta data
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Salvando CSV em: {output_csv}")
    df.to_csv(output_csv, index=False)

    try:
        print(f"💾 Salvando Parquet em: {output_parquet}")
        df.to_parquet(output_parquet, index=False)
    except Exception as e:
        print(f"⚠️ Não foi possível salvar Parquet (sem impacto crítico): {e}")

    # Pequeno resumo
    n_fluids = df["fluid_id"].nunique()
    n_heights = df[["fluid_id", "height_cm"]].drop_duplicates().shape[0]

    print(f"📊 Fluids distintos: {n_fluids}")
    print(f"📊 Combinações (fluid, altura): {n_heights}")
    print("=== Fim da construção do dataset de treino ===\n")

    return df


# -------------------------------------------------------------------------
# Execução direta pelo terminal
# -------------------------------------------------------------------------

if __name__ == "__main__":
    build_training_dataset()
