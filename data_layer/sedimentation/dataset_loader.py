# data_layer/sedimentation/dataset_loader.py

import os
import json

# ------------------------------------------------------------------
# Project root e caminho absoluto do JSON
# ------------------------------------------------------------------
# Este arquivo está em:
#   data_layer/sedimentation/dataset_loader.py
# Então: dirname → .../data_layer/sedimentation
#        ..        → .../data_layer
#        ..        → .../  (raiz do projeto)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH_JSON = os.path.join(PROJECT_ROOT, "data", "DadosSedimentation.json")


def try_load_dataset():
    """
    Carrega o dataset consolidado de sedimentação a partir de
    data/DadosSedimentation.json, usando caminho absoluto.

    Estrutura esperada (gerada por tools/generate_sedimentation_json.py):

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
          ...
        }
      },
      "6": { ... },
      ...
    }
    """
    if not os.path.exists(DATA_PATH_JSON):
        raise FileNotFoundError(
            f"Dataset JSON não encontrado em:\n{DATA_PATH_JSON}\n\n"
            "Gere novamente com: tools/generate_sedimentation_json.py"
        )

    with open(DATA_PATH_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("Dataset JSON vazio ou em formato inesperado.")

    return dataset
