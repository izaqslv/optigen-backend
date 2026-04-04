# business_layer/sedimentation/profiles.py

"""
Funções de alto nível para trabalhar com os perfis C(z,t)
no dataset consolidado de sedimentação.

Todas as funções assumem que o dataset vem de:
    data_layer.sedimentation.dataset_loader.try_load_dataset()
e que foi gerado por tools/generate_sedimentation_json.py.
"""

from typing import Dict, Any, List


# ------------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------------
def _get_fluid(dataset: Dict[str, Any], fluid_id: int) -> Dict[str, Any]:
    """
    Localiza um fluido no dataset, corrigindo o tipo de chave (str/int).
    No JSON, as chaves vêm como strings ("5", "6"...), então
    convertemos fluid_id → str(int(fluid_id)).
    """
    key = str(int(fluid_id))
    if key not in dataset:
        available = sorted(int(k) for k in dataset.keys())
        raise ValueError(
            f"Fluid ID {fluid_id} not found. "
            f"Available fluids: {available}"
        )
    return dataset[key]


# ------------------------------------------------------------------
# API pública
# ------------------------------------------------------------------
def list_available_fluids(dataset: Dict[str, Any]) -> List[int]:
    """Retorna lista ordenada de fluid_ids disponíveis (como inteiros)."""
    return sorted(int(fid) for fid in dataset.keys())


def list_heights_for_fluid(dataset: Dict[str, Any], fluid_id: int) -> List[float]:
    """
    Lista alturas (cm) disponíveis para um fluido.

    No JSON, as chaves de altura são strings; aqui convertemos para float.
    """
    fluid = _get_fluid(dataset, fluid_id)
    profiles = fluid.get("profiles", {})

    if not profiles:
        raise ValueError(f"No profiles found for fluid {fluid_id}.")

    return sorted(float(h) for h in profiles.keys())


def get_profile_timeseries(
    dataset: Dict[str, Any],
    fluid_id: int,
    height: float,
    show_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Retorna série temporal de concentração para um fluido e altura.

    - Faz a correção de tipos das chaves (altura string vs float).
    - Opcionalmente inclui 'metadata' com features do fluido.
    """
    fluid = _get_fluid(dataset, fluid_id)
    profiles = fluid.get("profiles", {})

    if not profiles:
        raise ValueError(f"No profiles found for fluid {fluid_id}.")

    # As chaves de altura vêm como string, ex: "8.0"
    h_key = str(float(height))
    if h_key not in profiles:
        available = sorted(float(h) for h in profiles.keys())
        raise ValueError(
            f"Height {height} not available for fluid {fluid_id}. "
            f"Available heights: {available}"
        )

    prof = profiles[h_key]

    result = {
        "fluid_id": int(fluid_id),
        "height": float(height),
        "tempo": prof["tempo"],
        "concentracao": prof["concentracao"],
    }

    if show_metadata:
        result["metadata"] = fluid.get("features", {})

    return result




def load_dataset_grouped(dataset_json):
    """
    Normaliza o dataset bruto (JSON) para um dicionário agrupado por fluido:

        {
          fluid_id: {
            "features": {...},
            "profiles": {
                altura_cm: {
                    "height": altura_cm,
                    "times": [...],
                    "values": [...],
                }
            }
          }
        }

    Aceita tanto:
      - lista de registros (formato novo do DadosSedimentation.json), quanto
      - dicionário antigo {fluid_id -> dados_do_fluido}.
    """
    grouped = {}

    # -----------------------------
    # CASO 1: dataset já é dict {fluid_id: {...}}
    # -----------------------------
    if isinstance(dataset_json, dict):
        for fid_key, fluid in dataset_json.items():
            fid = int(fid_key)
            features = fluid.get("features", {})
            profiles = fluid.get("profiles", {})

            grouped[fid] = {"features": features, "profiles": {}}

            for h_key, prof in profiles.items():
                h = float(h_key)
                times = prof.get("tempo") or prof.get("times") or []
                values = prof.get("concentracao") or prof.get("values") or []

                grouped[fid]["profiles"][h] = {
                    "height": h,
                    "times": list(times),
                    "values": list(values),
                }

        return grouped

    # -----------------------------
    # CASO 2: lista de registros (formato novo)
    # Cada item deve ter: fluid_id, height, times, values, features
    # -----------------------------
    if isinstance(dataset_json, list):
        for item in dataset_json:
            if not isinstance(item, dict):
                raise ValueError("Each record in JSON must be an object/dict.")

            fid = int(item["fluid_id"])
            h = float(item["height"])

            times = item.get("times") or item.get("tempo") or []
            values = item.get("values") or item.get("concentracao") or []
            features = item.get("features", {})

            if fid not in grouped:
                grouped[fid] = {"features": features, "profiles": {}}

            if h not in grouped[fid]["profiles"]:
                grouped[fid]["profiles"][h] = {
                    "height": h,
                    "times": [],
                    "values": [],
                }

            grouped[fid]["profiles"][h]["times"].extend(times)
            grouped[fid]["profiles"][h]["values"].extend(values)

        return grouped

    # -----------------------------
    # Qualquer outro tipo é erro
    # -----------------------------
    raise TypeError(
        f"Dataset must be a list or dict, got {type(dataset_json).__name__}"
    )

