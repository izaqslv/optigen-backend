"""
profiles_dataset_builder.py.
Constrói em memória o dataset utilizado pelos módulos de
visualização dos perfis experimentais.
A única fonte oficial dos dados experimentais é:
    module_fluid_engineering/data/datasets/DadosSedimentation.xlsx
Não existe mais dependência de arquivos JSON.
"""

from module_fluid_engineering.model_layer.dataset.data_loader import (load_and_align_data)
from core.paths import EXCEL_DATA

def build_profiles_dataset():
    """
    Constrói o dataset esperado por profiles.py,
    plotter.py e profiles_routes.py.
    Retorna exatamente a mesma estrutura anteriormente
    obtida a partir do JSON.
    """

    measurements, fluids_meta = load_and_align_data(EXCEL_DATA)

    dataset = {}

    for fluid_id in sorted(measurements["fluid_id"].unique()):

        df_fluid = measurements[
            measurements["fluid_id"] == fluid_id
            ].copy()

        metadata = (
            fluids_meta[
                fluids_meta["fluid_id"] == fluid_id
                ]
            .iloc[0]
            .to_dict()
        )

        profiles = {}

        for altura, grupo in df_fluid.groupby("altura"):
            grupo = grupo.sort_values("tempo")

            profiles[str(float(altura))] = {

                "altura": float(altura),

                "tempo": grupo["tempo"].tolist(),

                "concentracao":
                    grupo["concentracao"].tolist()

            }

        dataset[str(fluid_id)] = {

            "fluid_id": int(fluid_id),

            "features": metadata,

            "profiles": profiles

        }

    return dataset

