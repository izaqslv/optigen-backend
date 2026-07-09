# core/paths.py

from pathlib import Path

# ======================================================================
# DIRETÓRIOS PRINCIPAIS
# ======================================================================

ROOT = Path(__file__).resolve().parent.parent

# ======================================================================
# MÓDULOS
# ======================================================================

FLUID_ENGINEERING = ROOT / "module_fluid_engineering"

WORK_INSTRUCTIONS = ROOT / "module_work_instructions"

KNOWLEDGE_MANAGEMENT = ROOT / "module_knowledge_management"

# ======================================================================
# FLUID ENGINEERING
# ======================================================================

FE_DATA = FLUID_ENGINEERING / "data"

FE_MODEL_LAYER = FLUID_ENGINEERING / "model_layer"

FE_BUSINESS_LAYER = FLUID_ENGINEERING / "business_layer"




# ======================================================================
# DATASETS
# ======================================================================

# DATASETS = FE_DATA
DATASETS = FE_DATA / "datasets"
UPLOADS = FE_DATA / "uploads"

# EXCEL_DATA = FE_DATA / "DadosSedimentation.xlsx"
EXCEL_DATA = DATASETS / "DadosSedimentation.xlsx"
EXCEL_DATA_1A15 = FE_DATA / "DadosSedimentation_1a15.xlsx"

# ======================================================================
# ARTIFACTS
# ======================================================================

ARTIFACTS = FE_MODEL_LAYER / "artifacts"

MODELS = ARTIFACTS / "models"

RUNS = ARTIFACTS / "runs"

METRICS = ARTIFACTS / "metrics"

FEATURES = ARTIFACTS / "features"

SHAP = ARTIFACTS / "shap"

CONFIG = ARTIFACTS / "config"

PLOTS = ARTIFACTS / "plots"

# ======================================================================
# KNOWLEDGE MANAGEMENT
# ======================================================================

KM_DATA = KNOWLEDGE_MANAGEMENT / "data"

# ======================================================================
# WORK INSTRUCTIONS
# ======================================================================

WI_DATA = WORK_INSTRUCTIONS / "data"

WI_UPLOADS = WI_DATA / "uploads"

GENERATED_ITS = WI_DATA / "generated_its"


# ======================================================================
# CREATE DIRECTORIES
# ======================================================================

for path in (
        DATASETS,
        UPLOADS,
        MODELS,
        CONFIG,
        RUNS,
        METRICS,
        FEATURES,
        SHAP,
        PLOTS,
        WI_UPLOADS,
        GENERATED_ITS,
):
    path.mkdir(parents=True, exist_ok=True)