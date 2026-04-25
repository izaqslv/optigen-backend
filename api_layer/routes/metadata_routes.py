from fastapi import APIRouter, HTTPException

from business_layer.sedimentation.fluids_metadata import FLUIDS_METADATA

from fastapi import Depends
from api_layer.security.dependencies import get_current_user

router = APIRouter(
    prefix="/metadata",
    tags=["metadata"]
)

@router.get("/{fluid_id}")
def get_metadata(fluid_id: int, user: str = Depends(get_current_user)):
    if fluid_id not in FLUIDS_METADATA:
        raise HTTPException(404, "Fluid not found")

    return FLUIDS_METADATA[fluid_id]