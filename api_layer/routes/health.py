from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

current_model = None
current_model_name = None

@router.get("/")
def root():
    return {"message": "🧠 OptiGen ativo"}

@router.get("/health")
def health():
    if current_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded"}
        )
    return {"status": "ok", "model": current_model_name}