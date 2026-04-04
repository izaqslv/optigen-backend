import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_layer.routes import health, profiles, metadata, auth_routes
from data.database import engine
from data.models import Base
from api_layer.routes import users


app = FastAPI(
    title="OptiGen Intelligence Service",
    version="2.0",
    description="Industrial AI Platform - Modular Architecture"
)

# 🔥 Routers
app.include_router(health.router)
app.include_router(profiles.router)
app.include_router(metadata.router)
app.include_router(auth_routes.router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Base.metadata.create_all(bind=engine)
app.include_router(users.router, prefix="/users", tags=["users"])

