import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_layer.routes import health_routes, profiles_routes, metadata_routes, auth_routes
from data.database import engine
from data.models import Base, User
from api_layer.routes import users_routes
from api_layer.routes.analysis_routes import router as analysis_router
from api_layer.routes.predict_V3_routes import router as V3_router
from api_layer.routes.simulate_V3_routes import router as simulate_V3_routes

app = FastAPI(
    title="OptiGen Intelligence Service",
    version="2.0",
    description="Industrial AI Platform - Modular Architecture"
)

# 🔥 Routers
app.include_router(health_routes.router)
app.include_router(profiles_routes.router)
app.include_router(metadata_routes.router)
app.include_router(auth_routes.router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Base.metadata.create_all(bind=engine)
app.include_router(users_routes.router, prefix="/users", tags=["users"])
# app.include_router(analysis_router)
app.include_router(V3_router)
app.include_router(simulate_V3_routes)



