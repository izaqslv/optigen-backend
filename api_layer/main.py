from dotenv import load_dotenv
load_dotenv() # ✅ CARREGAR VARIÁVEIS DE AMBIENTE PRIMEIRO!
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use("Agg")
from api_layer.routes import health_routes, profiles_routes, metadata_routes, auth_routes, users_routes
from api_layer.routes.predict_routes import router as pred_router
from api_layer.routes.simulate_routes import router as simulate_routes
from core.database import engine, Base
from api_layer.routes.it_routes import router as it_routes
from api_layer.routes import performance_routes


app = FastAPI(
    title="OptiGen Intelligence Service",
    version="2.0",
    description="Industrial AI Platform - Modular Architecture"
)

# Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# CRIAR TABELAS
Base.metadata.create_all(bind=engine)

# Routers
app.include_router(health_routes.router)
app.include_router(profiles_routes.router)
app.include_router(metadata_routes.router)
app.include_router(auth_routes.router)
app.include_router(users_routes.router, prefix="/users", tags=["users"])
app.include_router(pred_router)
app.include_router(simulate_routes)
app.include_router(it_routes)
app.include_router(performance_routes.router)
