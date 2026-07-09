from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# 🔥 pegar do ambiente (Render ou local)
DATABASE_URL = os.getenv("DATABASE_URL")

# fallback local (caso rode no PC)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./optigen.db"

# 🔥 engine inteligente (detecta tipo)
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    # engine = create_engine(DATABASE_URL)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()