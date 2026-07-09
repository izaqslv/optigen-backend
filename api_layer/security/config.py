import os
from dotenv import load_dotenv

# Força o carregamento do .env
load_dotenv()

# Centraliza as configurações para evitar erro 401
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

if not SECRET_KEY:
    print("⚠️ AVISO: SECRET_KEY não encontrada no arquivo .env!")
