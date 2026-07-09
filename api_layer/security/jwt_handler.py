from datetime import datetime, timedelta
from jose import jwt
from api_layer.security.config import SECRET_KEY, ALGORITHM

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # Aumentamos para 24h para facilitar testes

def create_access_token(user_data: dict):
    """
    Cria um token JWT que carrega as permissões do usuário.
    O 'user_data' virá do banco de dados.
    """
    to_encode = {
        "sub": user_data.get("username"),
        "user_id": str(user_data.get("id")),
        "plan": user_data.get("plan_type"),
        "modules": user_data.get("modules"),
        "company": user_data.get("company_name"),
        "full_name": user_data.get("full_name")
    }
    # Define a expiração (padrão 60 minutos)
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    # Gera a chave criptografada
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
