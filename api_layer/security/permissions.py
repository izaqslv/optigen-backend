from fastapi import HTTPException, Depends
from jose import jwt
from api_layer.security.config import SECRET_KEY, ALGORITHM
from api_layer.routes.auth_routes import oauth2_scheme

def require_module(module_key: str):
    def verify_module(token: str = Depends(oauth2_scheme)):
        try:
            # Usa a mesma chave e algoritmo do config.py
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_modules = payload.get("modules", [])

            if module_key not in user_modules:
                raise HTTPException(
                    status_code=403,
                    detail=f"Acesso negado: Módulo '{module_key}' não contratado."
                )
            return payload
        except Exception as e:
            # Isso ajudará a ver o erro real no console do PyCharm
            print(f"❌ Erro de Autenticação: {str(e)}")
            raise HTTPException(status_code=401, detail="Sessão expirada ou inválida.")

    return verify_module
