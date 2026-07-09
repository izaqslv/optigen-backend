from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import jwt

# Importações respeitando a sua estrutura de pastas
from core.models import User
from api_layer.security.db import get_db
from api_layer.security.jwt_handler import create_access_token, SECRET_KEY, ALGORITHM
from api_layer.security.hashing import verify_password

# Define como o sistema vai extrair o token das requisições
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login-swagger")


class LoginRequest(BaseModel):
    username: str
    password: str


# Modelo de como os dados do usuário serão enviados para o frontend
class UserOut(BaseModel):
    id: str # Adicionado para retornar o user_id
    username: str
    company_name: str | None = None
    plan_type: str | None = None
    modules: list[str] | None = None
    full_name: str | None = None


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)


def authenticate_user(username: str, password: str, db: Session):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(data.username, data.password, db)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    # Passamos o dicionário completo do usuário para o token
    token = create_access_token(user.__dict__)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login-swagger")
def login_swagger(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = create_access_token(user.__dict__)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
async def read_users_me(token: str = Depends(oauth2_scheme)):
    """
    Esta é a nova rota que o seu Streamlit vai chamar para
    saber quais módulos liberar na sidebar.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return UserOut(
            id=payload.get("user_id"), # Retorna o user_id do payload (added em 14junho2026)
            username=payload.get("sub"),
            company_name=payload.get("company"),
            plan_type=payload.get("plan"),
            modules=payload.get("modules"),
            full_name=payload.get("full_name")
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Sessão expirada. Faça login novamente.")
