from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from sqlalchemy.orm import Session
from data.models import User
from api_layer.security.db import get_db
from api_layer.security.jwt_handler import create_access_token
from api_layer.security.hashing import verify_password

from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str


router = APIRouter(prefix="/auth", tags=["auth"])

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
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }
