from fastapi import APIRouter, Depends, HTTPException, status
import logging
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import uuid
from core.models import WorkInstruction, TrainingQuiz, TrainingResult, User
from api_layer.security.db import get_db
from api_layer.security.permissions import require_module
from module_knowledge_management.performance_engine import OptiGenPerformanceEngine

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/knowledge-management",
    tags=["Performance & Learning"]
)
engine = OptiGenPerformanceEngine()

# --- MODELOS PYDANTIC (DEFINIDOS AQUI PARA EVITAR NameError) ---
class Question(BaseModel):
    pilar: str
    pergunta: str
    opcoes: List[str]
    resposta_correta: str
    justificativa: Optional[str] = None

class Quiz(BaseModel):
    questions: List[Question]

class IT(BaseModel):
    id: str
    title: str
    content: Optional[dict] = None

class QuizResult(BaseModel):
    it_id: str
    score: float
    pillars: Optional[Dict[str, float]] = None

# --- ENDPOINTS ---

@router.get("/it-content/{it_id}")
async def get_it_content(
    it_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_module("performance_academy"))
):
    try:
        it = db.query(WorkInstruction).filter(WorkInstruction.id == uuid.UUID(it_id)).first()
    except:
        it = db.query(WorkInstruction).filter(WorkInstruction.id == it_id).first()
        
    if not it:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="IT not found")
    import json
    import ast
    # Tenta converter content_text para dicionário usando métodos robustos
    content_data = None
    try:
        # Tenta JSON padrão
        content_data = json.loads(it.content_text)
    except:
        try:
            # Tenta converter string de dicionário Python (com aspas simples)
            content_data = ast.literal_eval(it.content_text)
        except:
            content_data = {"texto_bruto": it.content_text}
            
    return {"title": it.title, "content": content_data, "content_text": it.content_text}

@router.get("/its", response_model=List[dict])
def list_available_its(
        db: Session = Depends(get_db),
        token_data: dict = Depends(require_module("performance_academy"))
):
    """Lista todas as ITs salvas no banco."""
    # Nota: No seu models.py original não existe o campo is_approved, 
    # por isso retornamos todas as ITs para não quebrar o código.
    its = db.query(WorkInstruction).all()
    return [{"id": str(it.id), "title": it.title} for it in its]

@router.post("/chat-ia")
def chat_with_it(
    chat_data: Dict,
    db: Session = Depends(get_db),
    token_data: dict = Depends(require_module("performance_academy"))
):
    it_id = chat_data.get("it_id")
    question = chat_data.get("question")
    
    try:
        it = db.query(WorkInstruction).filter(WorkInstruction.id == uuid.UUID(it_id)).first()
    except:
        it = db.query(WorkInstruction).filter(WorkInstruction.id == it_id).first()
        
    if not it:
        raise HTTPException(status_code=404, detail="IT não encontrada")
        
    response = engine.answer_question_from_it(it.title, it.content_text, question)
    
    return {"response": response}

@router.post("/approve-it/{it_id}")
async def approve_it(
    it_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_module("performance_academy"))
):
    it = db.query(WorkInstruction).filter(WorkInstruction.id == it_id).first()
    if not it:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="IT not found")
    it.is_approved = "True"
    db.commit()
    return {"message": f"IT '{it.title}' approved successfully!"}



@router.post("/generate-quiz/{it_id}")
def generate_quiz(
        it_id: str,
        db: Session = Depends(get_db),
        token_data: dict = Depends(require_module("performance_academy"))
):
    """Gera um Quiz via IA para uma IT específica e salva no banco."""
    try:
        it = db.query(WorkInstruction).filter(WorkInstruction.id == uuid.UUID(it_id)).first()
    except:
        it = db.query(WorkInstruction).filter(WorkInstruction.id == it_id).first()
        
    if not it:
        raise HTTPException(status_code=404, detail="IT não encontrada")

    # Chama o cérebro da IA (usando o método real do motor)
    quiz_data = engine.generate_quiz_from_it(it.title, it.content_text)

    # Salva no banco (usando o modelo TrainingQuiz real)
    new_quiz = TrainingQuiz(
        it_id=it.id,
        questions=quiz_data["questoes"]
    )
    db.add(new_quiz)
    db.commit()

    return {"message": "Quiz gerado e salvo com sucesso", "quiz": quiz_data}

@router.post("/submit-result")
def submit_training_result(
        result_data: dict,
        db: Session = Depends(get_db),
        token_data: dict = Depends(require_module("performance_academy"))
):
    """Salva a nota do operador com cálculo ponderado Alumar."""
    user_id = token_data["user_id"]
    it_id = result_data["it_id"]

    # Verifica tentativas existentes
    existing_attempts = db.query(TrainingResult).filter(
        TrainingResult.user_id == uuid.UUID(str(user_id)),
        TrainingResult.it_id == uuid.UUID(str(it_id))
    ).count()

    if existing_attempts >= 2:
        raise HTTPException(status_code=403, detail="Limite de 2 tentativas atingido.")

    # --- MOTOR DE CÁLCULO PONDERADO ALUMAR ---
    weights = {
        "Comportamento Seguro": 0.30, "Operação": 0.30, "Processo e Qualidade": 0.20,
        "Consciência Ambiental": 0.10, "ABS e RH": 0.05, "Manutenção": 0.05
    }
    
    raw_pillars = result_data.get("pillars", {})
    weighted_score = 0.0
    
    for pillar, weight in weights.items():
        score_p = float(raw_pillars.get(pillar, 0.0))
        # REGRA DE ZERAMENTO ALUMAR: < 7.0 ZERA O PILAR
        if score_p < 7.0:
            score_p = 0.0
        weighted_score += (score_p * weight)
    
    final_score = round(weighted_score, 2)

    new_result = TrainingResult(
        user_id=uuid.UUID(str(user_id)),
        it_id=uuid.UUID(str(it_id)),
        score=final_score,
        status="Apto" if final_score >= 7.0 else "Pendente",
        pillars_performance=raw_pillars
    )
    db.add(new_result)
    db.commit()
    return {"message": "Resultado registrado", "score": final_score, "status": new_result.status}

@router.get("/my-results")
def get_my_results(
    db: Session = Depends(get_db),
    token_data: dict = Depends(require_module("performance_academy"))
):
    """Retorna o histórico de resultados do usuário logado."""
    user_id = token_data["user_id"]
    results = db.query(TrainingResult).filter(TrainingResult.user_id == uuid.UUID(str(user_id))).all()
    return [{"it_id": str(r.it_id), "score": r.score, "status": r.status, "pillars": r.pillars_performance} for r in results]

@router.get("/skills-matrix")
def get_skills_matrix(
    db: Session = Depends(get_db),
    token_data: dict = Depends(require_module("performance_academy"))
):
    """Retorna a matriz de versatilidade baseada no TrainingResult."""
    all_results = db.query(TrainingResult).all()
    matrix_data = []
    official_pillars = ["Comportamento Seguro", "Consciência Ambiental", "ABS e RH", "Processo e Qualidade", "Manutenção", "Operação"]
    
    user_cache = {}
    it_cache = {}

    for r in all_results:
        u_id = str(r.user_id)
        it_id = str(r.it_id)
        if u_id not in user_cache:
            user = db.query(User).filter(User.id == r.user_id).first()
            user_cache[u_id] = user.username if user else "Desconhecido"
        if it_id not in it_cache:
            it = db.query(WorkInstruction).filter(WorkInstruction.id == r.it_id).first()
            it_cache[it_id] = it.title if it else "IT Removida"

        existing_entry = next((item for item in matrix_data if item["user_id"] == u_id and item["it_id"] == it_id), None)
        
        if not existing_entry:
            new_entry = {
                "user_id": u_id, "operator_name": user_cache[u_id],
                "it_id": it_id, "it_title": it_cache[it_id],
                "pillar_scores": {p: 0.0 for p in official_pillars}
            }
            for pillar, score in r.pillars_performance.items():
                norm_p = pillar.strip().replace("&", "e")
                if norm_p in new_entry["pillar_scores"]:
                    new_entry["pillar_scores"][norm_p] = score
            matrix_data.append(new_entry)
        else:
            for pillar, score in r.pillars_performance.items():
                norm_p = pillar.strip().replace("&", "e")
                if norm_p in existing_entry["pillar_scores"]:
                    existing_entry["pillar_scores"][norm_p] = max(existing_entry["pillar_scores"][norm_p], score)

    return matrix_data
