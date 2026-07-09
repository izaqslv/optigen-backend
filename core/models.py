from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from core.database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # --- NOVAS COLUNAS (Sincronizadas com o seu Supabase) ---
    plan_type = Column(String, default="standard")
    modules = Column(JSON, default=[])
    company_name = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    full_name = Column(String, nullable=True) # Nome completo do operador para certificados


class WorkInstruction(Base):
    """Armazena o texto e metadados das ITs para uso na Academia."""
    __tablename__ = "work_instructions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    title = Column(String)
    content_text = Column(String)  # Texto bruto extraído para a IA ler
    pdf_url = Column(String)       # Caminho/URL do arquivo gerado
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    quizzes = relationship("TrainingQuiz", back_populates="it")
    results = relationship("TrainingResult", back_populates="it")


class TrainingQuiz(Base):
    """Armazena os Quizzes gerados pela IA para cada IT."""
    __tablename__ = "training_quizzes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    it_id = Column(UUID(as_uuid=True), ForeignKey("work_instructions.id"))
    questions = Column(JSON)  # Estrutura: list[dict] com perguntas e respostas
    created_at = Column(DateTime, default=datetime.utcnow)

    it = relationship("WorkInstruction", back_populates="quizzes")


class TrainingResult(Base):
    """Registra a performance do operador e a Matriz de Versatilidade."""
    __tablename__ = "training_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    it_id = Column(UUID(as_uuid=True), ForeignKey("work_instructions.id"))
    score = Column(Float)
    status = Column(String)  # "Apto" ou "Pendente"
    pillars_performance = Column(JSON)  # Notas por pilar (Segurança, Qualidade, etc)
    completed_at = Column(DateTime, default=datetime.utcnow)

    it = relationship("WorkInstruction", back_populates="results")

