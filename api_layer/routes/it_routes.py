from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
import os, shutil, uuid, json
from module_work_instructions.it_orchestrator import ITOrchestrator
from api_layer.security.permissions import require_module
from core.models import WorkInstruction
from sqlalchemy.orm import Session
from api_layer.security.db import get_db


router = APIRouter(
    prefix="/work-instructions",
    tags=["Work Instructions (IT)"]
)

# Diretório de uploads temporários
TEMP_UPLOAD_DIR = "module_work_instructions/data/temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Função de Dependência para garantir um Orquestrador LIMPO por requisição
# Isso elimina o "Efeito Memória" entre simulações diferentes.
def get_orchestrator():
    return ITOrchestrator(output_dir="module_work_instructions/data/generated_its")

@router.post("/generate")
async def generate_from_file(
        file: UploadFile = File(...),
        filename_prefix: str = Form("IT_Alumar"),
        token_data: dict = Depends(require_module("it_agent")),
        orchestrator: ITOrchestrator = Depends(get_orchestrator),
        db: Session = Depends(get_db)
):
    """
    Endpoint Profissional: Recebe qualquer arquivo (Áudio, Vídeo, PDF, Word)
    e retorna a IT estruturada + PDF oficial + Word editável.
    """
    # Gera um nome único para o arquivo para evitar conflitos no Windows 11
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)

    try:
        # Salva o arquivo temporariamente para processamento
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detecta o tipo de entrada para orientar a IA
        ext = file.filename.split(".")[-1].lower()
        input_type = "document" if ext in ["pdf", "docx"] else "media"

        # Orquestra o processamento completo usando a instância isolada do orquestrador
        result = orchestrator.process_and_generate(
            file_path=temp_path,
            input_type=input_type,
            filename_prefix=filename_prefix
        )

        # --- CORREÇÃO CIRÚRGICA ---
        # 1. Usamos o filename_prefix que você enviou como título
        # 2. Salvamos o JSON da IT no content_text para que o dashboard mostre a IT real
        new_it = WorkInstruction(
            user_id=token_data["user_id"],
            title=filename_prefix,
            content_text=json.dumps(result["data"]),
            pdf_url=result["pdf_url"]
        )
        db.add(new_it)
        db.commit()
        # --------------------------

        return {
            "message": "Processamento concluído com sucesso",
            "data": result["data"],
            "pdf_url": result["pdf_url"],
            "word_url": result.get("word_url", "")
        }

    except Exception as e:
        print(f"ERRO CRÍTICO NO MÓDULO IT: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Falha no processamento: {str(e)}")

    finally:
        # Limpa o arquivo temporário para manter a saúde do servidor
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/download/{filename}")
async def download_it_file(filename: str):
    """
    Endpoint de Download: Suporta tanto PDF quanto DOCX (Word).
    """
    file_path = os.path.join("data/generated_its", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")

    # Define o media_type dinamicamente com base na extensão
    ext = filename.split(".")[-1].lower()
    media_type = 'application/pdf' if ext == 'pdf' else 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )
