from typing import List
from pydantic import BaseModel
from openai import OpenAI
import PyPDF2, docx, os


class SafetyMatrix(BaseModel):
    riscos: List[str]
    controles_criticos: List[str]
    criterios_parada: List[str]
    equipamentos_ferramentas: List[str]
    epis: List[str]


class Step(BaseModel):
    passo_n: int
    o_que_fazer: str
    como_fazer: str
    por_que_fazer: str
    tempo_estimado: str
    riscos_especificos: List[str]
    medidas_controle: List[str]


class AlumarWorkInstruction(BaseModel):
    titulo: str
    objetivo: str
    local: str
    matriz_seguranca: SafetyMatrix
    fluxo_execucao: List[Step]


class OptiGenITEngine:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def transcribe_media(self, file_path: str) -> str:
        """Transcreve áudio ou vídeo usando OpenAI Whisper."""
        with open(file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text

    def extract_from_pdf(self, file_path: str) -> str:
        """Extrai texto de arquivos PDF."""
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def extract_from_docx(self, file_path: str) -> str:
        """Extrai texto de arquivos Word."""
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def process_multimodal_input(self, input_data: str, input_type: str = "text") -> AlumarWorkInstruction:
        """O Cérebro: Transforma qualquer entrada na IT estruturada da Alumar."""
        system_msg = """
        Você é o Agente de Inteligência Operacional Sênior do OptiGen para a ALUMAR.
        Sua missão é criar uma Instrução de Trabalho (IT) impecável e segura.

        REGRAS DE OURO:
        1. Identifique todos os Riscos Críticos (Prensamento, Choque, Queda, Metal Líquido).
        2. Para cada Risco, defina um Controle Crítico e um Critério de Parada.
        3. O Passo a Passo deve ser didático: O que fazer, Como fazer e Por que fazer.
        4. Use terminologia industrial técnica (ex: LOTO, Ponte ECL, Redução).
        """

        prompt = f"TIPO DE ENTRADA: {input_type}\n\nCONTEÚDO BRUTO:\n{input_data}"

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            response_format=AlumarWorkInstruction,
        )

        return response.choices[0].message.parsed

    def answer_question_from_it(self, it_title: str, it_content: str, question: str) -> str:
        """Responde a uma pergunta do usuário com base no conteúdo de uma IT específica."""
        system_msg = f"""
        Você é um Consultor Técnico Especialista em Instruções de Trabalho (ITs) da ALUMAR.
        Sua função é responder perguntas de operadores com base EXCLUSIVAMENTE no conteúdo da IT fornecida.
        Se a resposta não puder ser encontrada na IT, informe educadamente que a informação não está disponível no documento.
        Seja conciso, objetivo e use a terminologia técnica apropriada.
        """

        prompt = f"""
        IT: {it_title}
        Conteúdo da IT:
        {it_content}

        Pergunta do Operador: {question}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


