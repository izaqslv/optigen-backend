from pydantic import BaseModel, Field
from typing import List
import os
from openai import OpenAI
from api_layer.security.config import load_dotenv

load_dotenv()

# 1. Definimos como deve ser uma pergunta do Quiz
class QuizQuestion(BaseModel):
    pilar: str = Field(
        description="Um dos 6 pilares: Segurança, Meio Ambiente, ABS/RH, Qualidade, Manutenção ou Operação")
    pergunta: str = Field(description="A pergunta técnica baseada na IT")
    opcoes: List[str] = Field(description="5 opções de resposta")
    resposta_correta: str = Field(description="A opção correta (exatamente como escrita nas opções)")
    justificativa: str = Field(description="Breve explicação do porquê esta é a resposta correta")


class QuizSchema(BaseModel):
    titulo_trilha: str
    questoes: List[QuizQuestion]


# 2. O Motor de Geração
class OptiGenPerformanceEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_quiz_from_it(self, it_title: str, it_content: str) -> dict:
        """
        Lê o texto de uma IT e gera um Quiz pedagógico.
        """
        prompt = f"""
        Você é um Instrutor Técnico Operacional (ITO) especialista na metodologia Alumar.
        Sua tarefa é criar um Quiz de capacitação baseado na Instrução de Trabalho (IT) abaixo.

        IT Título: {it_title}
        Conteúdo: {it_content}

        REGRAS:
        1. Gere exatamente 12 perguntas (duas para cada pilar: Segurança (Comportamento seguro), Meio Ambiente (Consciência Ambiental), ABS/RH (ferramentas e sistemas existentes unidade que estão na rotina do trabalhador, e estão associados ao sistema de gestão e RH), Qualidade (Processo e Qualidade), Manutenção e Operação).
        2. As perguntas devem ser técnicas e desafiadoras.
        3. Se o operador errar a pergunta de 'Segurança', ele deve ser alertado sobre o risco crítico.
        """

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Ou o modelo que você preferir
            messages=[
                {"role": "system", "content": "Você é um mentor de excelência operacional industrial."},
                {"role": "user", "content": prompt}
            ],
            response_format=QuizSchema,
        )

        return response.choices[0].message.parsed.dict()

    def answer_question_from_it(self, it_title: str, it_content: str, question: str) -> str:
        """
        Responde perguntas do operador baseando-se estritamente no conteúdo da IT (RAG).
        """
        prompt = f"""
        Você é um Consultor Técnico Especialista da Alumar. 
        Sua tarefa é responder a dúvida de um operador usando APENAS as informações da Instrução de Trabalho (IT) abaixo.
        
        IT: {it_title}
        CONTEÚDO DA IT:
        {it_content}
        
        PERGUNTA DO OPERADOR:
        {question}
        
        INSTRUÇÕES:
        1. Seja direto, técnico e foque em segurança.
        2. Se a informação não estiver na IT, diga educadamente que não encontrou essa informação no manual técnico.
        3. Use um tom profissional e de suporte.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro ao consultar IA: {str(e)}"
