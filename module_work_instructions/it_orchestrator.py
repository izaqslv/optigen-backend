import os, uuid
from module_work_instructions.it_optigen_multimodal_engine import OptiGenITEngine
from module_work_instructions.it_pdf_generator import it_PDFGenerator
from module_work_instructions.it_word_generator import it_WordGenerator
from core.paths import GENERATED_ITS

class ITOrchestrator:
    def __init__(self, output_dir=GENERATED_ITS):
        # Usando o nome correto da classe: OptiGenITEngine
        self.engine = OptiGenITEngine()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_and_generate(self, file_path, input_type, filename_prefix="IT_EMPRESA"):
        # 1. Extração da informação bruta dependendo do tipo de arquivo
        ext = file_path.split(".")[-1].lower()

        if ext == "pdf":
            raw_content = self.engine.extract_from_pdf(file_path)
        elif ext in ["docx", "doc"]:
            raw_content = self.engine.extract_from_docx(file_path)
        elif ext in ["mp3", "wav", "mp4", "mov", "m4a"]:
            raw_content = self.engine.transcribe_media(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

        # 2. Decodifica a informação usando a IA (método correto: process_multimodal_input)
        it_object = self.engine.process_multimodal_input(raw_content, input_type)

        # Converte o objeto Pydantic para dicionário para os geradores
        it_data = it_object.dict()

        # 3. Gera nomes únicos para os arquivos
        unique_id = uuid.uuid4().hex[:6]
        pdf_filename = f"{filename_prefix}_{unique_id}.pdf"
        word_filename = f"{filename_prefix}_{unique_id}.docx"

        pdf_path = os.path.join(self.output_dir, pdf_filename)
        word_path = os.path.join(self.output_dir, word_filename)

        # 4. Gera o PDF oficial
        pdf_gen = it_PDFGenerator(it_data, pdf_path)
        pdf_gen.generate()

        # 5. Gera o Word editável
        word_gen = it_WordGenerator(it_data, word_path)
        word_gen.generate()

        return {
            "data": it_data,
            "raw_content": raw_content,
            "pdf_filename": pdf_filename,
            "word_filename": word_filename,
            "pdf_url": f"/it/download/{pdf_filename}",
            "word_url": f"/it/download/{word_filename}"
        }