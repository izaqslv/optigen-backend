# 1. Usar uma imagem Python leve e estável
FROM python:3.11-slim
# 2. Definir diretório de trabalho dentro do container
WORKDIR /app

# 3. Instalar dependências do sistema necessárias para LightGBM e Pandas
# libgomp1 é essencial para o LightGBM funcionar no Linux
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar o arquivo de dependências primeiro (otimiza o cache do build)
COPY requirements.txt .

# 5. Instalar dependências (bibliotecas) do Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar todo o código do projeto para dentro do container
COPY . .

# 7. Definir o PYTHONPATH para garantir que o Python encontre os módulos api_layer e model_layer
ENV PYTHONPATH=/app

# 8. Expor a porta que o FastAPI usará (a padrão é 8000), mas a minha localmente é 8010
EXPOSE 8010

# 8. Comando para iniciar a API usando Uvicorn
# O host 0.0.0.0 é obrigatório para rodar em containers (permite que a AWS/Render acesse o container)
CMD ["uvicorn", "api_layer.main:app", "--host", "0.0.0.0", "--port", "8010"]
