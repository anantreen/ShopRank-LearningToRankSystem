FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY shoprank ./shoprank
COPY api ./api
RUN pip install --no-cache-dir -e '.[api]'
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

