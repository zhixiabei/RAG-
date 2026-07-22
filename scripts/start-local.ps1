$ErrorActionPreference = "Stop"

docker compose up -d postgres qdrant minio
python -m uvicorn rag_app.main:app --app-dir backend/src --host 127.0.0.1 --port 8080

