FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY app/ app/
COPY models/ models/
COPY data/ data/
COPY ui/ ui/
COPY .env.example .env

# Expose ports: 8000 for FastAPI API, 8501 for Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

ENV PYTHONPATH=/workspace
ENV HOST=0.0.0.0
ENV PORT=8000

# Default entrypoint runs FastAPI API service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
