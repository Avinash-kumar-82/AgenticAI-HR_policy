
FROM python:3.11-slim

WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy application source
COPY src/ /app/src/

# Copy HR policy documents if this folder exists
COPY uploads/ /app/uploads/

# Create writable data directory
RUN mkdir -p /app/data

# FastAPI port
EXPOSE 8080
#
# Start FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
