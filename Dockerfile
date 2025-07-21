FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies needed for PyTorch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # Install torch_geometric dependencies specifically
    && pip install torch_sparse torch_scatter torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.3.0+cpu.html \
    # Clean up pip cache
    && rm -rf /root/.cache/pip

COPY . .

# Ensure models directory exists even if empty initially
RUN mkdir -p /app/models \
    && mkdir -p /app/data/raw \
    && mkdir -p /app/data/processed \
    && mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.apis.main:app", "--host", "0.0.0.0", "--port", "8000"]