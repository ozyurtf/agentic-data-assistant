FROM python:3.11-slim

WORKDIR /app

# Install system deps (needed for pymavlink, matplotlib, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy root requirements file
COPY ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
