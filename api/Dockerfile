FROM python:3.13.7-slim

# Define build argument for API port
ARG API_PORT=8001
ENV API_PORT=${API_PORT}

WORKDIR /app

# Install system deps (needed for pymavlink, matplotlib, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy root requirements file
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI code
COPY . .

EXPOSE ${API_PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $API_PORT --reload"]