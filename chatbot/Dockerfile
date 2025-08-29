FROM python:3.13.7-slim

# Define build argument for chatbot port
ARG CHATBOT_PORT=8000
ENV CHATBOT_PORT=${CHATBOT_PORT}

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE ${CHATBOT_PORT}

CMD ["sh", "-c", "chainlit run app.py --port $CHATBOT_PORT -w"]