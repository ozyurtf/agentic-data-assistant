**Note**: The original code used to build the UI is taken from [here](https://github.com/ArduPilot/UAVLogViewer), and I am implementing/integrating the features below on top of the UI:

## Features

- Backend API development [██████████] 100% Complete
- Agentic chatbot development and integration [██████████] 100% Complete
- Session management [██████████] 100% Complete
- Rate limiting [██████████] 100% Complete
- Integration of data analytics tools executable by LLMs with result interpretation [██████████] 100% Complete
- Redis caching for extracted data [██████████] 100% Complete
- Web scraping integration with Firecrawl MCP server [██████████] 100% Complete
- Multi-service Docker orchestration [██████████] 100% Complete
- File management system [██████░░░░] 60% In Progress
- Authentication [████░░░░░░] 40% In Progress
- Chat history persistence and storage [░░░░░░░░░░] 0% Not Started
- Safety validation for LLM-generated data visualization code [░░░░░░░░░░] 0% Not Started
- MCP server for executing data visualization code [░░░░░░░░░░] 0% Not Started

## Demo 

<div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
  <iframe width="1000" height="506" 
          src="https://www.youtube.com/embed/vtJJbjGfosw" 
          style="max-width: 1000px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"
          frameborder="0" 
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
          allowfullscreen>
  </iframe>
</div>

## Running  

#### 1. Configure Environment Variables


Create a .env file in the root folder with the following values:

```env 
# Cesium 
VUE_APP_CESIUM_TOKEN=<your_cesium_ion_token>   # Get from https://ion.cesium.com/signin
VUE_APP_CESIUM_RESOURCE_ID=3

# MapTiler 
VUE_APP_MAPTILER_KEY=<your_maptiler_key>       # Get from https://docs.maptiler.com/cloud/api/authentication-key/

# OpenAI 
OPENAI_API_KEY=<your_openai_api_key>           # Get from https://platform.openai.com/api-keys

# Firecrawl
FIRECRAWL_API_KEY=<your_firecrawl_api_key>     # Get from https://www.firecrawl.dev

# Chainlit
CHAINLIT_AUTH_SECRET=<your_chainlit_secret>    # See https://docs.chainlit.io/authentication/overview

# App settings
USER_AGENT=drone-chatbot
API_BASE_URL=http://127.0.0.1:8001
VUE_APP_API_BASE_URL=http://127.0.0.1:8001     # API_BASE_URL and VUE_APP_API_BASE_URL should be the same

# Redis 
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### 2. Create and Activate a Virtual Environment

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

#### 3. Install Dependencies 

**macOS/Linux:**
```bash
brew install redis
```

**Windows:**
```bash
choco install redis-64
```

```bash
pip install -r requirements.txt

npm install -g firecrawl-mcp
```

#### 4. Start Redis Service

**macOS (one-time setup):**
```bash
brew services start redis
```

**Windows:**
```bash
# Start Redis service (if installed via Chocolatey)
redis-server

# Or run as Windows service
redis-server --service-install
redis-server --service-start
```

**Verify Redis is running:**
```bash
redis-cli ping
```

#### 5. Run with Docker 

```bash
docker build -t ui .
```

```bash
docker build -t chatbot .
``` 

```bash 
docker build -t fastapi .
```

```bash
docker compose up
```

Visit `http://localhost:8080/` to interact with the UI and chatbot.