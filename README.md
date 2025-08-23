**Note**: The original code used to build the UI is taken from [here](https://github.com/ArduPilot/UAVLogViewer), and I am implementing/integrating the components below on top of the UI:

## Features

- Backend API development **(Done)**
- Chat history persistance and storage **(Not started yet)**
- Agentic chatbot development and integration **(Done)**
- File management system **(In progress)**
- Session management **(Done)**
- Authentication **(In progress)**
- Rate limiting **(Done)**
- Redis caching for extracted data **(Done)**
- Web scraping integration with Firecrawl MCP server **(Done)**
- MCP server that allows LLMs to execute Python code **(Not started yet)**
- Code execution
- Integration of data analytics tools executable by LLMs with result interpretation **(Done)**
- Multi-service Docker orchestration **(In progress)**

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
# Cesium (required)
VUE_APP_CESIUM_TOKEN=<your_cesium_ion_token>   # Get from https://ion.cesium.com/signin
VUE_APP_CESIUM_RESOURCE_ID=3

# MapTiler (required)
VUE_APP_MAPTILER_KEY=<your_maptiler_key>       # Get from https://docs.maptiler.com/cloud/api/authentication-key/

# OpenAI (required)
OPENAI_API_KEY=<your_openai_api_key>           # Get from https://platform.openai.com/api-keys

# Firecrawl (required)
FIRECRAWL_API_KEY=<your_firecrawl_api_key>     # Get from https://www.firecrawl.dev

# Chainlit (required)
CHAINLIT_AUTH_SECRET=<your_chainlit_secret>    # See https://docs.chainlit.io/authentication/overview

# App settings
USER_AGENT=drone-chatbot
API_BASE_URL=http://127.0.0.1:8001
VUE_APP_API_BASE_URL=http://127.0.0.1:8001     # API_BASE_URL and VUE_APP_API_BASE_URL should be the same
```

#### 2. Create and Activate a Virtual Environment

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
**Windows (Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

#### 3. Install Dependencies 

```bash
# Install Redis (macOS)
brew install redis

# Install Python dependencies
pip install -r requirements.txt

# Install Firecrawl MCP
npm install -g firecrawl-mcp
```

#### 4. Start Redis Service

**macOS (one-time setup):**
```bash
brew services start redis
```

**Verify Redis is running:**
```bash
redis-cli ping
# Should return "PONG"
```

#### 5. Run with Docker 

```bash
docker build -t <your-username>/uavlogviewer . 

docker run \
  -e VUE_APP_CESIUM_TOKEN=<your_cesium_ion_token> \
  -it -p 8080:8080 \
  -v ${PWD}:/usr/src/app \
  <your-username>/uavlogviewer
```

#### 6. Start Services Locally

```bash
cd chatbot
```

```bash
# Start the Chainlit chatbot
chainlit run app.py
```

```bash
# Start the FastAPI backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Visit `http://localhost:8080/` to interact with the chatbot.