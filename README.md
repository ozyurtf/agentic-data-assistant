**Note**: The main motivation behind this project is to refresh my knowledge of designing scalable and robust AI systems. The original code used to build the UI is taken from [here](https://github.com/ArduPilot/UAVLogViewer), and I am implementing/integrating the components below on top of the UI:

- Backend API development (Done)
- API Gateway (Not started yet)
- 
- Chat history persistance and storage (Not started yet)
- Agentic chatbot development and integration (Done)
- File management system (In progress)
- Authentication and session management (In progress)
- Rate limiting (Done)

<div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
  <iframe width="900" height="506" 
          src="https://www.youtube.com/embed/vtJJbjGfosw" 
          frameborder="0" 
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
          allowfullscreen>
  </iframe>
</div>

### Running  

### 1. Configure Environment Variables


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
```

### 2. Create and Activate a Virtual Environment

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

### 3. Install Dependencies 

```bash
pip install -r chatbot/requirements.txt
npm install -g firecrawl-mcp
```

### 4. Run with Docker 

```bash
docker build -t <your-username>/uavlogviewer . 

docker run \
  -e VUE_APP_CESIUM_TOKEN=<your_cesium_ion_token> \
  -it -p 8080:8080 \
  -v ${PWD}:/usr/src/app \
  <your-username>/uavlogviewer
```

### 5. Start Services Locally

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