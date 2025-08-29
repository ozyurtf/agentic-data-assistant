**Note**: The original code used to build the UI is taken from [here](https://github.com/ArduPilot/UAVLogViewer), and I am implementing/integrating the features below on top of the UI:

# Features

- Backend API development <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Agentic chatbot development and integration <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Session management <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Rate limiting <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Integration of data analytics tools executable by LLMs with result interpretation <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Redis caching for extracted data <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Web scraping integration with Firecrawl <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Multi-service Docker orchestration <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- File management system <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Authentication <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Chat history persistence and storage <span style="color: #dc3545; font-weight: bold;">**(Not Started Yet)**</span>
- Safety validation for LLM-generated data visualization code <span style="color: #dc3545; font-weight: bold;">**(Not Started Yet)**</span>
- MCP server for executing data visualization code <span style="color: #dc3545; font-weight: bold;">**(Not Started Yet)**</span>

# Demo 

<div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
<iframe width="1000" height="506" src="https://www.youtube.com/embed/xH6kAIWTbsk?si=sdeVjQmkABcztcv4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="max-width: 1000px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"></iframe>
</div>

# Running  

## Create a Folder Inside `fastpi`

```bash
mkdir -p fastapi/files
```

## Configure Environment Variables

Create a `.env` file in the root folder with the following values. The environment variables will be automatically loaded when you run the development server:

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
CHAINLIT_AUTH_SECRET=<your_chainlit_secret>    # Get from https://docs.chainlit.io/authentication/overview

# Set the maximum file size allowed for uploading
MAX_FILE_SIZE_MB=100

# Set how long cached data should stay in Redis (in seconds)
CACHE_TTL_SECONDS=3600

# Set the number of message types that can be processed in a single request.
MAX_MESSAGE_TYPES=3

# Redis 
REDIS_HOST=localhost
REDIS_PORT=6380

# App settings
USER_AGENT=drone-chatbot
API_BASE_URL=http://127.0.0.1:8001
VUE_APP_API_BASE_URL=http://127.0.0.1:8001     # API_BASE_URL and VUE_APP_API_BASE_URL should be the same
VUE_APP_CHATBOT_URL=http://localhost:8000
```
 
**Warning**: It's recommended to use Docker for the most stable experience. Running locally may sometimes result in issues with file uploads that can cause page refreshes or processing failures. The Docker version provides better isolation and consistent behavior. 


## Run with Docker 

```bash
# Build UI (Dockerfile is in root folder)
docker build -t ui .
```

```bash
# Build chatbot (Dockerfile is in chatbot folder)
docker build -t chatbot ./chatbot
``` 

```bash 
# Build fastapi (Dockerfile is in fastapi folder)
docker build -t fastapi ./fastapi
```

Once the Docker images are built, run the following command to start the services.

```bash
docker compose up
```

Visit `http://localhost:8080/` to interact with the UI and chatbot. Enter `admin` in the email field and `password` in the password field to log in.

## Run without Docker

### 0. Prerequisites

- **Python**: 3.11 or higher
- **Git**: For submodule initialization
- **Chocolatey** (Windows): Package manager for installing dependencies

### 1. Initialize Git Submodules

```bash
git submodule init

git submodule update
```

### 2. Create and Activate a Virtual Environment

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

### 3. Install Dependencies 

```bash
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
brew install redis

brew install node
```

**Windows:**
```bash
choco install redis-64

choco install nodejs-lts
```

### 4. Start Redis Service

**macOS:**
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

### 5. Run UI, Chatbot and API

Run the commands below in 3 separate terminals. 

```bash
npm install

npm install --save-dev dotenv

npx update-browserslist-db@latest

npm run dev
```

```bash
cd chatbot

chainlit run app.py
```

```bash
cd fastapi

python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Visit `http://localhost:8080/` to interact with the UI and chatbot. Enter `admin` in the email field and `password` in the password field to log in.

