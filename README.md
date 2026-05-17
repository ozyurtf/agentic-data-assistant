**Note**: The code used to build the UI can be seen in the `Reference` section at the bottom. And I am implementing/integrating the features below on top of it.

# Features

- Multi-agent system <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Multi-step reasoning <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- Backend development <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Session management <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Rate limiting <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Caching extracted data with Redis <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Web scraping <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Photorealistic 3D map <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Data validation with Pydantic <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Sign-up and log-in mechanisms with e-mail and password <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- Password hashing with Argon2 <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- JWT authentication <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- AWS PostgreSQL integration to store user information <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- AWS S3 bucket integration to store the uploaded files <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- Manual high-quality and diverse data collection to evaluate the system <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- Online and offline evaluation system with LangSmith <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- Tracking the evaluation metrics in a dashboard <span style="color:rgb(241, 140, 16); font-weight: bold;">**(In Progress)**</span>
- Multi-service Docker orchestration <span style="color: #28a745; font-weight: bold;">**(Done)**</span>
- Deployment in AWS <span style="color: #28a745; font-weight: bold;">**(Done)**</span>

# Demo

<a href="https://www.youtube.com/watch?v=yvr2dXFJIT0"><img src="images/demo.png" alt="Demo video on YouTube" width="100%"></a>

# System

<p align="center"><img src="images/authentication.png" alt="" width="100%"></p>
<p align="center"><img src="images/file-upload.png" alt="" width="100%"></p>
<p align="center"><img src="images/data-extraction-tool.png" alt="" width="100%"></p>
<p align="center"><img src="images/agents.png" alt="" width="100%"></p>

# Evaluation 

## Manual Data Collection
- When preparing the dataset to evaluate the systems, I prepared different groups of datasets to be able to evaluate the system from different/diverse perspectives.
  - Queries that require information available in the uploaded file
    1. Queries that require multi-step reasoning (category: `multi_step_reasoning`)
    2. Queries that require extracting and returning specific information from the uploaded file (category: `extractive`)
    3. Queries that require relevant information from external web pages (listed below) to be used when generating the answer (category: `external_knowledge_usage`)
    4. Prompts that request multiple tasks to be completed (category: `multi_task`)

  - Queries that require information not available in the uploaded file
    1. Queries that measure the system's awareness of external knowledge related to the uploaded file (cateogry: `external_ardupilot_logs`, `external_mavlink_common` or `external_mavlink_dialect`)
    2. Daily-life queries that are not related to this topic at all (category: `out_of_scope`)
    3. Queries that are technical but cannot be answered using the information available in the uploaded file (category: `not_found`)

The list of web pages that have the technical information that might be beneficial for the agents:
- ArduCopter onboard log messages: `https://ardupilot.org/copter/docs/logmessages.html`
- Standard MAVLink common messages: `https://mavlink.io/en/messages/common.html`
- ArduPilot MAVLink dialect messages: `https://mavlink.io/en/messages/ardupilotmega.html`

## Offline Evaluation
- **Context score** (whether all the required data and information is available in the context)
- **Correctnes score** with LLM as a judge (whether the answer semantically matches with the ground truth)
- **Exact match score** (for questions that require extracting specific data from the uploaded file)
- **Node selection** (whether the right nodes are chosen for execution)
- **Tool selection** (whether the right tools are chosen for execution)
- **C-DNF** ("Correct data not found") score (sometimes the user asks a question, but the required data may not exist in the uploaded file. It is important for the system to detect this correclty, and answer that the required data was not found in the uploaded file instead of making assumptions).
- **Average task completion rate** (out of all the user requests in a prompt, how many are completed successfully?)
- **Conciseness**

## Online Evaluation
- **P50/P90/P99 latency**
- **Total token usage**
- **Total cost**
- **Node failure rate**
- **Tool failure rate**
- **Cache hit rate**
- **Ratio of failed answers**
- **User-reported feedback**

## Evaluation Platform
To track these metrics, there were many options such as: 

- LangSmith
- OpenAI evaluation platform 
- Anthropic evaluation platform
- Manual evaluation with custom Python code and Weights & Biases

Considering that I had already used LangChain and LangGraph during the process, and that LangSmith already provides many features that make it easy to evaluate the system and build dashboards, I decided to use **LangSmith**.

## Dashboard

`To be announced`

# Running  

Create a `files` folder inside `api`

```bash
mkdir -p api/files
```

## Configure Environment Variables

Create an `.env` file in the root folder with the following values. The environment variables will be automatically loaded when you run the development server:

```env 
# Cesium 
VUE_APP_CESIUM_TOKEN=<your_cesium_ion_token>   # Get from https://ion.cesium.com/signin
VUE_APP_CESIUM_RESOURCE_ID=3

# Google Maps Platform
VUE_APP_GOOGLE_MAPS_KEY=<your_google_maps_key>

# MapTiler 
VUE_APP_MAPTILER_KEY=<your_maptiler_key>       # Get from https://docs.maptiler.com/cloud/api/authentication-key/

# OpenAI 
LLM_PROVIDER=openai
OPENAI_API_KEY=<your_openai_api_key>           # Get from https://platform.openai.com/api-keys

# Firecrawl
FIRECRAWL_API_KEY=<your_firecrawl_api_key>     # Get from https://www.firecrawl.dev

# Chatbot
CHAINLIT_AUTH_SECRET=<your_chainlit_secret>    # Get from https://docs.chainlit.io/authentication/overview

# Set the maximum file size allowed for uploading
MAX_FILE_SIZE_MB=100

# Set how long cached data should stay in Redis (in seconds)
CACHE_TTL_SECONDS=3600

# Set the number of data types that can be extracted from the file in a single request.
MAX_MESSAGE_TYPES=3

# App settings
USER_AGENT=drone-chatbot

# Ports and hosts 
API_HOST=localhost
API_PORT=8001

CHATBOT_HOST=localhost
CHATBOT_PORT=8000

UI_HOST=0.0.0.0
UI_PORT=8080

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<enter_a_password_for_redis>

VUE_APP_API_BASE_URL=http://localhost:8001
VUE_APP_CHATBOT_URL=http://localhost:8000

# Note: 
# If you change the API_HOST, API_PORT, CHATBOT_HOST, or CHATBOT_PORT,
# you should reflect these changes in VUE_APP_API_BASE_URL and VUE_APP_CHATBOT_URL as well:
# VUE_APP_API_BASE_URL=http://API_HOST:API_PORT
# VUE_APP_CHATBOT_URL=http://CHATBOT_HOST:CHATBOT_PORT
```

## Run with Docker Locally

To start building containers and running services, make sure Docker Desktop application is running and run the containers:

```bash
docker-compose up -d
```

Visit `http://localhost:8080/` to interact with the UI and chatbot. The page may take a few moments to load.

Once the page is loaded, enter `admin` in the email field and `password` in the password field to log in to the application. 

**Warning**: Please log in first before uploading a file. 

To stop all services, you can run:

```bash
docker-compose down
```

## Run with Docker in AWS

**1) Create EC2 Instance**
- AMI: Ubuntu 24.04 LTS
- Instance type: m7i-flex.large
- Storage: 20–30 GB
- Number of instances: 1
- Security group rules: Allow ports `22` (SSH from your IP), `8080` (UI), `8000` (Chatbot), `8001` (API) (0.0.0.0/0 for testing)

**2) Connect and Prepare the Machine**
  
```bash  
ssh -i your-key.pem ubuntu@your-public-ip
  
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
  
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
  
exit
```

```bash
ssh -i your-key.pem ubuntu@your-public-ip
```

**3) Deploy Code**

Clone your repository and configure environment:

```bash  
git clone https://github.com/ozyurtf/agentic-data-assistant.git
cd agentic-data-assistant

# Create and edit .env using the variables listed in the "Configure Environment Variables" section above
touch .env
nano .env   # set OPENAI_API_KEY, VUE_APP_CESIUM_TOKEN, FIRECRAWL_API_KEY, etc.
```

**4) Launch Services**
  
```bash
docker-compose up -d
```

**5) Access**

- UI at `http://your-public-ip:8080`
- Chatbot at `http://your-public-ip:8000`
- API docs at `http://your-public-ip:8001/docs`
- Default login: `admin` / `password`

# Notes

## Configuration Flexibility

The system is fully configurable via the `.env` file:

- **Ports**: Change any service port by modifying `UI_PORT`, `API_PORT`, `CHATBOT_PORT`, or `REDIS_PORT`
- **Hosts**: Configure service hosts using `UI_HOST`, `API_HOST`, `CHATBOT_HOST`, or `REDIS_HOST`

The application will automatically use your configured values throughout the entire stack.

## CORS Configuration
- The API uses CORS and currently allows requests from:
  `- http://localhost:8080` (Vue frontend)
  `- http://localhost:8000` (Chatbot)

If you run the frontend/chatbot on a different host or port (or deploy to a domain),
update `allow_origins` in `api/main.py` so it includes the new origin(s). 

# References

1) UAV Log Viewer: `https://github.com/ArduPilot/UAVLogViewer`