from pathlib import Path
from models import *
from process import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query, Body
from dotenv import load_dotenv
import asyncio
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import json
from langchain_core.vectorstores import InMemoryVectorStore
from collections import defaultdict, deque
from fastapi.responses import JSONResponse
from typing import List

load_dotenv()

upload_dir = Path("files")
upload_dir.mkdir(exist_ok=True)
executor = ThreadPoolExecutor(max_workers=4)
embedding_model = OpenAIEmbeddings()
vectorstore = InMemoryVectorStore(embedding_model)
file_stack = defaultdict(deque)
url_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    executor.shutdown(wait=True)

app = FastAPI(title="Drone Log API", 
              description="API for processing drone flight logs", 
              version="1.0.0",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware,
                   allow_origins = ["http://localhost:3000", "http://localhost:8080"], 
                   allow_credentials = True,
                   allow_methods = ["GET", "POST", "DELETE"],
                   allow_headers = ["*"])

@app.post("/api/files/{file_id}", response_model = FileReceiveResponse, status_code = 201, description = "Upload a drone flight log file")
async def receive_file(file_id: str, file: UploadFile = File(...), user_id: str = Header(alias="user-id")):
    """
    Receives a drone flight log file via POST request and temporarily saves it to the local filesystem.

    - Supports only .bin and .log file extensions.
    - Rejects files larger than 100MB.
    - Saves the file with a prefixed file_id to avoid naming collisions.
    - Associates the uploaded file with the requesting user (via "user-id" header).
    - Stores file metadata (ID, path, name) in an in-memory per-user file stack for later access.
    - Cleans up partially written files in case of errors.
    """
    if not file.filename.endswith(('.bin', '.log')):
        raise HTTPException(status_code=400, detail="Only .bin and .log files are supported")
    
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail= "File too large. Max size is 100MB")
    
    file_name = file.filename
    file_path = upload_dir / f"{file_id}_{file_name}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)    
            
        file_data = {"file_id": file_id,
                     "file_path": str(file_path),
                     "filename": file.filename}  
        
        file_stack[user_id].append(file_data)                 
        return FileReceiveResponse(**file_data)
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code = 500, detail = f"Failed to upload file: {str(e)}")
    
@app.post("/api/process", description="Process a drone flight log file")
async def process_file(msg_types: List[str] = Query(...), user_id: str = Header(alias="user-id")):
    """
    Processes the most recently uploaded drone flight log file for a specific user.

    - Requires "msg_types" (list of message types to extract) and a `user-id` header.
    - If no file is associated with the user, returns a 404 error.
    - If a file is found, it is removed from the user's file stack, read, and processed.
    - Returns selected contents based on the provided message types.
    - On error, returns a 500 response with error details.
    """
    user_stack = file_stack.get(user_id, [])
    
    if not user_stack:
        return JSONResponse(status_code=404, 
                            content={"success": False,
                                     "error": "No file found for this user",
                                     "data": None})
    
    file_data = user_stack.pop()
    
    try:
        file_path = file_data["file_path"]
        page_contents = read_data(file_path, msg_types)
        return JSONResponse(status_code=200,
                            content={"success": True,
                                     "message": "File processed successfully",
                                     "data": page_contents,
                                     "metadata": {"file_id": file_data["file_id"],
                                                  "filename": file_data["filename"],
                                                  "msg_types": msg_types}})
        
    except Exception as e:
        file_id = file_data.get("file_id", "unknown")
        return JSONResponse(status_code=500, 
                            content={"success": False,
                                     "error": f"Failed to process file: {str(e)}",
                                     "data": None,
                                     "metadata": {"file_id": file_id}})

    
@app.delete("/api/files", description="Delete the most recent uploaded file for the user")
async def delete_file(user_id: str = Header(...)):
    if user_id not in file_stack or not file_stack[user_id]:
        raise HTTPException(status_code=404, detail="No files available for this user")
    user_stack = file_stack[user_id]
    file_data = user_stack.popleft()
    file_path = file_data['file_path']
    if file_path.exists():
        file_path.unlink()
    return {"message": f"Top file '{file_data['file_id']}' deleted successfully"}


@app.post("/api/vectorstore/update")
async def update_vectorstore(request: UpdateVectorstoreRequest):
    try:        
        docs = [Document(page_content=page_content) for page_content in request.page_contents]
        vectorstore.add_documents(docs)
        return {"status": "updated", "message": f"Vectorstore updated with {len(docs)} documents"}
    except Exception as e:
        return {"status": "error", "message": f"Exception: {str(e)}"}
    
@app.get("/api/vectorstore/query")
async def query_vectorstore(query: str):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    try:
        relevant_docs = retriever.invoke(query)
        if not relevant_docs:
            return {"context": "", "message": "No relevant documents found."}
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"Relevant documents are retrieved from vectorstore")
        return {"context": context, "message": "Relevant documents are retrieved"}

    except Exception as e:
        return {"context": "", "message": f"Error during retrieval: {str(e)}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}