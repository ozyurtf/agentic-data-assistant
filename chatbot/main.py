from pathlib import Path
from models import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query, Body
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Dict, List
from collections import defaultdict
from pymavlink import mavutil
from fastapi.responses import JSONResponse
load_dotenv()

upload_dir = Path("files")
upload_dir.mkdir(exist_ok=True)
executor = ThreadPoolExecutor(max_workers=4)
file_stack = defaultdict(list)
cached_data = {}

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
    print(f"Receiving file upload: {file_id} for user: {user_id}")
    print(f"File name: {file.filename}")
    print(f"File size: {file.size}")
    
    if not file.filename.endswith(('.bin', '.log', '.tlog')):
        raise HTTPException(status_code=400, detail="Only .bin, .log, and .tlog files are supported")
    
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
        print(f"File uploaded successfully. File stack for user {user_id}: {file_stack[user_id]}")
        
        return FileReceiveResponse(**file_data)
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code = 500, detail = f"Failed to upload file: {str(e)}")
    
@app.post("/api/process", description="Process a drone flight log file")
async def process_file(col_map: Dict[str, List[str]] = Body(...), user_id: str = Header(alias="user-id")):
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
    
    file_data = user_stack[-1] 
    
    try:
        data = defaultdict(list)
        rem_msg_types = []
        for msg_type in col_map.keys():
            cache_key = (user_id, file_data["file_path"], msg_type, tuple(sorted(col_map[msg_type])))
            if cache_key in cached_data:
                data[msg_type] = cached_data[cache_key]
                print(f"Data received from cache for {msg_type}.")
            else:
                rem_msg_types.append(msg_type)
        
        if len(rem_msg_types) > 0:        
            file_path = file_data["file_path"]
            mlog = mavutil.mavlink_connection(file_path)
            while True:
                msg = mlog.recv_match(type=rem_msg_types)
                if msg is None:
                    break
                msg_type = msg.get_type()
                if msg_type in col_map:
                    needed_cols = col_map[msg_type]
                    msg_dict = msg.to_dict()
                    row = {col: msg_dict[col] for col in needed_cols if col in msg_dict}
                    if row:  
                        data[msg_type].append(row)
            
            # Cache the newly processed data for each message type
            for msg_type in rem_msg_types:
                cafche_key = (user_id, file_data["file_path"], msg_type, tuple(sorted(col_map[msg_type])))
                cached_data[cache_key] = data[msg_type]
            print(f"Data processed and cached for {len(rem_msg_types)} message types.")
                
        return JSONResponse(status_code=200,
                            content={"success": True,
                                     "message": "File processed successfully",
                                     "data": data,
                                     "metadata": {"file_id": file_data["file_id"],
                                                  "filename": file_data["filename"]}})
        
    except Exception as e:
        file_id = file_data.get("file_id", "unknown")
        return JSONResponse(status_code=500, 
                            content={"success": False,
                                     "error": f"Failed to process file: {str(e)}",
                                     "data": None,
                                     "metadata": {"file_id": file_id}})

    
@app.get("/api/files", description="Get the most recent uploaded file for the user")
async def get_file(user_id: str = Header(alias="user-id")):
    if user_id not in file_stack or not file_stack[user_id]:
        raise HTTPException(status_code=404, detail="No files available for this user")
    user_stack = file_stack[user_id]
    file_data = user_stack[-1] 
    return file_data

@app.delete("/api/files", description="Delete the most recent uploaded file for the user")
async def delete_file(user_id: str = Header(...)):
    if user_id not in file_stack or not file_stack[user_id]:
        raise HTTPException(status_code=404, detail="No files available for this user")
    user_stack = file_stack[user_id]
    file_data = user_stack.pop()
    file_path = file_data['file_path']
    if file_path.exists():
        file_path.unlink()
    return {"message": f"Top file '{file_data['file_id']}' deleted successfully"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}