from openai import AsyncOpenAI
import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv
from process import *
import os 
from chainlit.types import ThreadDict
from main import url_cache, find_url, file_stack, DynamicTableParser
import requests
import traceback

load_dotenv()
client = AsyncOpenAI()
cl.instrument_openai()
base_url = os.getenv("API_BASE_URL")
user_id = "fozyurt"

settings = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 500}
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    print(f"Auth attempt: {username}")
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    else:
        return None

@cl.on_chat_start
async def start_chat():
    print("Chat started!")
    system_msg = "You are a drone flight data analyst. Answer questions accordingly."
    cl.user_session.set("chat_history", [{"role": "system", "content": system_msg}])

@cl.on_message
async def main(message: cl.Message):
    try:
        chat_history = cl.user_session.get("chat_history")
        query = f"User's Query: {message.content}"
        url = find_url(query)
        
        msg_types = find_relevant_types(query)
        
        response = requests.post(f"{base_url}/api/process",
                                 params= {"msg_types": msg_types},
                                 headers={"user-id": user_id})
        
        page_contents = response.json()["data"]
        if page_contents:
            requests.post(f"{base_url}/api/vectorstore/update", 
                          json={"page_contents": page_contents})
            # requests.delete(f"{base_url}/api/files", headers = {"user_id": user_id})
        
        if url and url not in url_cache:
            web_page_chunks = []
            url_cache[url] = True
            parser = DynamicTableParser(url)
            extracted_data = parser.extract_all_data()
            for key in extracted_data:
                chunk = extracted_data[key]
                web_page_chunks.append(json.dumps(chunk))
            print(web_page_chunks)
            requests.post(f"{base_url}/api/vectorstore/update",
                          params={"docs": web_page_chunks})
            
        query_response = requests.get(f"{base_url}/api/vectorstore/query", params={"query": query})
        retrieved_context = query_response.json()["context"]
        
        if retrieved_context:
            query += f"\nRetrieved Context: {retrieved_context}"
            
        prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"), 
            ("user", "{query}")
        ])  
        
        prompts = prompt_template.format_messages(chat_history=chat_history, query=query)
        chat_history.append({"role": "user", "content": query})
        
        openai_messages = [{"role": convert_role(prompt.type), "content": prompt.content} for prompt in prompts]    
        
        print("Calling OpenAI...")
        stream = await client.chat.completions.create(messages=openai_messages,
                                                      stream=True,
                                                      **settings)
        
        msg = cl.Message(content="")
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)
        
        chat_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", chat_history)
        await msg.update()
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        await cl.Message(content=f"Error: {str(e)}").send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")
    
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
    
@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")