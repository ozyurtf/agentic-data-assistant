import ast
import pandas as pd
import re
import json
import asyncio
import io 
import os 
import time
from typing import Dict, List, TypedDict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from chainlit.types import ThreadDict
from main import file_stack
import requests
import traceback
from pymavlink import mavutil
from matplotlib import pyplot as plt
from collections import defaultdict
from openai import AsyncOpenAI
import chainlit as cl
from dotenv import load_dotenv
from json import load
import hashlib

load_dotenv()
client = AsyncOpenAI()
cl.instrument_openai()
base_url = os.getenv("API_BASE_URL")
settings = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000}

# Global variable to hold the current message for streaming
current_streaming_msg = None

# Workflow functions with streaming integration
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ArduPilot Analysis State
class ArduPilotAnalysisState(TypedDict):
    query: str
    user_id: str
    file_id: str
    web_content: str
    msg_context: str
    col_map: Dict[str, List[str]]
    data: Dict[str, Any] 
    tool_decisions: List[str]
    tools_completed: List[str]
    tool_results: List[str]
    final_answer: str
    chat_history: Annotated[List[BaseMessage], "The messages in the conversation"]
    
def get_user_id():
    """Get user ID from Chainlit session"""
    user = cl.user_session.get("user")
    return user.identifier if user and hasattr(user, "identifier") else "anonymous"


async def stream_text_word_by_word(text: str, prefix: str = ""):
    """Helper function to stream text word by word"""
    global current_streaming_msg
    if current_streaming_msg and text:
        if prefix:
            await current_streaming_msg.stream_token(f"\n\n**{prefix}:**\n")
        
        # Split by words and spaces to preserve formatting
        tokens = re.split(r'(\s+)', str(text))
        for token in tokens:
            if token:  # Include all tokens (words and spaces)
                await current_streaming_msg.stream_token(token)
                # Only add delay for actual words, not spaces
                if token.strip():  # If it's a word (not just whitespace)
                    await asyncio.sleep(0.05)  # Adjust speed as needed

# Tool definitions with streaming outputs
@tool
def load_web_content(url: str) -> str:
    """
    Load web content from the given URL.
    Use this when user provides a URL and wants to extract content from it.
    """
    try:
        if not url:
            return "No URL provided"
        
        loader = FireCrawlLoader(url=url, mode="scrape")
        docs = loader.load()
        content = " ".join(doc.page_content for doc in docs)
        return f"Successfully loaded content from {url}. Content length: {len(content)} characters."
    except Exception as e:
        return f"Error loading web content: {str(e)}"

@tool
def extract_data(query: str, web_content: str = "") -> str:
    """
    Find the most relevant log message type(s) and the most relevant list of fields to the user query,
    read the data of these message types and fields from the file, and return the results.
    Use this when you need to read the file to answer the user's query.
    """
    try:
        user_id = get_user_id()        
        headers = {"user-id": user_id}
        response = requests.get(f"{base_url}/api/files", headers=headers)
        if response.status_code == 200:
            file_data = response.json()
            file_path = file_data.get("file_path", "")
            file_id = file_data.get("file_id", "")
            if file_path:
                print(f"Using uploaded file: {file_path}")
            else:
                return {
                    "msg_context": "",
                    "col_map": {},
                    "query": query, 
                    "data": {},
                    "error": "No file uploaded. Please upload a log file first."
                }
        else:
            return {
                "msg_context": "",
                "col_map": {},
                "query": query, 
                "data": {},
                "error": "No file uploaded. Please upload a log file first."
            }
        
        # Step 1: Read message types and fields
        step1_start = time.time()
        mlog = mavutil.mavlink_connection(file_path)
        msg_info = defaultdict(set)

        if state["msg_context"] == "":
            while True:
                msg = mlog.recv_match()
                if msg is None:
                    break
                msg_type = msg.get_type()
                msg_info[msg_type].update(msg.to_dict().keys())

            summary = {k: sorted(v) for k, v in msg_info.items()}
            lines = []
            for msg_type, fields in sorted(summary.items()):
                lines.append(f"Log message type: {msg_type}")
                lines.append(f"Fields: {fields}")
                lines.append("")
            
            msg_context = "\n".join(lines)
            state["msg_context"] = msg_context
        else:
            msg_context = state["msg_context"]

        step1_time = time.time() - step1_start
        print(f"Step 1 (Read message types and fields) completed in {step1_time:.2f} seconds")
        
        # Step 2: Extract column mapping
        step2_start = time.time()
        template = """
        User query: {query}

        Based on the field descriptions below:
        {web_content}

        Identify the most relevant log message type(s) and the most relevant list of fields within them
        needed to answer the user query.

        **Important:**  
        - It is very important that the log message type(s) and field(s) you return are part of the log message type(s) in this list: {msg_context}.
        - Respond with ONLY one Python dictionary in this exact format, no extra text or explanation:

        {{'LogMessageType': ['field1', 'field2', ...\n], ...\n}}

        - Replace 'LogMessageType' and field names with your best guesses, based on the provided field descriptions.  
        - Consider relationships between fields. For example, if the query asks for the time when the highest longitude is observed, return both the longitude field and the time field together.  
        - Only include fields necessary to answer the query, avoid irrelevant ones.  
        - Do NOT output placeholders or quotes around keys like 'log message type'.  
        - Do NOT include anything other than the Python dictionary.
        - If you are not sure about the log message types and/or field names, ask the user for clarification.
        """

        prompt = PromptTemplate(input_variables=["query", "web_content", "msg_context"], template=template)
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | model
        result = chain.invoke({"query": query, "web_content": web_content, "msg_context": msg_context})
        col_map = ast.literal_eval(result.content.strip())
        step2_time = time.time() - step2_start
        print(f"Step 2 (Extract column mapping) completed in {step2_time:.2f} seconds")
        
        # Step 3: Read data using the API endpoint
        step3_start = time.time()
        user_id = get_user_id()
        headers = {"user-id": user_id}
        
        response = requests.post(f"{base_url}/api/process", json=col_map, headers=headers)
                
        if response.status_code != 200:
            return {
                "msg_context": msg_context,
                "col_map": col_map,
                "query": query, 
                "data": {},
                "error": f"API request failed with status {response.status_code}: {response.text}"
            }
        
        response_data = response.json()
        if not response_data.get("success"):
            return {
                "msg_context": msg_context,
                "col_map": col_map,
                "query": query, 
                "data": {},
                "error": f"API processing failed: {response_data.get('error', 'Unknown error')}"
            }
        
        data = response_data["data"]
        result_data = {}
        for msg_type, rows in data.items():
            if rows:
                df = pd.DataFrame(rows)
                df.dropna(axis=1, how='all', inplace=True) 
                result_data[msg_type] = df
        
        step3_time = time.time() - step3_start
        print(f"Step 3 (Read data using API endpoint) completed in {step3_time:.2f} seconds")
        
        # Print total execution time
        total_time = step1_time + step2_time + step3_time
        print(f"Total execution time: {total_time:.2f} seconds")
        
        return {
            "msg_context": msg_context,
            "col_map": col_map,
            "query": query, 
            "data": result_data,
            "file_id": file_id
        }   
            
    except Exception as e:
        # Raise the error
        return {
            "msg_context": "",
            "col_map": {},
            "query": query, 
            "data": {},
            "error": f"Error in extract_data: {str(e)}"
        }    

# Analysis tools (these will be enhanced with streaming)
@tool
def maximum(data_description: str): 
    """Calculate the maximum value of numeric fields in the data."""
    return "Maximum calculation tool called - processing will be handled by workflow"

@tool  
def minimum(data_description: str):
    """Calculate the minimum value of numeric fields in the data."""
    return "Minimum calculation tool called - processing will be handled by workflow"

@tool
def average(data_description: str):
    """Calculate the average value of numeric fields in the data."""
    return "Average calculation tool called - processing will be handled by workflow"

@tool
def total_sum(data_description: str):
    """Calculate the sum of numeric fields in the data."""
    return "Sum calculation tool called - processing will be handled by workflow"        

@tool
def when_maximum(data_description: str):
    """Find when/where the maximum value occurred, including timestamp and context."""
    return "When maximum tool called - processing will be handled by workflow"

@tool
def when_minimum(data_description: str):
    """Find when/where the minimum value occurred, including timestamp and context."""
    return "When minimum tool called - processing will be handled by workflow"

@tool
def visualize(data_description: str):
    """Visualize the data in a plot."""
    return "Visualization tool called - processing will be handled by workflow"


async def decide_tools_to_use(state: ArduPilotAnalysisState) -> ArduPilotAnalysisState:
    """LLM decides which tools to use based on the user query from ALL available tools."""
    await stream_text_word_by_word("Analyzing query to determine required tools...\n", "Tool Selection")
    
    # Check if URL is present in the query
    url_pattern = re.compile(r'(?:(?:https?|ftp)://|www\.)\S+', re.IGNORECASE)
    has_url = bool(url_pattern.search(state["query"]))
        
    template = """
    User Query: {query}
    
    Available Tools (choose any combination needed):
    
    Tools:
    1. load_web_content: Extract the content of the given URL.
    2. extract_data: Extract the data most relevant to the user query so that it can be used to answer the user's query.
    3. maximum: Use this if the user's query requires calculating a maximum value of the given data.
    4. minimum: Use this if the user's query requires calculating a minimum value of the given data.
    5. average: Use this if the user's query requires calculating an average value of the given data.
    6. total_sum: Use this if the user's query requires calculating a sum of the given data.
    7. when_maximum: Use this if the user's query requires finding when the maximum value occurred in the given data.
    8. when_minimum: Use this if the user's query requires finding when the minimum or first instance of a value occurred in the given data.
    9. visualize: Use this if the user's query requires visualizing the given data.
    
    Based on the user query and these rules, determine which tools should be used. 
    
    Sometimes you might need only one tool. Sometimes you might need multiple tools. And sometimes you might not need any tool at all.
    When choosing the tools, think about which one would be useful to answer the user's query.

    **Important:** 
    - Consider the conversation context here {chat_history}. If the user is asking follow-up questions or referring to previous analysis, 
    you may need different tools or no tools at all.
    - Also, if the user is asking for a visualization, first extract relevant data using the extract_data tool. 
    After this, the data that will be used for visualization will be added to the state. 
    And then you can use this data and visualize it by using the visualize tool. 
    - Don't visualize the data unless you are explicitly asked.
    - If the user is asking for anomaly detection, use the anomaly_detection tool.  

    **Important:**
    - If the user is asking for anomaly detection, look at the data you have available after using the extract_data tool
    and check the issues/anomalies in the data. For example, if the user 

    Respond with ONLY the tool name(s) separated by commas, or "none" if no tools are needed.
    """
    
    prompt = PromptTemplate(input_variables=["query", "chat_history"], template=template)
    
    model_local = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    chain = prompt | model_local
    result = chain.invoke({
        "query": state["query"],
        "chat_history": state["chat_history"]
    })
    
    # Parse the response to get list of tools
    tool_response = result.content.strip().lower()
    
    if tool_response == "none":
        state["tool_decisions"] = []
        state["final_answer"] = "Hello! I can help you analyze your files and extract content from documentation. Please provide a URL to extract content from, or ask a question about log file data."
        await stream_text_word_by_word("No tools required for this query.")
    else:
        tools = [tool.strip() for tool in tool_response.split(',')]
        state["tool_decisions"] = tools
    
    state["tools_completed"] = []
    state["tool_results"] = []
    
    return state

async def execute_unified_tools(state: ArduPilotAnalysisState) -> ArduPilotAnalysisState:
    """Execute ANY tool selected by the LLM from the unified tool set with streaming output."""
    if not state["tool_decisions"]:
        return state
    
    # Find the next tool to execute
    pending_tools = [tool for tool in state["tool_decisions"] if tool not in state["tools_completed"]]
    
    if not pending_tools:
        await stream_text_word_by_word("All selected tools have been executed\n")
        return state
    
    current_tool = pending_tools[0]
    
    try:
        if current_tool == "load_web_content": 
            await stream_text_word_by_word("Extracting URL from query...\n")
            
            # Extract URL from query
            url_pattern = re.compile(r'(?:(?:https?|ftp)://|www\.)\S+', re.IGNORECASE)
            match = url_pattern.search(state["query"])
            url = match.group(0) if match else ""
            
            if url:
                await stream_text_word_by_word(f"Loading content from: {url}...\n")
            
            # Call the actual tool function
            tool_result = load_web_content.invoke({"url": url})
            
            # Update state with the results
            if url and "Successfully loaded" in tool_result:
                loader = FireCrawlLoader(url=url, mode="scrape")
                docs = loader.load()
                state["web_content"] = " ".join(doc.page_content for doc in docs)
                await stream_text_word_by_word("Web content successfully loaded and cached for analysis\n")
            
            state["tool_results"].append(f"Web content extraction result:\n{tool_result}")
            
        elif current_tool == "extract_data":
            await stream_text_word_by_word("Reading log messages from your file...\n")
            await stream_text_word_by_word("Extracting message types and fields...\n")
            
            # Debug: Print current state
            print(f"Current state query: {state.get('query', 'NOT SET')}")
            
            # Call the actual tool function
            tool_result = extract_data.invoke({
                "query": state["query"],
                "web_content": state.get("web_content", ""),
            })
            
            # Update the state for downstream analysis tools
            if tool_result:
                if "error" in tool_result:
                    await stream_text_word_by_word(f"Error during data extraction: {tool_result['error']}\n")
                    state["tool_results"].append(f"Data extraction error: {tool_result['error']}")
                else:
                    state["msg_context"] = tool_result["msg_context"]
                    state["col_map"] = tool_result["col_map"]
                    state["query"] = tool_result["query"]
                    state["data"] = tool_result["data"]
                    state["file_id"] = tool_result["file_id"]
                    
                    await stream_text_word_by_word("Relevant message types and fields extracted\n")
                    await stream_text_word_by_word(f"Data frames extracted for {len(tool_result['data'])} message types\n")
            
            state["tool_results"].append(f"Data extraction and analysis result:\n{tool_result}")
        
        elif current_tool == "visualize":
            await stream_text_word_by_word("Visualizing data...\n")
            await stream_text_word_by_word("Generating visualization code...\n")
            
            sampled_data = {}
            for key in state["data"]:
                sampled_data[key] = state["data"][key].sample(100, replace = True)

            plt.rcParams.update({'figure.dpi': 150,})                

            template = """
            User query: {query}

            I have a dataframe(s) in a dictionary called `state["data"]`. 
            Here is how the 100 rows sampled from it looks like: {sampled_data}. 

            Write a Plotly function in Python to visualize the data so that I can execute it and get the plot.

            Only give the code, nothing else. Don't include ```python or ``` or anything else. 
            Don't explain the data.

            Important: Make the plot look nice, readable, and high quality. 
            Don't use any other libraries than matplotlib, numpy, pandas, datetime, and other standard libraries
            that are already installed in the system.
            And make sure that the code you generate can be run with 1 click without needing any modification/change.
            
            Important: `state[data]` already exists, don't create a new one!
            If there are multiple dataframes, give separate plots for each one.

            Also, do not include any Python code in your response. 
            """

            prompt = PromptTemplate(input_variables=["query", "sampled_data"], template=template)
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            chain = prompt | model
            result = chain.invoke({"query": state["query"], "sampled_data": sampled_data})
            code = result.content.strip()
            code = code.replace("plt.show()", "")  
            
            # Execute the visualization code
            exec(code)
            
            # Get the current figure and create cl.Pyplot element
            fig = plt.gcf()
            fig.set_dpi(300)
            
            await stream_text_word_by_word("Generated plot.\n")
            
            # Use cl.Pyplot for better visualization display
            elements = [
                cl.Pyplot(name="plot", figure=fig, display="inline"),
            ]
            
            await cl.Message(
                content="Here is your visualization:",
                elements=elements,
            ).send()
            
            # Clear the figure to free memory
            plt.close(fig)                
            
        # Analysis tools (require data to be available)
        elif current_tool in ["maximum", "minimum", "average", "total_sum", "when_maximum", "when_minimum"]:
            if not state.get("data"):
                error_msg = f"{current_tool} requires data to be extracted first using extract_data"
                await stream_text_word_by_word(error_msg)
                state["tool_results"].append(f"Analysis tool error:\n{error_msg}")
            else:
                data = state["data"]
                result_parts = []
                
                await stream_text_word_by_word(f"Computing {current_tool} values...\n")
                
                if current_tool == "maximum":
                    for msg_type, df in data.items():
                        numeric_cols = df.select_dtypes(include=['number'])
                        if not numeric_cols.empty:
                            max_values = numeric_cols.max()
                            result_parts.append(f"Maximum values in {msg_type}:")
                            for col, val in max_values.items():
                                result_parts.append(f"  {col}: {val}")
                                
                elif current_tool == "minimum":
                    for msg_type, df in data.items():
                        numeric_cols = df.select_dtypes(include=['number'])
                        if not numeric_cols.empty:
                            min_values = numeric_cols.min()
                            result_parts.append(f"Minimum values in {msg_type}:")
                            for col, val in min_values.items():
                                result_parts.append(f"  {col}: {val}")
                                
                elif current_tool == "average":
                    for msg_type, df in data.items():
                        numeric_cols = df.select_dtypes(include=['number'])
                        if not numeric_cols.empty:
                            avg_values = numeric_cols.mean()
                            result_parts.append(f"Average values in {msg_type}:")
                            for col, val in avg_values.items():
                                result_parts.append(f"  {col}: {val:.2f}")

                elif current_tool == "total_sum": 
                    for msg_type, df in data.items(): 
                        numeric_cols = df.select_dtypes(include=['number'])
                        if not numeric_cols.empty: 
                            sum_values = numeric_cols.sum()
                            result_parts.append(f"Sum values in {msg_type}:")
                            for col, val in sum_values.items(): 
                                result_parts.append(f"  {col}: {val}")
                
                elif current_tool == "when_maximum":
                    await stream_text_word_by_word("Finding timestamps and context for maximum values...\n")
                    for msg_type, df in data.items():
                        numeric_cols = df.select_dtypes(include=['number'])
                        if not numeric_cols.empty:
                            result_parts.append(f"When maximum values occurred in {msg_type}:")
                            for col in numeric_cols.columns:
                                max_idx = df[col].idxmax()
                                max_row = df.loc[max_idx]
                                result_parts.append(f"  Maximum {col} ({max_row[col]}) occurred at:")
                                # Include all available context from that row
                                for field, value in max_row.items():
                                    if field != col:  # Don't repeat the max value itself
                                        result_parts.append(f"    {field}: {value}")
                                result_parts.append("")  # Add space between fields
                
                elif current_tool == "when_minimum":
                    await stream_text_word_by_word("Finding timestamps and context for minimum values...\n")
                    for msg_type, df in data.items():
                        numeric_cols = df.select_dtypes(include=['number'])
                        if not numeric_cols.empty:
                            result_parts.append(f"When minimum values occurred in {msg_type}:")
                            for col in numeric_cols.columns:
                                min_idx = df[col].idxmin()
                                min_row = df.loc[min_idx]
                                result_parts.append(f"  Minimum {col} ({min_row[col]}) occurred at:")
                                # Include all available context from that row
                                for field, value in min_row.items():
                                    if field != col:  # Don't repeat the min value itself
                                        result_parts.append(f"    {field}: {value}")
                                result_parts.append("")  # Add space between fields
                
                tool_result = "\n".join(result_parts)
                
                state["tool_results"].append(f"Results from {current_tool} tool:\n{tool_result}")
            
        elif current_tool == "none":
            tool_results = "No tools required."
            await stream_text_word_by_word(tool_results)
            state["tool_results"].append(tool_results)
            
        # Mark tool as completed
        state["tools_completed"].append(current_tool)
        
    except Exception as e:
        error_msg = f"Error executing {current_tool}: {str(e)}"
        await stream_text_word_by_word(error_msg)
        state["tool_results"].append(f"Tool execution error:\n{error_msg}")
        state["tools_completed"].append(current_tool)  # Mark as completed to avoid infinite loop
    
    return state

def check_unified_tool_completion(state: ArduPilotAnalysisState) -> str:
    """Check if all selected tools have been executed."""
    if not state["tool_decisions"]:
        return "generate_final_answer"  # No tools were selected
    
    pending_tools = [tool for tool in state["tool_decisions"] if tool not in state["tools_completed"]]
    
    if pending_tools:
        return "execute_tools"  # More tools to execute
    else:
        return "generate_final_answer"  # All tools completed, generate final answer

async def generate_final_answer(state: ArduPilotAnalysisState) -> ArduPilotAnalysisState:
    """Generate the final answer based on all collected information."""
    
    await stream_text_word_by_word("Generating comprehensive answer based on analysis results...\n", "Final Analysis")
    
    # Combine all tool results
    all_tool_results = "\n\n".join(state["tool_results"])
    
    template = """
    User Query: {query}
    
    Analysis Results from Multiple Tools:
    {tool_results}
    
    Web Content Context: {web_content}
    
    Based on the analysis results from all the tools used and the context from the ArduPilot documentation, 
    provide a comprehensive and clear answer to the user's query. Make sure to address all parts of the query.

    Sometimes, the same fields can be seen in different message types. 
    For example, the longitude and latitude fields can be seen in the GPS and AHR2 message types.
    In this case, you should use the fields from the message type that is most relevant to the user's query 
    and explain why you chose that message type when giving the answer.

    If you don't know the answer or you are not sure, ask clarification from the user to give more specific information.

    **Important:** Consider the conversation context here {chat_history}.
    If the user is asking follow-up questions or referring to previous analysis, you may need different tools or no tools at all.
    Also, when giving the answer, make sure that it is organized, clean, readable and in a nice format.
    """

    prompt = PromptTemplate(
        input_variables=["query", "tool_results", "web_content", "chat_history"], 
        template=template
    )
    
    chain = prompt | model
    result = chain.invoke({
        "query": state["query"],
        "tool_results": all_tool_results,
        "web_content": state["web_content"],
        "chat_history": state["chat_history"]
    })
    
    state["final_answer"] = result.content
    return state

# Create the unified LLM-driven workflow with async support
workflow = StateGraph(ArduPilotAnalysisState)

# Add nodes for the unified system
workflow.add_node("decide_tools", decide_tools_to_use)
workflow.add_node("execute_tools", execute_unified_tools)
workflow.add_node("generate_final_answer", generate_final_answer)

# Set the workflow flow
workflow.add_edge(START, "decide_tools")

# After deciding tools, execute them or generate answer if no tools needed
workflow.add_conditional_edges(
    "decide_tools",
    lambda state: "execute_tools" if state["tool_decisions"] else "generate_final_answer",
    {
        "execute_tools": "execute_tools",
        "generate_final_answer": "generate_final_answer"
    }
)

# After executing each tool, check if we need to execute more or generate final answer
workflow.add_conditional_edges(
    "execute_tools",
    check_unified_tool_completion,
    {
        "execute_tools": "execute_tools",  # Loop back to execute next tool
        "generate_final_answer": "generate_final_answer"  # All tools completed
    }
)

# End point
workflow.add_edge("generate_final_answer", END)
    
app = workflow.compile()

state = ArduPilotAnalysisState(
    query="",
    user_id="",
    file_id="",
    web_content="",
    msg_context="",
    col_map={},
    data={},
    tool_decisions=[],
    tools_completed=[],
    tool_results=[],
    final_answer="",
    chat_history=None
)

# Chainlit authentication and chat functions
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    with open("user.json", "r") as f:
        creds = json.load(f)
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash == creds["password_hash"]:
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )
    return None

@cl.on_chat_start
async def start_chat():
    print("ArduPilot Analysis Chatbot Started!")
    system_msg = "You are an agentic drone flight data analyst with access to powerful tools for analyzing ArduPilot log files and extracting content from documentation. You can use multiple tools to provide comprehensive answers."
    cl.user_session.set("chat_history", [{"role": "system", "content": system_msg}])

@cl.on_message
async def main(message: cl.Message):
    global current_streaming_msg
    
    try:
        chat_history = cl.user_session.get("chat_history")
        query = message.content

        state["query"] = query
        state["user_id"] = get_user_id()
        state["chat_history"] = chat_history
        
        # Create a streaming message for the response
        current_streaming_msg = cl.Message(content="")
        await current_streaming_msg.send()
        
        # Use the compiled workflow to handle the entire process
        try:
            # Run the workflow with streaming
            final_state = await app.ainvoke(state)
            
            # Get the final answer and stream it
            final_answer = final_state.get('final_answer', 'No answer generated')
            
            # Stream the final answer word by word
            await stream_text_word_by_word(final_answer, "Final Answer")
            
            # Show completion status
            tools_completed = final_state.get('tools_completed', [])
            if tools_completed:
                completion_msg = f"Analysis completed successfully using tools: {', '.join(tools_completed)}"
                await stream_text_word_by_word(completion_msg, "Completion Status")
            
        except Exception as workflow_error:
            await stream_text_word_by_word(f"Workflow error: {str(workflow_error)}")
        
        # Update chat history
        tools_completed = final_state.get('tools_completed', []) if 'final_state' in locals() else []
        chat_history.append({"tool_calls": tools_completed})
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": current_streaming_msg.content})
        cl.user_session.set("chat_history", chat_history)
        
        # Update the message
        await current_streaming_msg.update()
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        if current_streaming_msg:
            await stream_text_word_by_word(f"Error occurred: {str(e)}")
        else:
            await cl.Message(content=f"**Error occurred:** {str(e)}").send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")
    
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
    
@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")