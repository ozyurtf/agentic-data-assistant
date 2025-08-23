from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
import requests
from pymavlink import mavutil
import ast
import pandas as pd
from collections import defaultdict
from langchain_core.prompts import PromptTemplate
import os
import matplotlib
import matplotlib.pyplot as plt
from chainlit.types import ThreadDict
import threading
import asyncio
import hashlib
import json
matplotlib.use('Agg') 

load_dotenv()
base_url = os.getenv("API_BASE_URL")

def get_user_id():
    """Get user ID from Chainlit session"""
    user = cl.user_session.get("user")
    return user.identifier if user and hasattr(user, "identifier") else "anonymous"

def filter_data() -> dict:
    """
    Filter the data based on the col_map.
    """
    filtered_data = {}
    data = cl.user_session.get("data")
    col_map = cl.user_session.get("col_map")
    for msg_type, rows in data.items():
        if msg_type in col_map:
            filtered_data[msg_type] = rows[col_map[msg_type]]            
    return filtered_data

@tool
async def load_web_content(url: str) -> str:
    """
    Load web content from the given URL.
    Use this when user provides a URL and wants to extract content from it.
    """
    try:
        async with cl.Step(name="web scraping tool to extract web content", type="tool") as step:
            if not url:
                step.output = "No URL provided"
                return "No URL provided"
            
        # Attention: Web content is forgotton after the query is answered. 
        step.output = f"Loading web content from the {url}...\n"
        loader = FireCrawlLoader(url=url, mode="scrape")
        docs = loader.load()
        content = " ".join(doc.page_content for doc in docs)
        step.output += "Web content loaded successfully..."
        cl.user_session.set("web_content", content)
        return {"web_content": content}
    except Exception as e:
        step.output = f"Error loading web content: {str(e)}"
        return f"Error loading web content: {str(e)}"

@tool
async def extract_data(query: str) -> str:
    """
    This tool is used to extract the relevant data from the log file.
    It will find the most relevant log message type(s) and the most relevant list of fields to the user query,
    read the data of these message types and fields from the file, and return the results.
    """
    
    async with cl.Step(name="data extraction tool to find relevant data", type="tool") as step:
        try:
            # Add thinking process to the step
            step.output = "Starting data extraction process...\n"
            
            user_id = get_user_id()        
            headers = {"user-id": user_id}
            response = requests.get(f"{base_url}/api/files", headers=headers)
            
            step.output += "Retrieved file information from API\n"
            
            file_id = ""
            if response.status_code == 200:
                file_data = response.json()
                file_path = file_data.get("file_path", "")
                file_id = file_data.get("file_id", "")
                if file_path:
                    step.output += f"Using uploaded file: {file_path}\n"
                else:
                    step.output += "No file uploaded. Please upload a log file first.\n"
                    return f"No file uploaded. Please upload a log file first."
            else:
                step.output += f"API request failed with status {response.status_code}\n"
                return f"API request failed with status {response.status_code}: {response.text}"
            
            if cl.user_session.get("file_id") != file_id:
                step.output += "New file detected, fetching message schema...\n"
                
                # Get schema from API instead of reading file directly
                schema_response = requests.get(f"{base_url}/api/files/{file_id}/schema", headers=headers)
                
                if schema_response.status_code == 200:
                    schema_data = schema_response.json()
                    schema = schema_data["schema"]
                    
                    # Format for msg_context
                    lines = []
                    for msg_type, fields in sorted(schema.items()):
                        lines.append(f"Log message type: {msg_type}")
                        lines.append(f"Fields: {fields}")
                        lines.append("")
                    
                    msg_context = "\n".join(lines)
                    cl.user_session.set("msg_context", msg_context)
                    cl.user_session.set("data", {})
                    cl.user_session.set("file_id", file_id)
                else:
                    return "Failed to get file schema from API"
            else:
                msg_context = cl.user_session.get("msg_context")

            # Step 2: Extract column mapping
            step.output += "Using AI to identify relevant fields for your query...\n"
            step.output += f"Query: '{query}'\n"
            
            template = """
            User query: {query}

            Based on the field descriptions below:
            {web_content}

            Identify the most relevant log message type(s) and the most relevant list of fields within them
            needed to answer the user query. 

            ## IMPORTANT NOTE: 
            You don't have to call any of these tools all the time. Sometimes the user might 
            ask a follow up question or ask about something that can be answered from the chat history. 
            In those cases, don't call the tools that would normally be called.
            and just return the answer from the chat history. 
            
            CRITICAL: Call this tool and get all the relevant data based on the user query ONCE in one call.       

            **Which data I should extract if the user asks for the anomalies/issues observed during the flight?**
            Unless the user is specific about the data he wants to check for anomalies/issues, 
            you can check for ERR data and other data types that makes sense to you based on the user query if they are in this list: {msg_context} 
            
            IMPORTANT RULES:  
            - It is VERY IMPORTANT that the log message type(s) and field(s) you return are part of the log message type(s) in the msg_context: {msg_context}.
            The name of the log message type(s) and field(s) in the uploaded file might now always be the same as the ones in the msg_context. 
            For instance, you might see the log message type "ATTITUDE" in the uploaded file while you see "ATT" in the the msg_context. 
            A user might upload a file, ask for the maximum value of a pitch and it can be found in the "ATTITUDE" log message type in one file 
            and in the "ATT" log message type in the next file uploaded. Be careful about these. Always return the log message type(s) and field(s)
            in the msg_context.
            - Respond with ONLY one Python dictionary in this exact format, no extra text or explanation:

            {{'LogMessageType': ['field1', 'field2', ...\n], ...\n}}

            - Replace 'LogMessageType' and field names with your best guesses, based on the provided field descriptions.  
            - Consider relationships between fields. For example, if the query asks for the time when the highest longitude is observed, 
            return both the longitude field and the time field together.  
            - Only include fields necessary to answer the query, avoid irrelevant ones.  
            - Do NOT output placeholders or quotes around keys like 'log message type'.  
            - Do NOT include anything other than the Python dictionary.
            - If you are not sure about the log message types and/or field names, ask the user for clarification.
            """

            web_content = cl.user_session.get("web_content")
            if web_content == "":
                step.output += "No web content available.\n"
            else:
                step.output += "Web content available.\n"
                

            prompt = PromptTemplate(input_variables=["query", "web_content", "msg_context"], template=template)
            # Use the global model that's already configured
            chain = prompt | final_model
            result = chain.invoke({"query": query, "web_content": web_content, "msg_context": msg_context})
            col_map = ast.literal_eval(result.content.strip())
            cl.user_session.set("col_map", col_map)
            
            step.output += f"AI identified relevant fields: {col_map}\n"
            
            # Step 3: Read data using the API endpoint
            step.output += "Extracting data from log file using API...\n"
            
            user_id = get_user_id()
            headers = {"user-id": user_id}
            
            response = requests.post(f"{base_url}/api/process", json=col_map, headers=headers)
                    
            if response.status_code != 200:
                step.output += f"API request failed with status {response.status_code}\n"
                return f"API request failed with status {response.status_code}: {response.text}"
                
            response_data = response.json()
            if not response_data.get("success"):
                step.output += f"API processing failed: {response_data.get('error', 'Unknown error')}\n"
                return f"API processing failed: {response_data.get('error', 'Unknown error')}"


            step.output += "Successfully retrieved data from API\n"
            
            data = response_data["data"]
            
            final_data = {}
            
            step.output += "Processing and cleaning data...\n"
            
            for msg_type, rows in data.items():
                if rows:
                    df = pd.DataFrame(rows)
                    df.dropna(axis=1, how='all', inplace=True) 
                    final_data[msg_type] = df
                    step.output += f"  Processed {len(df)} rows for {msg_type}\n"
            
            step.output += f"Data extraction completed! Extracted {len(final_data)} message types.\n"
            cl.user_session.set("data", final_data)
            return {"data": final_data}
                
        except Exception as e:
            step.output += f"Error occurred: {str(e)}\n"
            return f"Error in extract_data: {str(e)}"

@tool
async def average(data_description: str):
    """
    Calculate the average value of numeric fields in the data.
    """
    async with cl.Step(name="average tool to calculate average values", type="tool") as step:
        step.output = "Starting average calculation process...\n"
        
        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."
        
        step.output += f"Found {len(data)} message types in the extracted data.\n"
        
        result_parts = []
        
        for msg_type, df in data.items():
            step.output += f"Processing {msg_type} with {len(df)} rows...\n"
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                step.output += f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n"
                
                avg_values = numeric_cols.mean()
                result_parts.append(f"Average values in {msg_type}:")
                
                for col, val in avg_values.items():
                    result_parts.append(f"  {col}: {val}")
                    step.output += f"Calculated average for {col}: {val:.6f}\n"
            else:
                step.output += f"No numeric fields found in {msg_type}.\n"
        
        step.output += "Average calculation completed successfully.\n"
        return {"average": "\n".join(result_parts)}

@tool
async def total_sum(data_description: str):
    """
    Calculate the sum of numeric fields in the data.
    
    """
    async with cl.Step(name="sum tool to calculate sum of numeric fields", type="tool") as step:
        step.output = "Starting sum calculation process...\n"
        
        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."
        
        step.output += f"Found {len(data)} message types in the extracted data.\n"
        
        result_parts = []
        
        for msg_type, df in data.items():
            step.output += f"Processing {msg_type} with {len(df)} rows...\n"
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                step.output += f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n"
                
                sum_values = numeric_cols.sum()
                result_parts.append(f"Sum of numeric fields in {msg_type}:")
                
                for col, val in sum_values.items():
                    result_parts.append(f"  {col}: {val}")
                    step.output += f"Calculated sum for {col}: {val}\n"
            else:
                step.output += f"No numeric fields found in {msg_type}.\n"
        
        step.output += "Sum calculation completed successfully.\n"
        return {"sum": "\n".join(result_parts)}

@tool
async def maximum(data_description: str):
    """
    Find the maximum value and when it occurred, including timestamp and context.
    If the user ask for only the maximum value, you can return the maximum value.
    But if the user asks for the maximum value and when it occurred, return the maximum value and when it occurred.
    """
    async with cl.Step(name="maximum tool to find maximum values", type="tool") as step:
        step.output = "Starting maximum value analysis...\n"
        
        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."
        
        step.output += f"Found {len(data)} message types in the extracted data.\n"
        
        result_parts = []
        
        for msg_type, df in data.items():
            step.output += f"Processing {msg_type} with {len(df)} rows...\n"
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"When maximum values occurred in {msg_type}:")
                step.output += f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n"
                
                for col in numeric_cols.columns:
                    max_idx = df[col].idxmax()
                    max_row = df.loc[max_idx]
                    max_value = max_row[col]
                    
                    result_parts.append(f"Maximum {col}: {max_value}")
                    step.output += f"Found maximum {col}: {max_value} at index {max_idx}\n"
                    
                    # Include all available context from that row
                    for field, value in max_row.items():
                        if field != col:  # Don't repeat the max value itself
                            result_parts.append(f"    {field}: {value}")
                    result_parts.append("")  # Add space between fields   
            else:
                step.output += f"No numeric fields found in {msg_type}.\n"
        
        step.output += "Maximum value analysis completed successfully.\n"
        return {"maximum": "\n".join(result_parts)}

@tool
async def minimum(data_description: str):
    """
    Find the minimum value and when it occurred, including timestamp and context.
    If the user ask for only the minimum value, you can return the minimum value.
    But if the user asks for the minimum value and when it occurred, return the minimum value and when it occurred.
    """
    async with cl.Step(name="minimum tool to find minimum values", type="tool") as step:
        step.output = "Starting minimum value analysis...\n"
        
        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."
        
        step.output += f"Found {len(data)} message types in the extracted data.\n"
        
        result_parts = []
        
        for msg_type, df in data.items():
            step.output += f"Processing {msg_type} with {len(df)} rows...\n"
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"When minimum values occurred in {msg_type}:")
                step.output += f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n"
                
                for col in numeric_cols.columns:
                    min_idx = df[col].idxmin()
                    min_row = df.loc[min_idx]
                    min_value = min_row[col]
                    
                    result_parts.append(f"  Minimum {col}: {min_value}")
                    step.output += f"Found minimum {col}: {min_value} at index {min_idx}\n"
                    
                    # Include all available context from that row
                    for field, value in min_row.items():
                        if field != col:  # Don't repeat the min value itself
                            result_parts.append(f"    {field}: {value}")
                    result_parts.append("")  # Add space between fields    
            else:
                step.output += f"No numeric fields found in {msg_type}.\n"
        
        step.output += "Minimum value analysis completed successfully.\n"
        return {"minimum": "\n".join(result_parts)} 

@tool
async def detect_oscillations(data_description: str):
    """
    Detect oscillatory patterns in the data, including periodic fluctuations and recurring cycles.
    """
    async with cl.Step(name="oscillation detection tool to detect oscillations", type="tool") as step:
        step.output = "Starting oscillation detection process...\n"
        
        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."     

        step.output += f"Found {len(data)} message types in the extracted data.\n"
        
        result_parts = []
        
        for msg_type, df in data.items():
            step.output += f"Processing {msg_type} with {len(df)} rows...\n"
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"Oscillation analysis for {msg_type}:")
                step.output += f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n"
                
                # Sort by timestamp if available
                sorted_df = df.copy()
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    sorted_df = df.sort_values(by=timestamp_cols[0])
                    step.output += f"Sorted data by {timestamp_cols[0]} for oscillation analysis.\n"
                
                for col in numeric_cols.columns:
                    step.output += f"Analyzing oscillations in {col}...\n"
                    
                    values = sorted_df[col].dropna()
                    if len(values) < 6:  # Need minimum points for oscillation detection
                        result_parts.append(f"  {col}: Insufficient data points for oscillation analysis")
                        continue
                    
                    # Method 1: Detect direction changes (peaks and troughs)
                    direction_changes = []
                    directions = []
                    
                    for i in range(1, len(values)):
                        if values.iloc[i] > values.iloc[i-1]:
                            directions.append('up')
                        elif values.iloc[i] < values.iloc[i-1]:
                            directions.append('down')
                        else:
                            directions.append('stable')
                    
                    # Count direction changes
                    changes = 0
                    for i in range(1, len(directions)):
                        if directions[i] != directions[i-1] and directions[i] != 'stable' and directions[i-1] != 'stable':
                            changes += 1
                            direction_changes.append(i)
                    
                    # Method 2: Calculate standard deviation and mean for variability
                    std_dev = values.std()
                    mean_val = values.mean()
                    coefficient_of_variation = std_dev / mean_val if mean_val != 0 else 0
                    
                    # Method 3: Detect local maxima and minima
                    peaks = []
                    troughs = []
                    
                    for i in range(1, len(values) - 1):
                        if values.iloc[i] > values.iloc[i-1] and values.iloc[i] > values.iloc[i+1]:
                            peaks.append((values.index[i], values.iloc[i]))
                        elif values.iloc[i] < values.iloc[i-1] and values.iloc[i] < values.iloc[i+1]:
                            troughs.append((values.index[i], values.iloc[i]))
                    
                    # Method 4: Calculate approximate frequency
                    total_cycles = (len(peaks) + len(troughs)) / 2
                    data_length = len(values)
                    
                    # Oscillation assessment
                    oscillation_score = 0
                    oscillation_indicators = []
                    
                    # High number of direction changes indicates oscillation
                    if changes > data_length * 0.3:  # More than 30% direction changes
                        oscillation_score += 2
                        oscillation_indicators.append(f"High direction changes: {changes}")
                    
                    # High coefficient of variation indicates variability
                    if coefficient_of_variation > 0.2:  # 20% variation
                        oscillation_score += 1
                        oscillation_indicators.append(f"High variability (CV: {coefficient_of_variation:.2f})")
                    
                    # Significant peaks and troughs
                    if len(peaks) >= 2 and len(troughs) >= 2:
                        oscillation_score += 2
                        oscillation_indicators.append(f"Multiple peaks ({len(peaks)}) and troughs ({len(troughs)})")
                    
                    # Regular spacing between peaks/troughs (if enough data)
                    if len(peaks) >= 3:
                        peak_intervals = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks)-1)]
                        if len(set(peak_intervals)) <= len(peak_intervals) * 0.5:  # Similar intervals
                            oscillation_score += 1
                            oscillation_indicators.append("Regular peak intervals detected")
                    
                    # Report findings
                    result_parts.append(f"  {col} oscillation analysis:")
                    result_parts.append(f"    Oscillation score: {oscillation_score}/6")
                    result_parts.append(f"    Direction changes: {changes} out of {len(directions)} transitions")
                    result_parts.append(f"    Peaks found: {len(peaks)}")
                    result_parts.append(f"    Troughs found: {len(troughs)}")
                    result_parts.append(f"    Coefficient of variation: {coefficient_of_variation:.3f}")
                    
                    if oscillation_score >= 3:
                        result_parts.append(f"    ⚠️  OSCILLATION DETECTED - Strong oscillatory pattern")
                        step.output += f"Strong oscillation detected in {col} (score: {oscillation_score}).\n"
                        
                        # Show key oscillation points
                        if peaks:
                            result_parts.append(f"    Peak values: {[f'{val:.2f}' for _, val in peaks[:5]]}")
                        if troughs:
                            result_parts.append(f"    Trough values: {[f'{val:.2f}' for _, val in troughs[:5]]}")
                            
                        # Show timestamps of oscillations if available
                        if timestamp_cols:
                            oscillation_times = []
                            for idx, _ in (peaks + troughs)[:5]:
                                time_val = sorted_df.loc[idx, timestamp_cols[0]]
                                oscillation_times.append(str(time_val))
                            result_parts.append(f"    Key oscillation times: {oscillation_times}")
                    
                    elif oscillation_score >= 1:
                        result_parts.append(f"    ℹ️  WEAK OSCILLATION - Some oscillatory characteristics")
                        step.output += f"Weak oscillation detected in {col} (score: {oscillation_score}).\n"
                    else:
                        result_parts.append(f"    ✅ NO OSCILLATION - Data appears stable/trending")
                        step.output += f"No significant oscillation in {col}.\n"
                    
                    # Add detailed indicators
                    if oscillation_indicators:
                        result_parts.append(f"    Indicators: {', '.join(oscillation_indicators)}")
                    
                    result_parts.append("")  # Space between columns
                
                result_parts.append("")  # Space between message types
            else:
                step.output += f"No numeric fields found in {msg_type}.\n"
                result_parts.append(f"No numeric data available in {msg_type} for oscillation analysis.")
                result_parts.append("")
        
        step.output += "Oscillation detection completed successfully.\n"
        return {"oscillations": "\n".join(result_parts)}        

@tool
async def detect_sudden_changes(data_description: str):
    """
    Detect sudden changes in the data.
    """
    async with cl.Step(name="sudden changes tool to detect anomalies", type="tool") as step:
        step.output = "Starting sudden changes detection process...\n"
        
        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."     

        step.output += f"Found {len(data)} message types in the extracted data.\n"
        
        result_parts = []
        
        for msg_type, df in data.items():
            step.output += f"Processing {msg_type} with {len(df)} rows...\n"
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"Sudden changes detected in {msg_type}:")
                step.output += f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n"
                
                # Sort by timestamp if available
                sorted_df = df.copy()
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    sorted_df = df.sort_values(by=timestamp_cols[0])
                    step.output += f"Sorted data by {timestamp_cols[0]} for change detection.\n"
                
                for col in numeric_cols.columns:
                    step.output += f"Analyzing sudden changes in {col}...\n"
                    
                    # Calculate percentage changes between consecutive values
                    values = sorted_df[col].dropna()
                    if len(values) < 2:
                        continue
                    
                    pct_changes = values.pct_change().fillna(0)
                    
                    threshold = 0.5  # 50% change
                    sudden_changes = pct_changes[abs(pct_changes) > threshold]
                    
                    if len(sudden_changes) > 0:
                        result_parts.append(f"  {col} sudden changes (>{threshold*100}%):")
                        step.output += f"Found {len(sudden_changes)} sudden changes in {col}.\n"
                        
                        for idx in sudden_changes.index:
                            change_row = sorted_df.loc[idx]
                            prev_idx = values.index[values.index.get_loc(idx) - 1] if values.index.get_loc(idx) > 0 else None
                            
                            current_value = change_row[col]
                            change_pct = sudden_changes[idx] * 100
                            
                            result_parts.append(f"    Change: {change_pct:.1f}% to {current_value}")
                            
                            # Include context from the row where change occurred
                            for field, value in change_row.items():
                                if field != col:
                                    result_parts.append(f"      {field}: {value}")
                            
                            # Add previous value context if available
                            if prev_idx is not None:
                                prev_value = sorted_df.loc[prev_idx, col]
                                result_parts.append(f"      Previous value: {prev_value}")
                            
                            result_parts.append("")  # Add space between changes
                    else:
                        result_parts.append(f"  No sudden changes detected in {col}")
                        step.output += f"No sudden changes detected in {col}.\n"
                
                result_parts.append("")  # Add space between message types
            else:
                step.output += f"No numeric fields found in {msg_type}.\n"
                result_parts.append(f"No numeric data available in {msg_type} for change detection.")
        
        step.output += "Sudden changes detection completed successfully.\n"
        return {"sudden_changes": "\n".join(result_parts)}

@tool
async def visualize(query: str):
    """
    Visualize the data.
    """
    async with cl.Step(name="visualize tool to create data visualization", type="tool") as step:
        step.output = "Starting visualization process...\n"

        data = filter_data()
        if not data:
            step.output += "No data available in session. Please extract data first.\n"
            return "No data available. Please extract data first."
        
        step.output += f"Found {len(data)} message types in the extracted data.\n"
        step.output += "Preparing data for visualization by sampling records...\n"
        
        # The data has multiple dataframes, so we need to sample from each one.
        keys = []
        for key in data:
            keys.append(key)

        plt.rcParams.update({'figure.dpi': 150,})                

        step.output += "Generating visualization code using AI model...\n"

        template = """
        Find the key that is most relevant for the chat history: {chat_history}.
        Here is the available keys: {keys}.

        Only give the key, nothing else.
        """

        prompt = PromptTemplate(input_variables=["chat_history", "keys"], template=template)
        # Use the global model that's already configured
        chain = prompt | model
        chat_history = cl.user_session.get("message_history")
        result = chain.invoke({"chat_history": chat_history, "keys": keys})
        key = result.content.strip()

        step.output += f"Found the key: {key}\n"
            
        sample_size = min(100, len(data[key]))
        sampled_data = data[key].sample(sample_size, replace=True)
        step.output += f"Sampled {sample_size} rows from {key} dataset.\n"
        
        template = """
        I have a dataframe in a dictionary called `data` that can be accessed using data[key].
        Here is the available key: {key}.

        Here is how the 100 rows sampled from the data looks like: {sampled_data}.

        Write a Plotly function in Python to visualize the data[key] so that I can execute it and get the plot.
        Make sure that the plot is relevant to the user's query: {query}.
        
        Only give the code, nothing else. Don't include ```python or ``` or anything else. 
        Don't explain the data.

        IMPORTANT NOTES 
        - Make the plot look nice, readable, and high quality. 
        - Don't use any other libraries than matplotlib, numpy, pandas, datetime, and other standard libraries
        that are already installed in the system.
        - Make sure that the code you generate can be run with 1 click without needing any modification/change.
        - `data` already exists, don't create a new one!
        - Do not include any Python code in your response. 
        - Don't remove a file/folder, create a new one, or do anything else that might affect the existing files/folders.
        - Don't install a new library or uninstall the existing ones.
        """
        
        col_map = cl.user_session.get("col_map")
        prompt = PromptTemplate(input_variables=["query", "sampled_data", "key"], template=template)
        # Use the global model that's already configured
        chain = prompt | final_model
        result = chain.invoke({"query": query, "sampled_data": sampled_data, "key": key})
        code = result.content.strip()
        code = code.replace("plt.show()", "")  
        
        step.output += "AI model successfully generated visualization code.\n"
        step.output += "Code has been prepared for execution.\n"
        
        # Execute the visualization code
        cl.user_session.set("code", code)

        step.output += f"Visualization ready using data from message types: {list(col_map.keys())}\n"
        step.output += "Visualization process completed successfully.\n"

        return f"Successfully generated the visualization code using the following message type and fields: {col_map}"

def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "final"

async def call_model(state: MessagesState):
    messages = state["messages"]
    
    # Add system message with clear instructions
    system_message = SystemMessage(content=f"""
    You are an assistant that analyzes flight log data. You have access to several tools.

    AVAILABLE TOOLS:
    Call extract_data tool FIRST when users ask about:
    - Anomalies/issues in the data
    - Maximum/minimum values
    - Average values
    - Sum calculations
    - Visualizations
    - Any analysis questions about the log data

    IMPORTANT RULES:
    - When they ask about anomalies, you can use the detect_sudden_changes and detect_oscillations tools to find the sudden changes and oscillations in the data and 
    then interpret whether they are anomalies or not.
    - You don't have to call any of these tools all the time. Sometimes the user might
    ask a follow up question or ask about something that can be answered from the chat history. 
    In those cases, don't call the tools that would normally be called.
    and just return the answer from the chat history. 
    - If what users asks for in their query is not available in the schema of the existing data, 
    you can call the extract_data tool to get the right data.""" + 
    
    f"Here is the schema of the existing data: {cl.user_session.get('col_map', {})} and here is the chat history: {cl.user_session.get('message_history', [])}")
    
    # Add system message to the beginning of messages if it's not already there
    messages_with_system = [system_message] + messages
    
    # Use the model that's already bound to tools (defined globally)
    response = await model.ainvoke(messages_with_system)
    return {"messages": [response]}
    

async def call_final_model(state: MessagesState):
    messages = state["messages"]    
    last_ai_message = messages[-1]

    response = await final_model.ainvoke(
        [
            SystemMessage("""
            Rewrite this in an organized, clean, readable and nice format. Don't just give pure numbers. Interpret them as well.

            If there are oscillations and sudden changes, think about whether they can be seen as anomalies/issues or not depending on the context.
            """),
            HumanMessage(last_ai_message.content),
        ]
    )
    
    return {"messages": [response]}

# Add recall_conversation to the tools list
# Attention: Anomalies tool is not necessary. 
tools = [load_web_content, extract_data, maximum, minimum, average, total_sum, visualize, detect_sudden_changes, detect_oscillations]
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=1000)
final_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=1.0, max_tokens=1000)

model = model.bind_tools(tools)
final_model = final_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", call_final_model)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")
builder.add_edge("final", END)

graph = builder.compile()

with open("user.json", "r") as f:
    users = json.load(f)

user_map = {user["user_id"]: user for user in users}

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    user = user_map.get(username)
    
    if user and hashlib.sha256(password.encode()).hexdigest() == user["password_hash"]:
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )
    
    return None

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("msg_context", "")
    cl.user_session.set("file_id", "")
    cl.user_session.set("web_content", "")
    cl.user_session.set("data", {})
    cl.user_session.set("message_history", [])

@cl.on_message
async def on_message(msg: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append(HumanMessage(content=msg.content))
    
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    async for msg, metadata in graph.astream({"messages": message_history}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (msg.content and not isinstance(msg, HumanMessage) and metadata["langgraph_node"] == "final"):
            await final_answer.stream_token(msg.content)
    
    await final_answer.send()
    
    # Store the AI response content as an AIMessage
    message_history.append(AIMessage(content=final_answer.content))

    cl.user_session.set("message_history", message_history)

    code = cl.user_session.get("code")
    if code:
        # Get the data from user session to make it available in exec scope
        data = cl.user_session.get("data")
        
        # Execute the code with the data variable available
        exec(code)
        
        # Get the current figure and create cl.Pyplot element
        fig = plt.gcf()
        fig.set_dpi(300)
        
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
        cl.user_session.set("code", "") 

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")
    
@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")