import ast
import asyncio
import json
import os
import time
import chainlit as cl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from models import *

matplotlib.use("Agg")
load_dotenv()

# API base URL
api_host = os.getenv("API_HOST")
api_port = os.getenv("API_PORT")
if api_host and api_port:
    base_url = f"http://{api_host}:{api_port}"
else:
    base_url = os.getenv("API_BASE_URL")


def get_user_id():
    """Get user ID from Chainlit session"""
    user = cl.user_session.get("user")
    user_id = user.identifier if user and hasattr(user, "identifier") else "anonymous"
    return user_id

def filter_data() -> dict:
    """
    Filter the data based on the col_map.
    """
    filtered_data = {}
    try:
        data = cl.user_session.get("data")
        col_map = cl.user_session.get("col_map")
        for msg_type, rows in data.items():
            if msg_type in col_map:
                filtered_data[msg_type] = rows[col_map[msg_type]]
    except Exception as e:
        print(f"Error filtering data: {str(e)}")
        return {}
    return filtered_data


# LLM provider setup (env var selects backend: anthropic default, or openai)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

if LLM_PROVIDER == "openai":
    _base_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, max_tokens=2000, streaming=True)
    qa_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, max_tokens=2000, streaming=True)
else:
    _base_model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0, max_tokens=2000, streaming=True)
    qa_model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2, max_tokens=2000, streaming=True)

# Plain-text model: used for extraction tasks where the LLM should return raw text/dict/code,
# not call tools. Claude is strict about tool-binding — using a tool-bound model here causes
# empty or malformed responses.
extract_model = _base_model

@tool
async def load_web_content(url: str) -> WebContentResult:
    """
    Load web content from the given URL.
    Use this when user provides a URL and wants to extract content from it.
    """
    try:
        async with cl.Step(name="", type="tool") as step:
            step.name = "Loading web content from the URL."
            await step.update()
            
            if not url:
                await step.stream_token("No URL provided")
                step.name = "Web content loading failed."
                await step.update()
                return WebContentResult(web_content="No URL provided")
            
            # Attention: Web content is forgotton after the query is answered. 
            app = Firecrawl()
            docs = app.scrape(url)
            content = docs.markdown
            await step.stream_token("Web content loaded successfully...")
            cl.user_session.set("web_content", content)
            # Update step name to show completion
            step.name = "Web content loading is done."
            await step.update()
            return WebContentResult(web_content=content)
    except Exception as e:
        await step.stream_token(f"Error loading web content: {str(e)}")
        step.name = "Web content loading failed."
        await step.update()
        return WebContentResult(web_content=f"Error loading web content: {str(e)}")


@tool
async def extract_data(query: str) -> DataExtractionResult:
    """
    This tool is used to extract the relevant data from the log file.
    It will find the most relevant log message type(s) and the most relevant list of fields to the user query,
    read the data of these message types and fields from the file, and return the results.
    """
    
    async with cl.Step(name="Starting data extraction process", type="tool") as step:
        try:
            # Step 1: Check for uploaded file
            await step.stream_token("Starting data extraction process...\n")
            
            user_id = get_user_id()        
            headers = {"user-id": user_id}
            response = requests.get(f"{base_url}/api/files", headers=headers)
            
            await step.stream_token("Retrieved file information from API\n")
            
            file_id = ""
            if response.status_code == 200:
                file_data = response.json()
                file_info = FileInfo(**file_data)
                file_path = file_info.file_path
                file_id = file_info.file_id
                if file_path:
                    await step.stream_token(f"Using uploaded file: {file_path}\n")
                else:
                    await step.stream_token("No file uploaded. Please upload a log file first.\n")
                    return DataExtractionResult(data={"error": "No file uploaded. Please upload a log file first."})
            else:
                await step.stream_token(f"API request failed with status {response.status_code}\n")
                return DataExtractionResult(data={"error": f"API request failed with status {response.status_code}: {response.text}"})
            
            # Step 2: Fetch or retrieve schema
            if cl.user_session.get("file_id") != file_id:
                await step.stream_token("New file detected, fetching message schema...\n")
                
                # Get schema from API instead of reading file directly
                schema_response = requests.get(f"{base_url}/api/files/{file_id}/schema", headers=headers)
                
                if schema_response.status_code == 200:
                    schema_data = schema_response.json()
                    schema_response_model = SchemaData(**schema_data)
                    schema = schema_response_model.schema
                    
                    # Format for msg_context
                    lines = []
                    for msg_type, fields in sorted(schema.items()):
                        lines.append(f"Log message type: {msg_type}")
                        lines.append(f"Fields: {fields}")
                        lines.append("")
                    
                    msg_context = "\n".join(lines)
                    cl.user_session.set("msg_context", msg_context)
                    cl.user_session.set("data", {})
                    cl.user_session.set("col_map", {})                    
                    cl.user_session.set("code", "")
                    cl.user_session.set("data", {})                    
                    cl.user_session.set("file_id", file_id)
                else:
                    return DataExtractionResult(data={"error": "Failed to get file schema from API"})
            else:
                msg_context = cl.user_session.get("msg_context")

            # Step 3: AI Analysis for column mapping
            await step.stream_token("Using AI to identify relevant data for your query...\n")
            await step.stream_token(f"User query: '{query}'\n")
            
            template = """
            Based on the user query: {query}, identify the most relevant log message type(s) 
            and the most relevant list of fields within them needed to answer the user query. 

            The log message type(s) and field(s) should be part of {msg_context}.

            If it is available, you can use the following web content to help you 
            identify the most relevant log message type(s) and then the most relevant list of fields within them 
            needed to answer the user query: {web_content}

            **Which log message type(s) and field(s) I should extract if the user asks for the anomalies/issues observed during the flight?**
            Unless the user is specific about the log message type(s) and field(s) he wants to check for anomalies/issues, 
            you can check for ERR log message type and other log message types that make sense to you based on the user query 
            if they are in this list: {msg_context} 
            
            IMPORTANT RULES:  
            - The `extract_data` tool is LIMITED to a maximum of {max_message_types} LogMessageTypes. 
            If you need more, extract the most relevant ones or make multiple requests.

            - It is VERY IMPORTANT that the log message type(s) and field(s) you return are 
            part of the log message type(s) in the `msg_context`: {msg_context}.
            
            - Respond with ONLY one Python dictionary in the exact format below, no extra text or explanation:

            {{'LogMessageType': ['field1', 'field2', ...\n], ...\n}}

            - Replace 'LogMessageType' and field names with your best guesses in {msg_context}, 
            based on the provided field descriptions.  

            - Consider relationships between fields. For example, if the query asks for the time when the highest longitude is observed, 
            return both the longitude field and the time field together.  

            - For multiple related values in the same query, group them in the same dictionary entry when they belong to the same message type.

            - Only include fields necessary to answer the query, avoid irrelevant ones.  

            - Do NOT output placeholders or quotes around keys like 'log message type'.  

            - Do NOT include anything other than the Python dictionary.
            
            EXAMPLES:
            - Query: "What is the average roll and pitch values?" → {{'ATT': ['Roll', 'Pitch']}}
            - Query: "Show me GPS latitude and longitude" → {{'GPS': ['Lat', 'Lng']}}
            - Query: "What are the maximum altitude and speed?" → {{'GPS': ['Alt'], 'VEL': ['Spd']}} (if in different message types)
            """

            web_content = cl.user_session.get("web_content")
            if web_content == "":
                await step.stream_token("No web content available.\n")
            else:
                await step.stream_token("Web content available.\n")
                
            prompt = PromptTemplate(
                input_variables=["query", "web_content", "msg_context", "max_message_types"],
                template=template
            )
            # Plain-text extraction — use the un-bound model so Claude doesn't try to call tools
            chain = prompt | extract_model
            result = chain.invoke(
                {
                    "query": query,
                    "web_content": web_content,
                    "msg_context": msg_context,
                    "max_message_types": os.getenv("MAX_MESSAGE_TYPES", 3)
                }
            )
            await step.stream_token(f"AI identified relevant fields: {result.content.strip()}\n")
            if result.content.strip() != "":
                col_map = ast.literal_eval(result.content.strip())
                
                # Validate the col_map using ColMapRequest model directly
                try:
                    ColMapRequest(col_map=col_map)
                    cl.user_session.set("col_map", col_map)
                except Exception as e:
                    await step.stream_token(f"Invalid column mapping format: {str(e)}. Please try again.\n")
                    step.name = "Invalid column mapping format."
                    await step.update()
                    return DataExtractionResult(data={"error": f"Invalid column mapping format: {str(e)}. Please try again."})
            else:
                await step.stream_token("No relevant fields found.\n")
                step.name = "No relevant fields found."
                await step.update()
                return DataExtractionResult(data={"error": "No relevant fields found."})
            
            await step.stream_token(f"AI identified relevant fields: {col_map}\n")
            
            # Step 4: Extract data using the API endpoint
            await step.stream_token("Extracting data from log file using API...\n")
            
            # In the extract_data function, replace this:
            user_id = get_user_id()
            headers = {"user-id": user_id}

            # Validate col_map using ColMapRequest model and send the request
            process_request = ColMapRequest(col_map=col_map)
            response = requests.post(f"{base_url}/api/process", json=process_request.dict(), headers=headers)
                    
            if response.status_code != 200:
                await step.stream_token(f"API request failed with status {response.status_code}\n")
                await step.stream_token(f"Here is the extracted col_map: {col_map}\n")
                return DataExtractionResult(data={"error": f"API request failed with status {response.status_code}: {response.text}"})
                
            response_data = response.json()
            process_response = ProcessResponse(**response_data)
            if not process_response.success:
                await step.stream_token(f"API processing failed: {process_response.error or 'Unknown error'}\n")
                return DataExtractionResult(data={"error": f"API processing failed: {process_response.error or 'Unknown error'}"})

            await step.stream_token("Successfully retrieved data from API\n")
            data = process_response.data or {}
            
            # Step 5: Process and clean the data
            final_data = {}
            
            await step.stream_token("Processing and cleaning data...\n")
            
            for msg_type, rows in data.items():
                if rows:
                    df = pd.DataFrame(rows)
                    df.dropna(axis=1, how='all', inplace=True) 
                    final_data[msg_type] = df
                    await step.stream_token(f"  Processed {len(df)} rows for {msg_type}\n")
            
            await step.stream_token(f"Data extraction completed! Extracted {len(final_data)} message types.\n")
            cl.user_session.set("data", final_data)
            
            # Update step name to show completion
            step.name = "Data extraction process is done."
            await step.update()
            return DataExtractionResult(data=final_data)
                
        except Exception as e:
            await step.stream_token(f"Error occurred: {str(e)}\n")
            # Update step name to show error
            step.name = "Data extraction process failed."
            await step.update()
            return DataExtractionResult(data={"error": f"Error in extract_data: {str(e)}"})


@tool
async def average(data_description: str) -> AverageResult:
    """
    Calculate the average value of numeric fields in the data.
    """
    async with cl.Step(name="Starting average calculation process", type="tool") as step:
        await step.stream_token("Starting average calculation process...\n")
        
        data = filter_data()
        if not data:
            await step.stream_token("No data available in session. Please extract data first.\n")
            return AverageResult(
                message="No data available",
                average="No data available. Please extract data first."
            )
        
        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                avg_values = numeric_cols.mean()
                result_parts.append(f"Average values in {msg_type}:")
                
                for col, val in avg_values.items():
                    result_parts.append(f"  {col}: {val}")
                    await step.stream_token(f"Calculated average for {col}: {val:.6f}\n")
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
        
        await step.stream_token("Average calculation completed successfully.\n")
        
        step.name = "Average calculation process is done."
        await step.update()
        return AverageResult(
            message="Average calculation completed successfully",
            average="\n".join(result_parts)
        )

@tool
async def total_sum(data_description: str) -> SumResult:
    """
    Calculate the sum of numeric fields in the data.
    
    """
    async with cl.Step(name="Starting sum calculation process", type="tool") as step:
        await step.stream_token("Starting sum calculation process...\n")
        
        data = filter_data()
        if not data:
            await step.stream_token("No data available in session. Please extract data first.\n")
            return SumResult(
                message="No data available",
                sum="No data available. Please extract data first."
            )
        
        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                sum_values = numeric_cols.sum()
                result_parts.append(f"Sum of numeric fields in {msg_type}:")
                
                for col, val in sum_values.items():
                    result_parts.append(f"  {col}: {val}")
                    await step.stream_token(f"Calculated sum for {col}: {val}\n")
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
        
        await step.stream_token("Sum calculation completed successfully.\n")
        
        step.name = "Sum calculation process is done."
        await step.update()
        return SumResult(
            message="Sum calculation completed successfully",
            sum="\n".join(result_parts)
        )

@tool
async def maximum(data_description: str) -> MaximumResult:
    """
    Find the maximum value and when it occurred, including timestamp and context.
    If the user ask for only the maximum value, you can return the maximum value.
    But if the user asks for the maximum value and when it occurred, return the maximum value and when it occurred.
    """
    async with cl.Step(name="Starting maximum value analysis", type="tool") as step:
        data = filter_data()
        if not data:
            step.name = "No data available for maximum value analysis."
            await step.update()
            return MaximumResult(
                message="No data available",
                maximum="No data available. Please extract data first."
            )
        
        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"When maximum values occurred in {msg_type}:")
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                for col in numeric_cols.columns:
                    max_idx = df[col].idxmax()
                    max_row = df.loc[max_idx]
                    max_value = max_row[col]
                    
                    result_parts.append(f"Maximum {col}: {max_value}")
                    await step.stream_token(f"Found maximum {col}: {max_value} at index {max_idx}\n")
                    
                    # Include all available context from that row
                    for field, value in max_row.items():
                        if field != col:  # Don't repeat the max value itself
                            result_parts.append(f"    {field}: {value}")
                    result_parts.append("")  # Add space between fields   
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
        
        await step.stream_token("Maximum value analysis completed successfully.\n")
        
        step.name = "Maximum value analysis is done."
        await step.update()
        return MaximumResult(
            message="Maximum value analysis completed successfully",
            maximum="\n".join(result_parts)
        )


@tool
async def minimum(data_description: str) -> MinimumResult:
    """
    Find the minimum value and when it occurred, including timestamp and context.
    If the user ask for only the minimum value, you can return the minimum value.
    But if the user asks for the minimum value and when it occurred, return the minimum value and when it occurred.
    """
    async with cl.Step(name="Starting minimum value analysis", type="tool") as step:
        data = filter_data()
        if not data:
            step.name = "No data available for minimum value analysis."
            await step.update()
            return MinimumResult(
                message="No data available",
                minimum="No data available. Please extract data first."
            )
        
        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"When minimum values occurred in {msg_type}:")
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                for col in numeric_cols.columns:
                    min_idx = df[col].idxmin()
                    min_row = df.loc[min_idx]
                    min_value = min_row[col]
                    
                    result_parts.append(f"  Minimum {col}: {min_value}")
                    await step.stream_token(f"Found minimum {col}: {min_value} at index {min_idx}\n")
                    
                    # Include all available context from that row
                    for field, value in min_row.items():
                        if field != col:  # Don't repeat the min value itself
                            result_parts.append(f"    {field}: {value}")
                    result_parts.append("")  # Add space between fields    
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
        
        await step.stream_token("Minimum value analysis completed successfully.\n")
        
        step.name = "Minimum value analysis is done."
        await step.update()
        return MinimumResult(
            message="Minimum value analysis completed successfully",
            minimum="\n".join(result_parts)
        ) 



@tool
async def detect_oscillations(data_description: str) -> OscillationResult:
    """
    Detect oscillatory patterns in the data, including periodic fluctuations and recurring cycles.
    """
    async with cl.Step(name="Starting oscillation detection process", type="tool") as step:
        data = filter_data()
        if not data:
            step.name = "No data available for oscillation detection."
            await step.update()
            return OscillationResult(
                message="No data available",
                oscillations="No data available. Please extract data first."
            )

        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"Oscillation analysis for {msg_type}:")
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                # Sort by timestamp if available
                sorted_df = df.copy()
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    sorted_df = df.sort_values(by=timestamp_cols[0])
                    await step.stream_token(f"Sorted data by {timestamp_cols[0]} for oscillation analysis.\n")
                
                for col in numeric_cols.columns:
                    await step.stream_token(f"Analyzing oscillations in {col}...\n")
                    
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
                    
                    # Method 4: Calculate approximate frequency and totals
                    data_length = len(values)
                    candidate_points = max(data_length - 2, 0)
                    peaks_count = len(peaks)
                    troughs_count = len(troughs)
                    peaks_pct = (peaks_count / candidate_points * 100) if candidate_points else 0.0
                    troughs_pct = (troughs_count / candidate_points * 100) if candidate_points else 0.0
                    
                    
                    # Report findings
                    result_parts.append(f"  {col} oscillation analysis:")
                    result_parts.append(f"    Total samples: {data_length}")
                    result_parts.append(f"    Direction changes: {changes} out of {len(directions)} transitions ({(changes/len(directions)*100 if len(directions) else 0):.1f}%)")
                    result_parts.append(f"    Peaks found: {peaks_count} of {candidate_points} candidate points ({peaks_pct:.1f}%)")
                    result_parts.append(f"    Troughs found: {troughs_count} of {candidate_points} candidate points ({troughs_pct:.1f}%)")
                    result_parts.append(f"    Coefficient of variation: {coefficient_of_variation:.3f}")                    
                    result_parts.append("")  # Space between columns
                
                result_parts.append("")  # Space between message types
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
                result_parts.append(f"No numeric data available in {msg_type} for oscillation analysis.")
                result_parts.append("")
        
        await step.stream_token("Oscillation detection completed successfully.\n")
        
        step.name = "Oscillation detection process is done."
        await step.update()
        return OscillationResult(
            message="Oscillation detection completed successfully",
            oscillations="\n".join(result_parts)
        )        

@tool
async def detect_sudden_changes(data_description: str) -> SuddenChangesResult:
    """
    Detect sudden changes in the data.
    """
    async with cl.Step(name="Starting change detection process", type="tool") as step:
        data = filter_data()
        if not data:
            step.name = "No data available for sudden changes detection."
            await step.update()
            return SuddenChangesResult(
                message="No data available",
                sudden_changes="No data available. Please extract data first."
            )

        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                # Sort by timestamp if available
                sorted_df = df.copy()
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    sorted_df = df.sort_values(by=timestamp_cols[0])
                    await step.stream_token(f"Sorted data by {timestamp_cols[0]} for change detection.\n")
                
                for col in numeric_cols.columns:
                    await step.stream_token(f"Analyzing sudden changes in {col}...\n")
                    
                    # Get clean numeric values and calculate percentage changes
                    values = sorted_df[col].dropna()
                    if len(values) < 2:
                        await step.stream_token(f"Insufficient data points in {col} for change detection.\n")
                        continue
                    
                    pct_changes = values.pct_change().fillna(0)
                    threshold = 0.5  # 50% change
                    
                    # Find sudden changes using consistent indexing
                    sudden_changes_count = 0
                    total_transitions = len(values) - 1
                                        
                    for i in range(1, len(values)):
                        change_pct = pct_changes.iloc[i]
                        
                        if abs(change_pct) > threshold:
                            sudden_changes_count += 1
                            
                            # Get indices for current and previous values
                            current_idx = values.index[i]
                            prev_idx = values.index[i-1]
                            
                            # Get the actual values
                            current_value = values.iloc[i]
                            prev_value = values.iloc[i-1]
                            change_percentage = change_pct * 100
                            
                            # Get the full row for context
                            change_row = sorted_df.loc[current_idx]
                            
                            result_parts.append(f"  Change: {change_percentage:.1f}% (from {prev_value} to {current_value}) in {col} column")
                            
                            # Include context from the row where change occurred
                            for field, value in change_row.items():
                                if field != col and pd.notna(value):
                                    result_parts.append(f"    {field}: {value}")
                            
                            result_parts.append("")  # Add space between changes
                    
                    # Summary for this column
                    if sudden_changes_count > 0:
                        pct_sudden = (sudden_changes_count / total_transitions * 100) if total_transitions else 0
                        result_parts.append(f"  Summary: {sudden_changes_count} sudden changes (>{threshold*100}%) out of {total_transitions} transitions ({pct_sudden:.1f}%) in {col} column")
                        await step.stream_token(f"Found {sudden_changes_count} sudden changes in {col}.\n")
                    else:
                        result_parts.append(f"  No sudden changes detected in {col} (0 of {total_transitions} transitions) in {col} column")
                        await step.stream_token(f"No sudden changes detected in {col} in {col} column.\n")
                    
                    result_parts.append("")  # Add space between columns
                
                result_parts.append("")  # Add space between message types
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
                result_parts.append(f"No numeric data available in {msg_type} for change detection.")
                result_parts.append("")
        
        await step.stream_token("Sudden changes detection completed successfully.\n")
        
        step.name = "Sudden changes detection process is done."
        await step.update()
        return SuddenChangesResult(
            message="Sudden changes detection completed successfully",
            sudden_changes="\n".join(result_parts)
        )

@tool
async def detect_outliers(data_description: str) -> OutlierResult:
    """
    Detect statistical outliers in the data using multiple detection methods.
    Identifies data points that deviate significantly from normal patterns.
    """
    async with cl.Step(name="Starting outlier detection process", type="tool") as step: 
        data = filter_data()
        if not data:
            step.name = "No data available for outlier detection."
            await step.update()
            return OutlierResult(
                message="No data available",
                outliers="No data available. Please extract data first."
            )

        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        
        result_parts = []
        
        for msg_type, df in data.items():
            await step.stream_token(f"Processing {msg_type} with {len(df)} rows...\n")
            
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                result_parts.append(f"Outlier analysis for {msg_type}:")
                await step.stream_token(f"Found {len(numeric_cols.columns)} numeric fields in {msg_type}.\n")
                
                # Sort by timestamp if available for better context
                sorted_df = df.copy()
                timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    sorted_df = df.sort_values(by=timestamp_cols[0])
                    await step.stream_token(f"Sorted data by {timestamp_cols[0]} for outlier context.\n")
                
                for col in numeric_cols.columns:
                    await step.stream_token(f"Analyzing outliers in {col}...\n")
                    
                    values = sorted_df[col].dropna()
                    if len(values) < 4:  # Need minimum points for outlier detection
                        result_parts.append(f"  {col}: Insufficient data points for outlier analysis")
                        continue
                    
                    outliers_found = {}
                    all_outlier_indices = set()
                    
                    # Method 1: Interquartile Range (IQR) Method
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers = values[(values < lower_bound) | (values > upper_bound)]
                    if len(iqr_outliers) > 0:
                        outliers_found['IQR'] = iqr_outliers
                        all_outlier_indices.update(iqr_outliers.index)
                        await step.stream_token(f"IQR method found {len(iqr_outliers)} outliers in {col}.\n")
                    
                    # Method 2: Z-Score Method (values beyond 2.5 standard deviations)
                    if len(values) >= 100:  # Z-score needs reasonable sample size
                        mean_val = values.mean()
                        std_val = values.std()
                        
                        if std_val > 0:  # Avoid division by zero
                            z_scores = abs((values - mean_val) / std_val)
                            z_outliers = values[z_scores > 2.5]  # 2.5 standard deviations
                            
                            if len(z_outliers) > 0:
                                outliers_found['Z-Score'] = z_outliers
                                all_outlier_indices.update(z_outliers.index)
                                await step.stream_token(f"Z-Score method found {len(z_outliers)} outliers in {col}.\n")
                    
                    # Method 3: Modified Z-Score using Median Absolute Deviation (MAD)
                    median_val = values.median()
                    mad = (values - median_val).abs().median()
                    
                    if mad > 0:  # Avoid division by zero
                        modified_z_scores = 0.6745 * (values - median_val) / mad
                        mad_outliers = values[abs(modified_z_scores) > 3.5]  # 3.5 MAD threshold
                        
                        if len(mad_outliers) > 0:
                            outliers_found['MAD'] = mad_outliers
                            all_outlier_indices.update(mad_outliers.index)
                            await step.stream_token(f"MAD method found {len(mad_outliers)} outliers in {col}.\n")
                    
                    # Method 4: Percentile Method (beyond 1st and 99th percentiles)
                    p1 = values.quantile(0.01)
                    p99 = values.quantile(0.99)
                    percentile_outliers = values[(values < p1) | (values > p99)]
                    
                    if len(percentile_outliers) > 0:
                        outliers_found['Percentile'] = percentile_outliers
                        all_outlier_indices.update(percentile_outliers.index)
                        await step.stream_token(f"Percentile method found {len(percentile_outliers)} outliers in {col}.\n")
                    
                    # Report findings
                    total_points = len(values)
                    outliers_total = len(all_outlier_indices)
                    outliers_pct = (outliers_total / total_points * 100) if total_points else 0
                    result_parts.append(f"  {col} outlier analysis:")
                    result_parts.append(f"    Total samples: {total_points}")
                    result_parts.append(f"    Data range: {values.min():.3f} to {values.max():.3f}")
                    result_parts.append(f"    Mean: {values.mean():.3f}, Median: {values.median():.3f}")
                    result_parts.append(f"    Standard deviation: {values.std():.3f}")
                    result_parts.append(f"    Total unique outliers found: {outliers_total} of {total_points} ({outliers_pct:.1f}%)")
                    
                    if outliers_found:
                        result_parts.append(f"    OUTLIERS DETECTED:")
                        
                        # Show method-specific results
                        for method, outlier_series in outliers_found.items():
                            result_parts.append(f"      {method} method: {len(outlier_series)} outliers")
                            
                            # Show top outliers with context
                            top_outliers = outlier_series.nlargest(5) if len(outlier_series.nlargest(5)) > 0 else outlier_series
                            bottom_outliers = outlier_series.nsmallest(5) if len(outlier_series.nsmallest(5)) > 0 else outlier_series
                            
                            extreme_outliers = set(top_outliers.index).union(set(bottom_outliers.index))
                            
                            for idx in list(extreme_outliers):  
                                outlier_row = sorted_df.loc[idx]
                                outlier_value = outlier_row[col]
                                
                                result_parts.append(f"        Top outlier values: {outlier_value:.3f}")
                                
                                # Add context from the row
                                for field, value in outlier_row.items():
                                    if field != col:  # Don't repeat the outlier value
                                        result_parts.append(f"          {field}: {value} in {col} column")
                                result_parts.append("")  # Space between outliers
                        
                        # Consensus outliers (found by multiple methods)
                        method_counts = {}
                        for idx in all_outlier_indices:
                            count = sum(1 for method_outliers in outliers_found.values() 
                                       if idx in method_outliers.index)
                            if count > 1:
                                method_counts[idx] = count
                        
                        if method_counts:
                            result_parts.append(f"    CONSENSUS OUTLIERS (detected by multiple methods):")
                            for idx, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                                outlier_row = sorted_df.loc[idx]
                                result_parts.append(f"      Value {outlier_row[col]:.3f} detected by {count} methods in {col} column")
                                
                                # Add timestamp context if available
                                if timestamp_cols:
                                    time_val = outlier_row[timestamp_cols[0]]
                                    result_parts.append(f"        Time: {time_val} in {col} column")
                        
                        # Statistical impact
                        outlier_percentage = outliers_pct
                        result_parts.append(f"    Impact: {outlier_percentage:.1f}% of data points are outliers in {col} column")
                                         
                    else:
                        result_parts.append(f"    NO OUTLIERS DETECTED - Data appears normally distributed in {col} column")
                        await step.stream_token(f"No outliers detected in {col} in {col} column.\n")
                    
                    result_parts.append("")  # Space between columns
                
                result_parts.append("")  # Space between message types
            else:
                await step.stream_token(f"No numeric fields found in {msg_type}.\n")
                result_parts.append(f"No numeric data available in {msg_type} for outlier analysis.")
                result_parts.append("")
        
        await step.stream_token("Outlier detection completed successfully.\n")
        
        step.name = "Outlier detection process is done."
        await step.update()
        return OutlierResult(
            message="Outlier detection completed successfully",
            outliers="\n".join(result_parts)
        )        

@tool
async def detect_events(event_description: str) -> EventResult:
    """
    Detect when specific events first occurred in the flight log data.
    This tool can find the first occurrence of conditions like:
    - GPS achieving 3D fix
    - Signal losses (GPS, RC, etc.)
    - Mode changes
    - Threshold crossings
    - Status changes
    
    Examples:
    - "When did GPS first achieve a 3D fix?"
    - "When did GPS signal first get lost?"
    - "When did GPS yaw become available?"
    - "When was the first instance of RC signal loss?"
    """
    async with cl.Step(name="Starting event detection process", type="tool") as step:
        data = filter_data()
        if not data:
            step.name = "No data available for event detection."
            await step.update()
            return EventResult(
                message="No data available",
                events="No data available. Please extract data first."
            )
        
        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        await step.stream_token(f"Analyzing event: '{event_description}'\n")
        
        # Use AI to interpret the event description and create detection logic
        template = """
        The user wants to detect when this event first occurred: {event_description}
        
        Available data message types and their fields: {col_map}
        
        Analyze the event description and determine the detection logic.
        
        IMPORTANT: Respond with ONLY a valid Python dictionary in this EXACT format:

        - "threshold":
            - Compare a numerical value against a specific threshold value. 
            - Example: {{"message_type": "GPS", "condition_type": "threshold", "field": "NSats", "operator": ">=", "value": 6, "description": "GPS achieves 3D fix"}}
            This detects when the number of satellites becomes 6 or more (typically needed for 3D GPS fix)

        - "loss": 
            - Detect when a signal, connection, or data stream is lost or becomes invalid. 
            - Example: {{"message_type": "GPS", "condition_type": "loss", "field": "Status", "description": "GPS signal lost"}}
            This detects when the "Status" field in the "GPS" message type first becomes invalid.

        - "availability": 
            - Detect when a field becomes available or valid for the first time. This is essentially the opposite of "loss".
            - Example: {{"message_type": "GPS", "condition_type": "availability", "field": "YawDeg", "description": "GPS yaw becomes available"}}
            This detect when YawDeg first has a valid value (non-zero, non-null)

        - "state_change":
            - Detect when a field transitions to a specific state or mode.
            - Example: {{"message_type": "MODE", "condition_type": "state_change", "field": "Mode", "target_state": "AUTO", "description": "Vehicle enters AUTO mode"}}
            This detects when the "Mode" field transitions to "AUTO".

        For the query, respond with ONE dictionary only. Nothing else.
        """
        
        col_map = cl.user_session.get("col_map", {})
        prompt = PromptTemplate(input_variables=["event_description", "col_map"], template=template)
        # Plain-text extraction — use the un-bound model
        chain = prompt | extract_model
        result = chain.invoke({"event_description": event_description, "col_map": col_map})
        await step.stream_token(f"AI parsed event detection config:\n{result.content.strip()}\n")
        
        try:
            response_text = result.content.strip()
            
            import re
            dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if dict_match:
                dict_str = dict_match.group(0)
                detection_config = ast.literal_eval(dict_str)
                await step.stream_token(f"Successfully parsed detection config: {detection_config}\n")
            else:
                raise ValueError("No dictionary found in AI response")
                
        except Exception as e:
            await step.stream_token(f"Error parsing detection config: {str(e)}\n")
            await step.stream_token(f"Raw AI response: '{response_text}'\n")
            return EventResult(
                message="Event detection failed",
                events=f"Error parsing event detection logic: {str(e)}. AI response was: '{response_text[:200]}...'"
            )
        
        # Find the target message type in available data
        target_msg_type = detection_config.get("message_type")
        matching_msg_types = []
        
        for msg_type in data.keys():
            if target_msg_type.lower() in msg_type.lower() or msg_type.lower() in target_msg_type.lower():
                matching_msg_types.append(msg_type)
        
        if not matching_msg_types:
            await step.stream_token(f"No matching message type found for '{target_msg_type}' in available data: {list(data.keys())}\n")
            return EventResult(
                message="Event detection failed",
                events=f"No data available for message type '{target_msg_type}'. Available types: {list(data.keys())}"
            )
        
        results = []
        
        for msg_type in matching_msg_types:
            await step.stream_token(f"Analyzing {msg_type} for event detection...\n")
            df = data[msg_type]
            
            if df.empty:
                continue
                
            # Sort by timestamp if available
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                df = df.sort_values(by=timestamp_cols[0])
                await step.stream_token(f"Sorted data by {timestamp_cols[0]}\n")
            
            condition_type = detection_config.get("condition_type")
            field = detection_config.get("field")
            
            # Check if the target field exists
            matching_fields = [col for col in df.columns if field.lower() in col.lower() or col.lower() in field.lower()]
            if not matching_fields:
                await step.stream_token(f"Field '{field}' not found in {msg_type}. Available fields: {list(df.columns)}\n")
                continue
                
            target_field = matching_fields[0]
            await step.stream_token(f"Using field '{target_field}' for detection\n")
            
            # Apply detection logic based on condition type
            event_detected = False
            first_occurrence = None
            
            if condition_type == "threshold":
                operator = detection_config.get("operator")
                value = detection_config.get("value")
                
                await step.stream_token(f"Checking threshold condition: {target_field} {operator} {value}\n")
                
                if operator == ">=":
                    condition_mask = df[target_field] >= value
                elif operator == "<=":
                    condition_mask = df[target_field] <= value
                elif operator == ">":
                    condition_mask = df[target_field] > value
                elif operator == "<":
                    condition_mask = df[target_field] < value
                elif operator == "==":
                    condition_mask = df[target_field] == value
                elif operator == "!=":
                    condition_mask = df[target_field] != value
                else:
                    await step.stream_token(f"Unknown operator: {operator}\n")
                    continue
                
                # Find first occurrence where condition is True
                first_true_indices = df[condition_mask].index
                if len(first_true_indices) > 0:
                    first_occurrence = df.loc[first_true_indices[0]]
                    event_detected = True
                    
            elif condition_type == "loss":
                message_type = detection_config.get("message_type", "")
                await step.stream_token(f"Checking loss condition for: {message_type} and {target_field}\n")
                
                # Detect when target_field first becomes None/"None"/"" (after strip)
                series = df[target_field]
                condition_mask = series.isna()
                if series.dtype == 'object':
                    s_str = series.astype(str).str.strip()
                    condition_mask |= (s_str == "") | (s_str.str.lower() == "none")
                
                first_true_indices = df[condition_mask].index
                if len(first_true_indices) > 0:
                    first_occurrence = df.loc[first_true_indices[0]]
                    event_detected = True
                    
            elif condition_type == "availability":
                await step.stream_token(f"Checking availability condition for field: {target_field}\n")
                
                # Check when field becomes available (non-null, non-zero, valid)
                condition_mask = (df[target_field].notna()) & (df[target_field] != 0)
                
                # For string fields, check for non-empty values
                if df[target_field].dtype == 'object':
                    condition_mask = (df[target_field].notna()) & (df[target_field].astype(str).str.strip() != '') & (df[target_field].astype(str) != '0')
                
                first_true_indices = df[condition_mask].index
                if len(first_true_indices) > 0:
                    first_occurrence = df.loc[first_true_indices[0]]
                    event_detected = True
                    
            elif condition_type == "state_change":
                target_state = detection_config.get("target_state")
                await step.stream_token(f"Checking state change to: {target_state}\n")
                
                # Find first occurrence of target state
                if df[target_field].dtype == 'object':
                    condition_mask = df[target_field].astype(str).str.contains(str(target_state), case=False, na=False)
                else:
                    condition_mask = df[target_field] == target_state
                
                first_true_indices = df[condition_mask].index
                if len(first_true_indices) > 0:
                    first_occurrence = df.loc[first_true_indices[0]]
                    event_detected = True
            
            # Record results
            if event_detected and first_occurrence is not None:
                await step.stream_token(f"Event detected in {msg_type} at index {first_occurrence.name}\n")
                
                result_entry = {
                    "message_type": msg_type,
                    "event_description": detection_config.get("description", event_description),
                    "field_checked": target_field,
                    "event_value": first_occurrence[target_field],
                    "full_context": dict(first_occurrence)
                }
                
                # Add timestamp if available
                if timestamp_cols:
                    result_entry["timestamp"] = first_occurrence[timestamp_cols[0]]
                    await step.stream_token(f"Event occurred at timestamp: {first_occurrence[timestamp_cols[0]]}\n")
                
                results.append(result_entry)
            else:
                await step.stream_token(f"Event not detected in {msg_type}\n")
        
        # Format results
        if results:
            await step.stream_token(f"Event detection completed! Found {len(results)} occurrence(s).\n")
            
            result_parts = [f"Event Detection Results for: '{event_description}'"]
            result_parts.append("=" * 60)
            
            for i, result in enumerate(results, 1):
                result_parts.append(f"\nOccurrence #{i}:")
                result_parts.append(f"  Message Type: {result['message_type']}")
                result_parts.append(f"  Description: {result['event_description']}")
                result_parts.append(f"  Field: {result['field_checked']}")
                result_parts.append(f"  Value: {result['event_value']}")
                
                if "timestamp" in result:
                    result_parts.append(f"  Timestamp: {result['timestamp']}")
                
                result_parts.append(f"  Full Context:")
                for field, value in result['full_context'].items():
                    result_parts.append(f"    {field}: {value}")
            
            # Update step name to show completion
            step.name = "Event detection process is done."
            await step.update()
            return EventResult(
                message="Event detection completed successfully",
                events="\n".join(result_parts),
                occurrences=[EventOccurrence(**result) for result in results]
            )
        else:
            await step.stream_token("No events detected matching the specified criteria.\n")
            # Update step name to show completion
            step.name = "Event detection process is done."
            await step.update()
            return EventResult(
                message="No events detected",
                events=f"No events found matching: '{event_description}'. The condition may not have occurred in the available data, or the detection criteria may need adjustment."
            )

@tool
async def visualize(query: str) -> VisualizationResult:
    """
    Visualize the data.
    """
    async with cl.Step(name="Starting visualization process.", type="tool") as step:
        try:
            data = filter_data()
        except Exception as e:
            await step.stream_token(f"Error filtering data: {str(e)}\n")
            step.name = "Visualization process is failed."
            await step.update()
            return VisualizationResult(
                message="Error filtering data",
                code_generated=False
            )
        if not data:
            step.name = "No data is available for visualization."
            await step.update()
            return VisualizationResult(
                message="No data available",
                code_generated=False
            )
        
        await step.stream_token(f"Found {len(data)} message types in the extracted data.\n")
        await step.stream_token("Preparing data for visualization by sampling records...\n")
        
        # The data has multiple dataframes, so we need to sample from each one.
        keys = []
        for key in data:
            keys.append(key)

        plt.rcParams.update({'figure.dpi': 150,})                

        await step.stream_token("Generating visualization code using AI model...\n")

        template = """
        Pick a key from the available keys: {keys} that is the most relevant
        for the user last query {query} and chat history: {chat_history}.
        Only give the key, nothing else. 

        If you cannot be able to find a key in {keys} that is relevant to the user query
        or if {keys} is empty, return "NO_KEY".
        """

        prompt = PromptTemplate(input_variables=["query", "keys", "chat_history"], template=template)
        # Plain-text extraction — use the un-bound model
        chain = prompt | extract_model

        # Get the last Human Message from chat history
        chat_history = cl.user_session.get("message_history")

        result = chain.invoke({"query": query, "keys": keys, "chat_history": chat_history})
        key = result.content.strip()

        if key == "NO_KEY":
            await step.stream_token("No data is relevant to the user query or the chat history.\n")
            step.name = "Visualization process is failed."
            await step.update()
            cl.user_session.set("code", "")
            cl.user_session.set("data", {})
            return VisualizationResult(
                message="No data is relevant to the user query or the chat history.",
                code_generated=False
            )

        await step.stream_token(f"Found the key: {key}\n")
            
        sample_size = min(100, len(data[key]))
        sampled_data = data[key].sample(sample_size, replace=True)
        await step.stream_token(f"Sampled {sample_size} rows from {key} dataset.\n")
        
        template = """
        I have a dataframe in a dictionary called `data` that can be accessed using data[key].
        Here is the available key: {key}.

        Here is how the 100 rows sampled from the data looks like: {sampled_data}.

        Write a matplotlib function in Python to visualize the data[key] so that the code can be executed properly to get the right plot.
        Make sure that the code is runnable directly and it generates the plot that is asked in the user query: {query}.
        
        Only give the code, nothing else. Don't include ```python or ``` or anything else. 
        Don't explain the data.

        IMPORTANT NOTES 
        - Make the plot look nice, readable, and high quality. 
        - Don't use any other libraries than matplotlib, numpy, pandas, datetime, and other standard libraries
        that are already installed in the system.
        - Make sure that the code you generate can be run with 1 click without needing any modification/change.
        - `data` already exists, don't create a new one!
        - Don't remove a file/folder, don't create a new one, or do anything else that might affect the existing files/folders.
        - Don't install a new library or uninstall the existing ones.
        """
        
        col_map = cl.user_session.get("col_map", {})
        prompt = PromptTemplate(input_variables=["query", "sampled_data", "key"], template=template)
        # Plain-text code generation — use the un-bound model
        chain = prompt | extract_model
        result = chain.invoke({"query": query, "sampled_data": sampled_data, "key": key})
        code = result.content.strip()
        code = code.replace("plt.show()", "")  
        
        await step.stream_token("AI model successfully generated visualization code.\n")
        
        # Execute the visualization code
        cl.user_session.set("code", code)

        await step.stream_token(f"Visualization ready using data from message type: {key}\n")

        # Update step name to show completion
        step.name = "Visualization process is done."
        await step.update()
        return VisualizationResult(
            message="Successfully generated the visualization code",
            code_generated=True
        )