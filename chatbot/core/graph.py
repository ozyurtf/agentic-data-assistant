from typing import Literal
import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from core.tools import (
    _base_model,
    qa_model,
    load_web_content,
    extract_data,
    maximum,
    minimum,
    average,
    total_sum,
    visualize,
    detect_sudden_changes,
    detect_oscillations,
    detect_outliers,
    detect_events,
)

tools = [
    load_web_content,
    extract_data,
    maximum,
    minimum,
    average,
    total_sum,
    visualize,
    detect_sudden_changes,
    detect_oscillations,
    detect_outliers,
    detect_events,
]

model = _base_model.bind_tools(tools)

tool_node = ToolNode(tools=tools)

def should_continue(state: MessagesState) -> Literal["tools", "qa"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "qa"

async def call_model(state: MessagesState):
    messages = state["messages"]

    system_message = SystemMessage(content=f"""
    You are an assistant that analyzes flight log data. You have access to several tools.

    AVAILABLE TOOLS:
    - `load_web_content`: Use this tool to load the web content if you find URL in the user query and if you think it should be extracted.
    - `extract_data`: Use this tool to extract the data that is relevant to the user's query.
    - `maximum`: Use this tool to find the maximum/highest value of a field.
    - `minimum`: Use this tool to find the minimum/lowest value of a field.
    - `average`: Use this tool to find the average value of a field.
    - `total_sum`: Use this tool to find the total sum of a field.
    - `visualize`: Use this tool to visualize the data.
    - `detect_sudden_changes`: Use this tool to detect sudden changes in the data.
    - `detect_oscillations`: Use this tool to detect oscillations in the data.
    - `detect_outliers`: Use this tool to detect outliers in the data.
    - `detect_events`: Use this tool when users ask about WHEN specific events occurred.

    Call `extract_data` tool FIRST when users ask about:
    - Anomalies/issues in the data
    - Maximum/minimum/average/sum values
    - Visualizations
    - Any analysis questions about the log data
    - Detecting events

    IMPORTANT RULES:
    - CRITICAL: Call `extract_data` tool ONLY ONCE per user query. Extract ALL related fields in a SINGLE call.
    - For queries asking about multiple related values (e.g., "roll and pitch", "latitude and longitude", "GPS and altitude"),
      extract ALL related fields in ONE call rather than making separate calls for each field.

    - When they ask about anomalies, you can use the `detect_sudden_changes`, `detect_oscillations`, and `detect_outliers` tools
    to find the sudden changes, oscillations, and outliers in the data and then interpret whether they are anomalies or not.

    - For event timing questions, use `detect_events` after extracting relevant data with `extract_data`.

    - You don't have to call any of these tools all the time. Sometimes the user might
    ask a follow up question or ask about something that can be answered from the `chat_history`.
    In those cases, don't call the tools that would normally be called.
    and just return the answer from the `chat_history`.

    - If the user asks for issues, you can extract the error (e.g., ERR) data if the data is
    part of the `msg_context` and analyze it.

    - If `col_map` is empty or if what users asks for in their query is not available in the data in the `col_map`
    and if you think it might be better to use another data or another fields/columns in the `msg_context`,
    call the `extract_data` tool to get the right data.

    EXAMPLES OF SINGLE CALLS:
    - "What is the average roll and pitch values?" → Extract BOTH roll AND pitch in ONE call
    - "Show me GPS latitude and longitude" → Extract BOTH latitude AND longitude in ONE call
    - "What are the maximum altitude and speed?" → Extract BOTH altitude AND speed in ONE call
    """ +
    f"You can see the `col_map`: {cl.user_session.get('col_map', {})}, " +
    f"the `chat_history`: {cl.user_session.get('message_history', [])}, " +
    f"and `msg_context`: {cl.user_session.get('msg_context', {})}")

    messages_with_system = [system_message] + messages
    response = await model.ainvoke(messages_with_system)
    return {"messages": [response]}


async def quality_assurance_agent(state: MessagesState):
    """
    Quality Assurance Agent that validates and improves the final answer before
    presenting to user.
    """
    async with cl.Step(name="Response is being evaluated", type="run") as step:
        messages = state["messages"]
        final_answer = messages[-1]

        # Get context from the conversation
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        # Get additional context from session
        col_map = cl.user_session.get("col_map", {})
        data_available = bool(cl.user_session.get("data", {}))

        await step.stream_token("Analyzing response quality...\n")

        qa_system_prompt = f"""
        You are a Quality Assurance Agent for flight log data analysis responses.
        Your role is to look from flight perspective and validate/approve/improve
        the final answer before it reaches the user based on your knowledge of flight data in general
        and the user's query.

        CONTEXT:
        - Original User Query: {user_query}

        TASK: Review the answer below and either return it as-is if it's good, or provide an improved version.

        VALIDATION CRITERIA:
        1. **Technical Accuracy**: Does the answer correctly interpret flight log data?
        2. **Completeness**: Does it fully address the user's question?
        3. **Clarity**: Is the explanation clear and easy to understand?
        4. **Context**: Does it make sense given the available data and query?
        5. **Formatting**: Is it well-structured and readable?

        VALIDATION RULES:
        - For event timing: Verify timestamps and event logic make sense
        - For anomaly detection: Check if interpretations are reasonable for flight data and
        if numerical analysis results can be interpreted as anomalies.
        - For visualizations: Confirm descriptions match typical flight patterns

        INSTRUCTIONS:
        Check the numerical analysis results, make sure they make sense and reinterpret them
        based on the user query if needed.
        If the answer is good as-is, return it exactly as provided.
        If improvements are needed, provide a better version that addresses any issues.
        DO NOT return JSON - return the actual answer content that should be shown to the user.
        DO NOT include your evaluation of whether the final answer was good or not. Just return the improved answer.
        Return ONLY ONE VERSION of the answer - either the original if it's good, or an improved version if needed.
        NEVER include both the original and improved version in your response.
        DO NOT include the original answer in the improved answer if you are improving it.
        DO NOT include any Python code for visualization in the answer.
        DO NOT include any emojis or icons in the final answer.

        COMMON ISSUES TO FIX:
        - Add missing context explanations
        - Improve formatting and structure
        - Correct technical inaccuracies
        - Make incomplete answers more complete
        - Fix unclear or confusing explanations
        - Convert LaTeX math notation to readable plain text
        - Replace mathematical symbols with clear descriptions
        - Format formulas in a user-friendly way without LaTeX syntax

        FORMATTING RULES:
        - Replace LaTeX sin, cos, sqrt functions with plain text versions
        - Convert fraction notation to (a/b) or "a divided by b"
        - Replace multiplication dots with * or ×
        - Convert Delta symbols to "change in" or "Δ"
        - Replace complex LaTeX with clear step-by-step explanations
        - Use simple ASCII characters instead of LaTeX symbols
        - Remove square brackets around formulas
        - Convert subscripts and superscripts to readable format
        - Return the answer in a organized, clean, readable and nice format.
        """

        await step.stream_token("Running quality assurance analysis...\n")

        qa_response = await qa_model.ainvoke([
            SystemMessage(qa_system_prompt),
            HumanMessage(f"Please validate this answer:\n\n{final_answer.content}")
        ])

        qa_content = qa_response.content.strip()

        await step.stream_token("Quality assurance check completed.\n")

        step.name = "Response is checked."
        await step.update()

        improved_response = AIMessage(content=qa_content)
        return {"messages": [improved_response]}


builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("qa", quality_assurance_agent)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
builder.add_edge("qa", END)

graph = builder.compile()
