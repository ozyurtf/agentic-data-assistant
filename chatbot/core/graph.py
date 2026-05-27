from typing import Literal
import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from core.tools import (
    base_model,
    load_web_content,
    extract_data,
    maximum,
    minimum,
    average,
    total_sum,
    visualize,
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
    detect_events,
]

model = base_model.bind_tools(tools)

tool_node = ToolNode(tools=tools)

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

STATIC_SYSTEM_PROMPT = """\
You are an assistant that analyzes flight log data. You have access to several tools.

AVAILABLE TOOLS:
- `load_web_content`: Use this tool to load the web content if you find URL in the user query and if you think it should be extracted.
- `extract_data`: Use this tool to extract the data that is relevant to the user's query.
- `maximum`: Use this tool to find the maximum/highest value of a field.
- `minimum`: Use this tool to find the minimum/lowest value of a field.
- `average`: Use this tool to find the average value of a field.
- `total_sum`: Use this tool to find the total sum of a field.
- `visualize`: Use this tool to visualize the data.
- `detect_events`: Use this tool when users ask about WHEN specific events occurred.

Call `extract_data` tool FIRST when users ask about:
- Anomalies/issues in the data
- Maximum/minimum/average/sum values
- Visualizations
- Detecting events
- And any analysis/technical questions about the uploaded log data

IMPORTANT RULES:
- CRITICAL: Call `extract_data` tool ONLY ONCE per user query. Extract ALL related fields in a SINGLE call.
- For queries asking about multiple related values (e.g., "roll and pitch", "latitude and longitude", "GPS and altitude"),
  extract ALL related fields in ONE call rather than making separate calls for each field.
- For event timing questions, use `detect_events` after extracting relevant data with `extract_data`.
- You don't have to call any of these tools all the time. Sometimes the user might
ask a follow up question or ask about something that can be answered from the `chat_history`.
In those cases, don't call the tools that would normally be called.
and just return the answer from the `chat_history`.
- If `col_map` is empty or if what users asks for in their query is not available in the data in the `col_map`
and if you think it might be better to use another data or another fields/columns in the `msg_context`,
call the `extract_data` tool to get the right data.

INSTRUCTIONS FOR GENERATING FINAL ANSWER:
- BE CONCISE. 1-3 sentences for most questions. Lead with the value, then context if needed.
- Verify numerical results make sense for the query before responding.
- No Python code in the answer.
- No emojis or icons.
- Use plain ASCII for math (e.g., a/b, *, sqrt, change in X) - never LaTeX.
- Return well-formatted, readable plain text.

EXAMPLES OF SINGLE CALLS:
- "What is the average roll and pitch values?" -> Extract BOTH roll AND pitch in ONE call
- "Show me GPS latitude and longitude" -> Extract BOTH latitude AND longitude in ONE call
- "What are the maximum altitude and speed?" -> Extract BOTH altitude AND speed in ONE call
"""


async def call_model(state: MessagesState):
    messages = state["messages"]

    dynamic_context = (
        f"You can see the `col_map`: {cl.user_session.get('col_map', {})}, "
        f"the `chat_history`: {cl.user_session.get('message_history', [])}, "
        f"and `msg_context`: {cl.user_session.get('msg_context', {})}"
    )

    system_message = SystemMessage(content=[
        {
            "type": "text",
            "text": STATIC_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_context,
        },
    ])

    messages_with_system = [system_message] + messages
    response = await model.ainvoke(messages_with_system)
    return {"messages": [response]}

builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile()