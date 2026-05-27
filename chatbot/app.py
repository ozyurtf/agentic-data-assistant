# app.py — Chainlit entry point. Run with: chainlit run app.py
# All graph and tool logic lives in core/. This file is just the adapter
# that wires Chainlit's UI handlers to the compiled graph.

import asyncio
import os
from typing import Dict

import chainlit as cl
import jwt
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi import Request, Response
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

try:
    from agents import set_trace_processors
    from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
    set_trace_processors([OpenAIAgentsTracingProcessor()])
except ImportError:
    pass

from core.graph import graph

matplotlib.use("Agg")
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-in-prod-please")
JWT_ALG = "HS256"
COOKIE_NAME = "auth_token"


def extract_jwt(headers: Dict[str, str]) -> str:
    headers_ci = {k.lower(): v for k, v in headers.items()}
    auth = headers_ci.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1]
    cookie_header = headers_ci.get("cookie", "")
    for piece in cookie_header.split(";"):
        name, _, value = piece.strip().partition("=")
        if name == COOKIE_NAME:
            return value
    return ""


@cl.header_auth_callback
def auth_callback(headers: Dict[str, str]):
    token = extract_jwt(headers)
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return cl.User(
        identifier=user_id,
        metadata={"provider": "jwt"},
    )


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

    final_content = ""
    async for message, metadata in graph.astream(
        {"messages": message_history},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        # Stream the final answer from the agent node (last call, no tool_calls)
        if (
            metadata.get("langgraph_node") == "agent"
            and message.content
            and not getattr(message, "tool_calls", None)
            and not getattr(message, "tool_call_chunks", None)
        ):
            content = message.content
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            final_content += content

    if final_content:
        for char in final_content:
            await final_answer.stream_token(char)
            await asyncio.sleep(0.01)

    await final_answer.send()

    message_history.append(AIMessage(content=final_answer.content))
    cl.user_session.set("message_history", message_history)

    # Handle visualization if code was generated
    code = cl.user_session.get("code")
    if code:
        try:
            data = cl.user_session.get("data")
            exec(code)
            fig = plt.gcf()
            fig.set_dpi(300)
            elements = [
                cl.Pyplot(name="plot", figure=fig, display="inline"),
            ]
            await cl.Message(
                content="Here is your visualization:",
                elements=elements,
            ).send()
            plt.close(fig)
            cl.user_session.set("code", "")
        except Exception as e:
            error_message = f"Error: {e}"
            for char in error_message:
                await final_answer.stream_token(char)
                await asyncio.sleep(0.01)


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_logout
def main(request: Request, response: Response):
    print("The user logged out!")
    response.delete_cookie("my_cookie")
