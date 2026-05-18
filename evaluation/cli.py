import os
import sys
from pathlib import Path

CHATBOT_DIR = Path(__file__).parent
sys.path.insert(0, str(CHATBOT_DIR))
os.chdir(CHATBOT_DIR)

import asyncio
import chainlit as cl
from langchain_core.messages import HumanMessage

class _NoopStep:
    def __init__(self, *a, **kw):
        self.name = kw.get("name", "")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def stream_token(self, *a, **kw):
        pass

    async def update(self):
        pass

    async def remove(self):
        pass


class _FakeSession(dict):
    def get(self, k, default=None):
        return super().get(k, default)

    def set(self, k, v):
        self[k] = v


cl.Step = _NoopStep
cl.user_session = _FakeSession()

from core.graph import graph

async def run_question(question: str) -> str:
    cl.user_session.set("data", {})
    cl.user_session.set("col_map", {})
    cl.user_session.set("msg_context", "")
    cl.user_session.set("file_id", "")
    cl.user_session.set("web_content", "")
    cl.user_session.set("message_history", [])

    config = {"configurable": {"thread_id": f"cli-{id(question)}"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    final = result["messages"][-1]
    return final.content if hasattr(final, "content") else str(final)


async def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Usage: python chatbot/cli.py <your question>")
        sys.exit(1)
    answer = await run_question(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
