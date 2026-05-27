import contextvars
import os
import sys
import uuid
from pathlib import Path
import asyncio
import chainlit as cl
from langchain_core.messages import HumanMessage

CHATBOT_DIR = Path(__file__).parent
sys.path.insert(0, str(CHATBOT_DIR))
os.chdir(CHATBOT_DIR)


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


_session_var: contextvars.ContextVar = contextvars.ContextVar("cl_user_session", default=None)


class _ContextualSession:
    def _store(self) -> dict:
        store = _session_var.get()
        if store is None:
            store = {}
            _session_var.set(store)
        return store

    def get(self, k, default=None):
        return self._store().get(k, default)

    def set(self, k, v):
        self._store()[k] = v


cl.Step = _NoopStep
cl.user_session = _ContextualSession()

# Import the graph AFTER stubbing cl.Step / cl.user_session so any
# module-level references in core.* bind to our stubs.
from core.graph import graph


async def run_question(question: str) -> str:
    _session_var.set({
        "data": {},
        "col_map": {},
        "msg_context": "",
        "file_id": "",
        "web_content": "",
        "message_history": [],
    })

    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4()}"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    final = result["messages"][-1]
    content = final.content if hasattr(final, "content") else str(final)
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return content


async def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit(1)
    answer = await run_question(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
