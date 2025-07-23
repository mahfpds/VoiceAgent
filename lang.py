"""
lang.py  -  LLM logic + async SQL history
----------------------------------------
pip install "langchain-community>=0.2.17" aiosqlite sqlalchemy
"""

from __future__ import annotations

import os, time
from typing import AsyncIterator, Any

from sqlalchemy.ext.asyncio import create_async_engine

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import SQLChatMessageHistory


# ────────── DB CONFIG ────────── #
DB_FILE = os.getenv("CONV_DB_FILE", "conversation_history.db")
ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_FILE}"
async_engine = create_async_engine(ASYNC_DB_URL, echo=False)
TABLE_NAME = "message_store"
# ─────────────────────────────── #

# ────────── PROMPT TEMPLATE ────────── #
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
        "### Language rule\n"
        "You MUST write every reply **exclusively in {response_language}**.\n\n"
        "### Persona\n"
        "You are Jane, a polite, concise scheduling assistant (< 20 words per reply). "
        "You repeat the caller's request to confirm understanding.\n\n"
        "### Domain knowledge\n"
        "- Dr. James is an eye doctor with 20 years' experience.\n"
        "- Clinic hours: San Francisco 10-14 Mon-Fri; New York 14-20 Mon-Fri.\n"
        "- If the user requests outside these hours, politely refuse.\n"
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]).partial(response_language="English")

# ────────── LLM & CHAIN ────────── #
llm = OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434")
chain = prompt_template | llm


# ────────── HISTORY HELPERS ────────── #
def _get_session_history(session_id: str) -> SQLChatMessageHistory:
    """Return (and lazily create) an async SQL-backed history."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,   # ← correct kw for LC‑0.3.27
        table_name=TABLE_NAME,
    )


def remove_last_assistant_message(session_id: str) -> None:
    """Erase the newest AI turn when the caller barges-in."""
    hist = _get_session_history(session_id)
    msgs = list(hist.messages)
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if getattr(m, "type", None) == "ai" or isinstance(m, AIMessage):
            del msgs[i]
            hist.clear()
            hist.add_messages(msgs)
            break


def _ensure_history(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs.setdefault("history", [])
    return kwargs


# ────────── WRAP CHAIN WITH HISTORY ────────── #
chain_with_history = RunnableWithMessageHistory(
    chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ────────── PUBLIC API ────────── #
def get_llm_response(text: str, session_id: str | None = None, **kw) -> str:
    """
    Synchronous helper.
    • If session_id is provided → persist to DB.
    • If None (warm-up ping)   → call plain chain (no history) to avoid
      sync/async mix-ups.
    """
    payload = _ensure_history({"input": text, **kw})
    if session_id:
        return chain_with_history.invoke(
            payload, config={"configurable": {"session_id": session_id}}
        )
    else:
        return chain.invoke(payload)           # ← no history for warm‑up


async def llm_stream(text: str, session_id: str | None = None, **kw) -> AsyncIterator[str]:
    """Async generator yielding streamed tokens while persisting history."""
    payload = _ensure_history({"input": text, **kw})
    async for chunk in chain_with_history.astream(
        payload, config={"configurable": {"session_id": session_id}}
    ):
        yield chunk

