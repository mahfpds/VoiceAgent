
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.messages import AIMessage  # NEW

# ───────── prompt template ───────── #
prompt_template = ChatPromptTemplate.from_messages([
    # HARD constraint: respond in a single specified language
    ("system", (
        "### Language rule\n"
        "You MUST write every reply **exclusively in {response_language}**. "
        "Do NOT include any other language.\n\n"
        "### Persona\n"
        "You are Jane, a polite, concise scheduling assistant (< 20 words per utterance). "
        "You repeat the caller's request to confirm understanding.\n\n"
        "### Domain knowledge\n"
        "- Dr. James is an eye doctor with 20 years' experience.\n"
        "- Clinic hours: San Francisco 10-14 Mon-Fri; New York 14-20 Mon-Fri.\n"
        "- If user requests outside hours, politely refuse.\n"
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]).partial(response_language="English")

llm = OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434")
# llm = OllamaLLM(model='gemma3n:e4b', base_url="http://localhost:11434")

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
_chat_sessions: dict[str, InMemoryChatMessageHistory] = {}

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = InMemoryChatMessageHistory()
    return _chat_sessions[session_id]


# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def remove_last_assistant_message(session_id: str, **kwargs) -> None:
    """
    Remove the most recent assistant/AI message from the session history.
    Used when the user barges-in and interrupts the bot mid-utterance.
    """
    history = _chat_sessions.get(session_id)
    if not history or not history.messages:
        return
    for idx in range(len(history.messages) - 1, -1, -1):
        msg = history.messages[idx]
        # Compatible with LangChain v0.2+ (AIMessage.type == "ai")
        is_ai = (
            getattr(msg, "type", None) == "ai"
            or isinstance(msg, AIMessage)
        )
        if is_ai:
            del history.messages[idx]
            break

def _ensure_history(kwargs: dict) -> dict:
    """Guarantee a 'history' key for the first (warm‑up) call."""
    if "history" not in kwargs:
        kwargs["history"] = []
    return kwargs

# replace the two helper functions with the versions below
def get_llm_response(text: str, session_id: str | None = None, **kw):
    """Synchronous one‑shot call."""
    payload = _ensure_history({"input": text, **kw})
    return chain.invoke(payload, config={"configurable": {"session_id": session_id}})

async def llm_stream(text: str, session_id: str | None = None, **kw):
    """Async generator of streamed tokens."""
    payload = _ensure_history({"input": text, **kw})
    async for chunk in chain.astream(payload, config={"configurable": {"session_id": session_id}}):
        yield chunk