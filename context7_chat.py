import json
import asyncio
import streamlit as st
from typing import List, Any, cast
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from context7_agent import run_agent_query, RagResponse, LLM_MODEL, EMBEDDING_MODEL, DATABASE_URL

st.set_page_config(page_title="Context7 RAG Chat", layout="centered")
st.title("🦾 Context7 RAG Agent")
st.info("Ask any question about your docs. Conversation context is preserved.")

# --- Helpers ----------------------------------------------------------------

def ensure_model_message(msg):
    if isinstance(msg, (ModelRequest, ModelResponse)):
        return msg
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        if msg["role"] == "user":
            return ModelRequest(parts=[UserPromptPart(content=msg["content"])], kind="request")
        elif msg["role"] == "assistant":
            return ModelResponse(parts=[TextPart(content=msg["content"])], kind="response")
    raise ValueError(f"Invalid message format: {msg}")


def render_history(history: List[Any]):
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and part.content.strip():
                    st.chat_message("user").write(part.content)
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart) and part.content.strip():
                    st.chat_message("assistant").write(part.content)


def to_serialisable(obj: Any) -> Any:
    """Return a JSON‑serialisable representation for arbitrary objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return {k: to_serialisable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [to_serialisable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    return str(obj)

# --- Session state -----------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = cast(List[Any], [])

st.session_state.chat_history = [
    ensure_model_message(m) for m in st.session_state.chat_history
]

# --- Render existing conversation -------------------------------------------
render_history(st.session_state.chat_history)

# --- User input handling -----------------------------------------------------
if user_query := st.chat_input("Type your question and press Enter…"):
    user_msg = ModelRequest(parts=[UserPromptPart(content=user_query)], kind="request")
    st.session_state.chat_history.append(user_msg)

    # Show it immediately so the user sees it while the agent thinks
    st.chat_message("user").write(user_query)

    with st.spinner("Agent thinking…"):
        async def call_agent(query, message_history):
            return await run_agent_query(query, message_history=message_history)
        agent_result = asyncio.run(call_agent(user_query, st.session_state.chat_history))

    if (
        agent_result
        and hasattr(agent_result, "output")
        and isinstance(agent_result.output, RagResponse)
    ):
        answer: RagResponse = agent_result.output
        reply_text = answer.answer
    else:
        reply_text = "[Error: No agent reply.]"

    assistant_msg = ModelResponse(parts=[TextPart(content=reply_text)], kind="response")
    st.session_state.chat_history.append(assistant_msg)

    st.rerun()

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.subheader("🗑️ Session")
    if st.button("🔄  New conversation"):
        st.session_state.chat_history.clear()
        st.rerun()

    st.divider()
    st.subheader("🛠️  Dev tools")

    if st.checkbox("Show raw conversation history"):
        st.json([to_serialisable(m) for m in st.session_state.chat_history])

    if st.button("💾  Download chat as JSON"):
        json_bytes = json.dumps(
            [to_serialisable(m) for m in st.session_state.chat_history],
            indent=2,
            default=to_serialisable
        ).encode()
        st.download_button(
            label="Download",
            data=json_bytes,
            file_name="context7_chat_history.json",
            mime="application/json"
        )

    with st.expander("Environment check", expanded=False):
        st.markdown(f"- **LLM model:** `{LLM_MODEL}`")
        st.markdown(f"- **Embedding model:** `{EMBEDDING_MODEL}`")
        st.markdown(f"- **Database URL set:** `{bool(DATABASE_URL)}`")
