from __future__ import annotations

import asyncio
import os
import sys
import json
import pydantic_core
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, AsyncGenerator, Optional

import asyncpg
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from prompts import RAG_AGENT_SYSTEM_PROMPT

# --- Environment Loading ---
# Load environment variables from .env file in the current directory
# Use absolute path to ensure it works regardless of execution directory
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)
print(f"Attempting to load .env from: {dotenv_path}")
if not os.path.exists(dotenv_path):
    print(f"Warning: .env file not found at {dotenv_path}")

# --- Configuration & Validation ---
# Specific keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-3.7-sonnet") # Default to sonnet via OpenRouter
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Specifically for embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") # OpenAI embedding model
DATABASE_URL = os.getenv("DATABASE_URL")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def validate_env_vars():
    """Validate required environment variables for OpenRouter LLM and OpenAI Embeddings."""
    errors = []
    # OpenRouter (for LLM)
    if not OPENROUTER_API_KEY or "YOUR_" in OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is missing or invalid in .env")
    if not LLM_MODEL:
        errors.append("LLM_MODEL is missing in .env (e.g., 'anthropic/claude-3-haiku-20240307')")

    # OpenAI (for Embeddings)
    if not OPENAI_API_KEY or "YOUR_" in OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is missing or invalid in .env (required for embeddings)")
    if not EMBEDDING_MODEL:
        errors.append("EMBEDDING_MODEL is missing in .env (e.g., 'text-embedding-3-small')")

    # Database
    if not DATABASE_URL:
        errors.append("DATABASE_URL is missing in .env")
    elif not DATABASE_URL.startswith(("postgresql://", "postgres://")):
         errors.append("DATABASE_URL does not look like a valid PostgreSQL connection string.")

    if errors:
        for error in errors:
            print(f"Configuration Error: {error}")
        print("Please ensure the .env file is correctly configured in the same directory as the script.")
        sys.exit(1) # Exit if configuration is invalid
    else:
        print("Environment variables validated successfully.")

validate_env_vars()

# --- Database Connection ---
@asynccontextmanager
async def database_connect() -> AsyncGenerator[asyncpg.Pool, None]: # Keep None possibility for initial connection failure
    """Connect to the database using the DATABASE_URL. Re-raises exceptions from the yielded block."""
    pool = None
    if not DATABASE_URL:
        print("Database URL not configured. Skipping database connection.")
        yield None # Yield None if no URL
        return # Exit cleanly

    try:
        print(f"Connecting to database...")
        pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1, max_size=10,
            command_timeout=60, timeout=300
        )
        print("Database connection established.")
        # Yield the pool. If an exception happens inside the 'async with' block,
        # it will propagate here.
        yield pool
    except (asyncpg.exceptions.InvalidCatalogNameError, OSError, ConnectionRefusedError, asyncio.TimeoutError) as e:
        # Handle specific connection errors
        print(f"Error connecting to database: {e}")
        print("Please ensure the database is running and the DATABASE_URL is correct.")
        raise
    finally:
        # This cleanup will run even if an exception occurred during yield.
        if pool:
            try:
                await pool.close()
                print("Database connection pool closed.")
            except Exception as close_exc:
                # Log error during close, but don't mask the original exception
                print(f"Error closing database pool: {close_exc}")

# --- Pydantic Models ---
class RagResponse(BaseModel):
    """Defines the structured output for the RAG agent."""
    answer: str = Field(description="The final answer to the user's query, synthesized from context.")
    retrieved_snippets: List[Dict[str, Any]] = Field(description="List of documentation snippets used to generate the answer.", default=[])
    confidence_score: Optional[float] = Field(description="A score between 0.0 and 1.0 indicating the agent's confidence in the answer.", default=None)

# --- Agent Dependencies ---
@dataclass
class AgentDependencies:
    """Dependencies for the Context7 RAG Agent"""
    # Renamed for clarity: this client is specifically for embeddings
    openai_embedding_client: AsyncOpenAI
    db_pool: Optional[asyncpg.Pool] # Allow Pool to be None

# --- Client and Model Initialization (Archon-Inspired) ---

# 1. Initialize OpenAI Client (for Embeddings)
try:
    openai_embedding_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    print("OpenAI client for embeddings initialized successfully.")
except Exception as e:
    print(f"Fatal Error: Could not initialize OpenAI client for embeddings: {e}")
    sys.exit(1)

# 2. Initialize Chat Model (via OpenRouter)
# Uses Pydantic-AI's OpenAIModel but configure its provider to point to OpenRouter.
try:
    # Create a dedicated AsyncOpenAI client configured for OpenRouter
    openrouter_llm_client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    # Configure the provider with this client
    openrouter_provider = OpenAIProvider(openai_client=openrouter_llm_client)
    # Create the Pydantic-AI model instance using the specified LLM_MODEL name
    chat_model_instance = OpenAIModel(
        model_name=LLM_MODEL,
        provider=openrouter_provider,
    )
    print(f"Chat model '{LLM_MODEL}' via OpenRouter initialized successfully.")
except Exception as e:
    print(f"Fatal Error: Could not initialize Chat Model via OpenRouter: {e}")
    sys.exit(1)


# --- Agent Definition ---
# Initialize the agent using the configured chat model instance
try:
    context7_agent = Agent(
        model=chat_model_instance, # Use the pre-configured model instance
        deps_type=AgentDependencies,
        output_type=RagResponse,
        system_prompt=RAG_AGENT_SYSTEM_PROMPT,
        model_settings={
            "temperature": 0.1,
            "max_tokens": 2500
        }
    )
    print("Context7 RAG Agent initialized successfully.")
except Exception as e:
     print(f"Fatal Error: Could not initialize Pydantic-AI Agent: {e}")
     sys.exit(1)


# --- Agent Tools ---
@context7_agent.tool
async def retrieve_context7_docs(ctx: RunContext[AgentDependencies], query: str, top_k: int = 5) -> str:
    """
    Retrieves relevant documentation snippets from the Context7 database based on the user query.
    Uses the dedicated OpenAI client for generating embeddings.

    Args:
        ctx: The run context containing dependencies (embedding client, DB pool).
        query: The user's search query.
        top_k: The maximum number of snippets to retrieve (default: 5).

    Returns:
        A formatted string containing the retrieved documentation snippets, or an error message.
    """
    print(f"Retrieving context for query: '{query}'")

    # Dependencies are now validated at startup, but check db_pool specifically as it can fail connection
    if not ctx.deps.db_pool:
        return "Error: Database connection is not available for retrieval."
    # openai_embedding_client is guaranteed by startup validation if we reach here

    try:
        # 1. Generate query embedding using the dedicated embedding client
        embedding_response = await ctx.deps.openai_embedding_client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL,
        )
        query_embedding = embedding_response.data[0].embedding
        embedding_str = str(query_embedding)

        # 2. Query the database (ensure table/column names match your schema)
        sql_query = """
            SELECT title, description, code, source_url, language,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM context7_docs -- Ensure 'context7_docs' is correct
            ORDER BY similarity DESC
            LIMIT $2;
        """
        rows = await ctx.deps.db_pool.fetch(sql_query, embedding_str, top_k)

        if not rows:
            print("No relevant snippets found in the database.")
            return "No relevant documentation snippets found for this query."

        # 3. Format the results
        formatted_snippets = []
        print(f"Found {len(rows)} relevant snippets.")
        for i, row in enumerate(rows):
            title = row.get('title', 'N/A')
            source_url = row.get('source_url', 'N/A')
            similarity = row.get('similarity', 0.0)
            language = row.get('language')
            description = row.get('description')
            code = row.get('code')

            snippet_text = f"<snippet index=\"{i+1}\" source=\"{source_url}\" similarity=\"{similarity:.4f}\">\n"
            snippet_text += f"Title: {title}\n"
            if language: snippet_text += f"Language: {language}\n"
            if description: snippet_text += f"Description: {description}\n"
            if code: snippet_text += f"Code:\n```\n{code}\n```\n"
            snippet_text += f"</snippet>"
            formatted_snippets.append(snippet_text)

        return "\n\n" + "\n\n".join(formatted_snippets) + "\n\n"

    # Specific database errors
    except asyncpg.exceptions.UndefinedTableError:
         print("Error: The table 'context7_docs' does not exist in the database.")
         return "Error: The required documentation table ('context7_docs') was not found."
    except asyncpg.exceptions.UndefinedColumnError as e:
         print(f"Error: A required column is missing in 'context7_docs': {e}")
         return f"Error: Database schema mismatch. Missing column: {e}"
    except asyncpg.exceptions.InvalidTextRepresentationError:
         print("Error: Could not convert embedding string to vector. Ensure pgvector is enabled.")
         return "Error: Database vector format issue."
    # General OpenAI API errors during embedding
    except Exception as e: # Catch OpenAI errors specifically if needed
        print(f"Error during documentation retrieval: {type(e).__name__}: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed traceback
        return f"An error occurred while retrieving documentation: {type(e).__name__}"


# --- Helper function for graph integration ---
async def run_context7_retrieve(query: str, top_k: int = 5) -> str:
    """
    Standalone helper to run the RAG retrieval process.
    Used by the LangGraph graph. Initializes its own dependencies.
    """
    # Ensure embedding client was initialized globally
    if not openai_embedding_client:
        return "Error: OpenAI client for embeddings is not configured globally."

    async with database_connect() as pool:
        if not pool:
            return "Error: Could not connect to the database for retrieval."

        # Create dependencies for the tool call
        deps = AgentDependencies(
            openai_embedding_client=openai_embedding_client,
            db_pool=pool
        )
        # Create a dummy RunContext
        dummy_run_context = RunContext(deps=deps, run_id="graph_retrieval_helper")

        # Call the tool function directly
        return await retrieve_context7_docs(dummy_run_context, query, top_k)


# --- Main Execution Logic ---
async def run_agent_query(
    query: str,
    message_history: Optional[List[Any]] = None  # Should be List[ModelMessage]
) -> Optional[Any]:
    """
    Sets up dependencies and runs the agent for a single query with optional chat history.
    Uses manual pool management to potentially avoid context manager issues with asyncpg.
    Expects message_history as a list of ModelMessage objects (not dicts).
    """
    print(f"\nProcessing query: '{query}'")
    pool = None # Initialize pool to None
    try:
        # Manually connect to the database
        if not DATABASE_URL:
            print("Database URL not configured. Cannot proceed.")
            return None
        try:
            print("Connecting to database manually...")
            pool = await asyncpg.create_pool(
                DATABASE_URL, min_size=1, max_size=10, command_timeout=60, timeout=300
            )
            print("Database connection established manually.")
        except (asyncpg.exceptions.InvalidCatalogNameError, OSError, ConnectionRefusedError, asyncio.TimeoutError) as db_conn_err:
            print(f"Database connection failed: {type(db_conn_err).__name__}: {db_conn_err}")
            # Return None to be handled by the caller (Streamlit UI)
            return None

        # Proceed only if pool connection was successful
        deps = AgentDependencies(
            openai_embedding_client=openai_embedding_client,
            db_pool=pool,
        )

        # Run the agent with full ModelMessage history for chat memory
        result = await context7_agent.run(
            query,
            deps=deps,
            message_history=message_history
        )
        # Return the full result object on success
        return result

    # Catch exceptions specifically from the agent run or dependency setup
    except Exception as e:
        print(f"Error during agent execution or setup: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Return None to indicate failure to the caller (Streamlit UI)
        return None
    finally:
        # Ensure the pool is closed if it was successfully created
        if pool:
            try:
                await pool.close()
                print("Database connection pool closed manually.")
            except Exception as close_exc:
                # Log error during close, but don't mask the original exception
                print(f"Error closing database pool manually: {close_exc}")

async def interactive_loop():
    """Runs an interactive command-line loop for the agent."""
    print("\n--- Context7 RAG Agent Interactive Mode ---")
    print("Using LLM:", LLM_MODEL)
    print("Using Embedding Model:", EMBEDDING_MODEL)
    print("Type your query below or 'exit' to quit.")
    messages: List[Dict[str, Any]] = [] # Initialize message history list
    while True:
        try:
            user_query = input("> ")
            if user_query.lower() in ['exit', 'quit']:
                break
            if not user_query.strip():
                continue

            # Pass the current message history to the agent run
            agent_run_result = await run_agent_query(user_query, message_history=messages)

            # Check if the run was successful and yielded a result object
            if agent_run_result:
                # Try to access the parsed output (RagResponse)
                response_output: Optional[RagResponse] = None
                if hasattr(agent_run_result, 'output') and isinstance(agent_run_result.output, RagResponse):
                    response_output = agent_run_result.output

                print("\nAssistant:")
                if response_output:
                    # Display the answer
                    print(response_output.answer)

                    # Display snippets if available
                    if response_output.retrieved_snippets:
                        print("\nSnippets Used:")
                        for i, snippet in enumerate(response_output.retrieved_snippets):
                            if isinstance(snippet, dict):
                                title = snippet.get('title', 'N/A')
                                sim = snippet.get('similarity', 0.0)
                                url = snippet.get('source_url', 'N/A')
                                print(f"  {i+1}. {title} (Similarity: {sim:.4f}) - {url}")
                            else:
                                print(f"  {i+1}. Invalid snippet format: {snippet}")

                    # Display confidence if available
                    if response_output.confidence_score is not None:
                        print(f"\nConfidence: {response_output.confidence_score:.2f}")

                elif hasattr(agent_run_result, 'raw_response'):
                    # Fallback if parsing failed but raw response exists
                    print("[Agent run completed but failed to parse output into RagResponse structure.]")
                    print(f"Raw Response: {agent_run_result.raw_response}")
                else:
                    # If no output and no raw response
                    print("[Agent run completed but produced no recognizable output.]")

                print("-" * 60)

                # Update message history using .new_messages() from the result object
                if hasattr(agent_run_result, 'new_messages'):
                    new_msgs = agent_run_result.new_messages()
                    if isinstance(new_msgs, list):
                        messages.extend(new_msgs)
                        # Optional: Print history for debugging
                        # print(f"Updated History: {messages}")
                    else:
                        print("Warning: agent_run_result.new_messages() did not return a list.")
                else:
                    print("Warning: agent_run_result does not have a 'new_messages' method.")

            else:
                # Handle case where run_agent_query returned None (due to an exception)
                print("Agent failed to produce a result object.")


        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred in the interactive loop: {type(e).__name__}: {e}")
            # import traceback; traceback.print_exc()

if __name__ == "__main__":
    # Startup validations ensure clients/agent are initialized if we reach here

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_loop())
    elif len(sys.argv) > 1:
        # Run a single query (Note: This path doesn't use history)
        query = " ".join(sys.argv[1:])
        # Run agent query directly, no history management here
        agent_run_result_single = asyncio.run(run_agent_query(query))
        if agent_run_result_single and hasattr(agent_run_result_single, 'output') and isinstance(agent_run_result_single.output, RagResponse):
            result_output = agent_run_result_single.output
            print("\nAssistant:")
            print(result_output.answer)
            if result_output.retrieved_snippets:
                print("\nSnippets Used:")
                for i, snippet in enumerate(result_output.retrieved_snippets):
                     if isinstance(snippet, dict):
                         title = snippet.get('title', 'N/A')
                         sim = snippet.get('similarity', 0.0)
                         url = snippet.get('source_url', 'N/A')
                         print(f"  {i+1}. {title} (Similarity: {sim:.4f}) - {url}")
                     else:
                         print(f"  {i+1}. Invalid snippet format.")
            if result_output.confidence_score is not None:
                 print(f"\nConfidence: {result_output.confidence_score:.2f}")
        else:
            print("Agent failed to produce a valid response.")
            # Optional: print raw response if available
            # if agent_run_result_single and hasattr(agent_run_result_single, 'raw_response'):
            #     print(f"Raw Response: {agent_run_result_single.raw_response}")
    else:
        # Default to interactive mode if no args
        print("No query provided. Starting interactive mode.")
        print("Usage:")
        print("  python context7_coder.py interactive          - Start interactive mode")
        print("  python context7_coder.py <your query here>    - Run a single query")
        asyncio.run(interactive_loop())
