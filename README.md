# Context7 Documentation Extractor & RAG Agent

This project provides a comprehensive system for extracting, storing, and querying curated documentation snippets from Context7 library endpoints. It supports Retrieval-Augmented Generation (RAG) workflows by embedding documentation snippets and enabling AI-powered question answering and code generation.

---

## Project Components

### 1. Documentation Extractor & Crawler

**Purpose:**  
Fetches documentation snippets from Context7 llms.txt endpoints, parses them, and stores them in a PostgreSQL database with vector embeddings for similarity search.

**Key Files:**  
- `context7_extractor.py`: Extracts text snippets and generates embeddings using the OpenAI API (for RAG).  
- `init_db.py`: Initializes the PostgreSQL database schema.  
- `create_context7_docs_table.sql`: SQL script to create the `context7_docs` table with the pgvector extension.  
- `db_inspect.py`: Utility to inspect and manage stored snippets.

### 2. RAG Agent & Query Interface

**Purpose:**  
Provides an AI assistant that answers user queries by retrieving relevant documentation snippets and generating answers or runnable code.

**Key Files:**  
- `context7_coder.py`: Core agent implementation handling environment setup, database connection, retrieval, and AI model integration.  
- `agent_prompts.py`: Contains the system prompt guiding the agent's behavior and workflow.  
- `context7_coder_chat.py`: Streamlit-based chat UI for interactive querying of the RAG agent with graph-aware capabilities.

**Graph-Aware Options:**  
- `context7_agent_simple.py`: A standalone, simple CLI agent that runs without invoking the LangGraph workflow for fast, direct queries.  
- `langgraph_context7_workflow.py`: An optional LangGraph workflow for multi-step planning, code generation, and refinement. This workflow is automatically invoked by the main agent when deeper reasoning or code execution is required.

---

## Setup Instructions

### Dependencies

Install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate  # For Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Database Setup

You can use either a local PostgreSQL instance with pgvector or a managed service like Supabase.

#### Local PostgreSQL with Docker

```bash
docker run --name context7-pgvector \
  -e POSTGRES_PASSWORD=your_secret_password \
  -p 54533:5432 \
  -v /path/to/your/local/data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```

Then connect to the database and initialize it by running:

```bash
python init_db.py
```

This command executes the SQL in `create_context7_docs_table.sql` to create the necessary table and enable the vector extension.

#### Supabase PostgreSQL

- Enable the vector extension in your Supabase project.  
- Connect using your Supabase credentials.  
- Run the SQL commands in `create_context7_docs_table.sql`.

### Environment Configuration

Create a `.env` file in the `context7` directory with the following variables:

```dotenv
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=54533

OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small

OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=openai/gpt-4.1-mini
DATABASE_URL=postgresql://user:password@host:port/dbname
```

Adjust the values according to your setup.  
- `OPENAI_API_KEY` is required for generating embeddings.  
- `OPENROUTER_API_KEY` and `LLM_MODEL` are used to configure the chat model.

---

## Running the Extractor & Agent

### Extract Documentation Snippets

Run the extractor script to fetch and store documentation snippets:

```bash
# Navigate to the crawler folder and run the extractor script
cd crawler
python context7_extractor.py
```

### Query the Agent

The project offers both simple and graph-aware CLI options, as well as a Streamlit chat UI.

#### Interactive CLI Options

- **Simple CLI (non-graph-aware):** Use this for fast, direct queries without invoking the multi-step LangGraph workflow:

```bash
python context7_agent_simple.py "your question here"
```

- **Graph-Aware CLI:** This version invokes a LangGraph workflow for multi-step reasoning, code generation, and refinement. You will see a `[workflow invoked]` message in the terminal when deeper processing is performed:

```bash
python context7_agent.py "your question here"
```

- **Single Query (graph-aware):**

```bash
python context7_agent.py "your question here"
```

#### Streamlit Chat UI

Launch the interactive chat UI that supports the graph-aware multi-stage process:

```bash
streamlit run context7_chat.py
```

The Streamlit UI preserves conversation context up to the training context window size of the model, which even for models with declared 1 million token windows is effectively around 160,000 to 200,000 tokens. Staying within this context window reduces hallucinations and improves response accuracy.

---

## Developer Tools in Streamlit

- **Session Management:** Start a new conversation to clear the chat history.  
- **Show Raw Conversation History:** Toggle to view the full JSON chat history.  
- **Download Chat as JSON:** Export the entire conversation history.  
- **Environment Check:** Verify key environment variables—including LLM model, embedding model, and database URL—to ensure proper configuration.

---

## Database Schema

The `context7_docs` table stores documentation snippets along with vector embeddings for similarity search.

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table to store Context7 documentation
CREATE TABLE context7_docs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library_id VARCHAR(255) NOT NULL,
    title TEXT,
    description TEXT,
    source_url VARCHAR(1024) NOT NULL,
    language VARCHAR(50),
    code TEXT,
    embedding VECTOR(1536),
    CONSTRAINT unique_library_source UNIQUE (library_id, source_url)
);
```

---

## Agent Behavior & Workflow

The RAG agent adopts a multi-stage graph pipeline with capabilities to adapt to the query complexity:

- **Reasoner:** Structures the search strategy and develops a chain-of-thought reasoning process.  
- **Retrieve:** Fetches the relevant documentation snippets from the database.  
- **Generate Code:** Writes executable Python code based on the context and reasoning.  
- **Refine:** Polishes the generated code for clarity and conciseness.  
- **Human-in-Loop:** Awaits final user review and feedback.

By default:  
- Simple queries bypass the graph pipeline for faster responses.  
- Complex or code-related queries trigger the LangGraph-based multi-step workflow (via `graph_workflow.py`), ensuring thorough reasoning and robust code generation.

---

## Inspecting & Managing Data

Use `db_inspect.py` to inspect and manage stored documentation snippets:

```bash
python db_inspect.py --count
python db_inspect.py --list [LIMIT]
python db_inspect.py --view SNIPPET_ID
python db_inspect.py --delete-library LIBRARY_ID
python db_inspect.py --delete-all
```

---

## Acknowledgments

- Cole Medin & the Archon project for inspiration on graph‑centric design patterns. 
- Context7 for providing the optimized documentation endpoints.   
- Pydantic‑AI for the agent and tool framework.  
- LangGraph for the orchestration engine powering the optional multi‑agent workflow.
