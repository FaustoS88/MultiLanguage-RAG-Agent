# Context7 Documentation Extractor & RAG Agent

This project provides a comprehensive system for extracting, storing, and querying curated documentation snippets from Context7 library endpoints. It supports Retrieval-Augmented Generation (RAG) workflows by embedding documentation snippets and enabling AI-powered question answering and code generation.

---

## Project Components

### 1. Documentation Extractor & Crawler

- **Purpose:** Fetches documentation snippets from Context7 `llms.txt` endpoints, parses them, and stores them in a PostgreSQL database with vector embeddings for similarity search.
- **Key Files:**
  - `context7_extractor.py`: Extracts text snippets and generates embeddings using OpenAI API (for RAG).
  - `init_db.py`: Initializes the PostgreSQL database schema.
  - `create_context7_docs_table.sql`: SQL script to create the `context7_docs` table with pgvector extension.
  - `db_inspect.py`: Utility to inspect and manage stored snippets.

---

### 2. RAG Agent & Query Interface

- **Purpose:** Provides an AI assistant that answers user queries by retrieving relevant documentation snippets and generating answers or runnable code.
- **Key Files:**
  - `context7_coder.py`: Core agent implementation handling environment setup, database connection, retrieval, and AI model integration.
  - `agent_prompts.py`: Contains the system prompt guiding the agent's behavior and workflow.
  - `context7_coder_chat.py`: Streamlit-based chat UI for interactive querying of the RAG agent.

---

## Setup Instructions

### Dependencies

Install required Python packages:

```bash
python -m venv venv
source venv/bin/activate  # for Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
### Database Setup

You can use either a local PostgreSQL instance with pgvector or a managed service like Supabase.

#### Local PostgreSQL with Docker

```bash
docker run --name context7-pgvector \
  -e POSTGRES_PASSWORD=your_secret_password \
  -p 54533:5432 \
  -v /Users/faustosaccoccio/context7_postgres_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```

- Replace `your_secret_password` with a strong password.
- Ensure the volume path exists or Docker has permission to create it.

Connect to the database and run:

```bash
python init_db.py
```

This executes the SQL in `create_context7_docs_table.sql` to create the necessary table and enable the vector extension.

#### Supabase PostgreSQL

- Enable the `vector` extension in your Supabase project.
- Connect using your Supabase credentials.
- Run the SQL commands in `create_context7_docs_table.sql`.

---

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

- Adjust values according to your setup.
- `OPENAI_API_KEY` is required for embeddings.
- `OPENROUTER_API_KEY` and `LLM_MODEL` configure the chat model.

---

## Running the Extractor & Agent

### Extract Documentation Snippets

Run the extractor script to fetch and store snippets:

```bash
# Extract and generate embeddings
cd crawler
python context7_extractor.py
```
---

### Query the Agent

#### Interactive CLI

```bash
python context7_agent.py
```

Type your queries and get answers with relevant documentation snippets.

#### Single Query

```bash
python context7_agent.py "your question here"
```

#### Streamlit Chat UI

Run the chat interface:

```bash
streamlit run context7_chat.py
```

The Streamlit UI preserves conversation context up to the training context window size of the model, which even for models with declared 1 million token windows is effectively around 160,000 to 200,000 tokens. Staying within this context window reduces hallucinations and improves response accuracy.

##### Developer Tools in Streamlit

- **Session Management:** Start a new conversation to clear chat history.
- **Show Raw Conversation History:** Toggle to visualize the full JSON chat history.
- **Download Chat as JSON:** Download the entire conversation history as a JSON file.
- **Environment Check:** View key environment variables such as LLM model, embedding model, and database URL status to verify configuration.

These tools help with debugging, session control, and environment verification directly from the UI.

---

## Database Schema

The `context7_docs` table stores documentation snippets with vector embeddings for similarity search.

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

The RAG agent uses a multi-stage graph pipeline:

- **Reasoner:** Crafts chain-of-thought reasoning.
- **Coder:** Writes executable Python code from question, context, and reasoning.
- **Refiner:** Polishes and comments code for clarity.
- **Human-in-Loop:** Final review and feedback.

The agent answers queries by retrieving relevant snippets from the database and synthesizing answers or runnable code, strictly based on Context7 documentation list.

---

## Inspecting & Managing Data

Use `db_inspect.py` to manage stored snippets:

```bash
python db_inspect.py --count
python db_inspect.py --list [LIMIT]
python db_inspect.py --view SNIPPET_ID
python db_inspect.py --delete-library LIBRARY_ID
python db_inspect.py --delete-all
```

---

## Summary

This project enables dynamic, version-specific documentation retrieval and AI-assisted coding using Context7 libraries. It combines web crawling, vector embeddings, and advanced AI models to provide a powerful RAG system for developers.
