from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from openai import AsyncOpenAI, RateLimitError, OpenAIError
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import List
import psycopg2
import logging
import asyncio
import random
import time
import re
import os

# --- Logging Configuration ---
# Silence noisy httpx/httpcore INFO logs
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
# Progress logger
progress = logging.getLogger("context7.progress")
progress.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
progress.addHandler(handler)

load_dotenv()

# --- Database Configuration ---
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "54533")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Approx token limit for embedding model (conservative estimate based on characters)
# text-embedding-3-small limit is 8191 tokens. Avg 4 chars/token -> ~32k chars
# Let's use a safer limit, e.g., 28000 characters.
MAX_EMBEDDING_CHARS = 28000

# --- Context7 Configuration ---
# List of Context7 library IDs to fetch (Sorted Alphabetically)
CONTEXT7_LIBRARY_IDS = sorted([
    "/modelcontextprotocol/servers",              # MCP Server
    "/modelcontextprotocol/modelcontextprotocol", # MCP Protocol
    "/aws-samples/amazon-bedrock-samples",        # amazon bedrock
    "/themanojdesai/python-a2a",                  # a2a_python
    "/google/a2a",                                # google_a2a
    "/angular/angular",                           # angular
    "/unclecode/crawl4ai",                        # crawl4ai
    "/tiangolo/fastapi",                          # fastapi
    "/golang/go",                                 # Go
    "/google-gemini/generative-ai-js",            # google gemini
    "/googleapis/python-genai",                   # Google Gen AI Python SDK
    "/jestjs/jest",                               # Jest
    "/matplotlib/matplotlib",                     # matplotlib
    "/nestjs/docs.nestjs.com",                    # nest.js & docs.nestjs.com
    "/vercel/next.js",                            # next.js
    "/vega/altair",                               # altair
    "/python/cpython",                            # python
    "/microsoftdocs/cpp-docs",                    # C++     
    "/facebook/react",                            # react
    "/solana-labs/solana",                        # solana
    "/anza-xyz/solana-sdk",                       # solana sdk (Using general repo)
    "/rust-lang/rust",                            # rust
    "/ethereum/solidity",                         # solidity
    "/tradingview/lightweight-charts",            # Trading_view 
    "/arunkbhaskar/pinescript",                   # pinescript
    "/quantconnect/documentation",                # QuantConnect
    "/quantconnect/research",
    "/quantconnect/tutorials"
    "/supabase/supabase",                         # supabase
    "/microsoft/TypeScript",                      # typescript
    "/upstash/context7",                          # Context7 MCP Server (so the rag knows how to use the docs)
    "/web3/web3.js",                              # web3.js
    "/ethereum/web3.py",                          # web3.py
    "/pydantic/pydantic-ai",                      # pydantic-ai
    "/crewaiinc/crewai",                          # crew.ai
    "/run-llama/llama_index",                     # llama_index
    "/langchain-ai/langgraph",                    # langgraph
    "/langfuse/langfuse-docs",                    # langfuse
    "/langchain-ai/langchain",                    # langchain
    "/huggingface/smolagents",                    # smolagents
    "/scikit-learn/scikit-learn",                 # scikit-learn
    "/pandas-dev/pandas",                         # pandas
    "/numpy/numpy.org",                           # numpy
    "/tensorflow/docs",                           # tensorflow
    "/pytorch/pytorch",                           # pytorch
    "/huggingface/transformers",                  # transformers
    "/berriai/litellm",                           # litellm
    "/hkuds/lightrag",                            # lightrag
    "/getzep/graphiti",                           # graphiti
    
])
# Base URL for Context7 library pages
CONTEXT7_BASE_URL = "https://context7.com"

# --- Crawler Configuration ---
CRAWLER_DELAY_SECONDS = 2 # Delay between crawl requests

def parse_llms_text(raw_text: str, library_id: str) -> list:
    """Parses the raw llms.txt content into structured snippets."""
    snippets = []
    library_path = library_id.lstrip('/')
    raw_snippets = raw_text.split('----------------------------------------')

    for snippet_index, raw_snippet in enumerate(raw_snippets): # Add index for placeholder
        raw_snippet = raw_snippet.strip()
        if not raw_snippet:
            continue

        data = {
            "library_id": library_id,
            "title": None,
            "description": None,
            "source_url": None,
            "language": None,
            "code": None
        }

        # Use regex to extract fields (case-insensitive matching for keys)
        title_match = re.search(r'^TITLE:\s*(.*)', raw_snippet, re.IGNORECASE | re.MULTILINE)
        desc_match = re.search(r'^DESCRIPTION:\s*(.*)', raw_snippet, re.IGNORECASE | re.MULTILINE)
        source_match = re.search(r'^SOURCE:\s*(https?://[^\s]+)', raw_snippet, re.IGNORECASE | re.MULTILINE)
        lang_match = re.search(r'^LANGUAGE:\s*(.*)', raw_snippet, re.IGNORECASE | re.MULTILINE)
        # Code might span multiple lines, capture everything after CODE: ``` until ```
        code_match = re.search(r'^CODE:\s*```[^\n]*\n(.*?)\n```', raw_snippet, re.IGNORECASE | re.MULTILINE | re.DOTALL)

        if title_match:
            data["title"] = title_match.group(1).strip()
        if desc_match:
            data["description"] = desc_match.group(1).strip()

        # Handle source_url: Use parsed value or generate placeholder
        if source_match:
            data["source_url"] = source_match.group(1).strip()
        else:
            placeholder_title = data["title"] if data["title"] else f"snippet_{snippet_index}"
            slug_title = re.sub(r'\W+', '-', placeholder_title).lower().strip('-')
            if not slug_title or slug_title == f"snippet-{snippet_index}":
                 slug_title = f"snippet-{snippet_index}"
            data["source_url"] = f"https://context7.com/{library_path}/placeholder_{slug_title}"
            print(f"Warning: Missing SOURCE for snippet in {library_id}. Using placeholder: {data['source_url']}")

        if lang_match:
            data["language"] = lang_match.group(1).strip()
        if code_match:
            data["code"] = code_match.group(1).strip()
        elif not any([title_match, desc_match, source_match, lang_match]):
            data["code"] = raw_snippet

        if data["title"] or data["code"]:
            snippets.append(data)
        else:
             print(f"Warning: Skipping snippet in {library_id} due to missing critical data (title/code). Content: {raw_snippet[:100]}...")

    return snippets

async def get_embedding(text: str, client: AsyncOpenAI, snippet_ref: str) -> List[float]:
    """Get embedding vector from OpenAI, checking size limit, with retry/back-off."""
    if not text:
        print(f"Warning: Attempted to embed empty text for {snippet_ref}. Returning zero vector.")
        return [0.0] * 1536

    if len(text) > MAX_EMBEDDING_CHARS:
        print(f"Warning: Snippet text for {snippet_ref} exceeds character limit ({len(text)} > {MAX_EMBEDDING_CHARS}). Skipping embedding, returning zero vector.")
        return [0.0] * 1536

    max_retries = 5
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(0.2 + random.random() * 0.1)
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding

        except RateLimitError:
            delay = base_delay * (2 ** attempt) + random.random()
            print(f"Rate limit hit for {snippet_ref}, retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

        except OpenAIError as e:
            print(f"OpenAI error for {snippet_ref}: {e}. Retrying in {base_delay}s...")
            await asyncio.sleep(base_delay)

        except Exception as e:
            print(f"Unexpected error for {snippet_ref}: {e}. Aborting retries.")
            break

    print(f"Failed to get embedding for {snippet_ref} after {max_retries} attempts.")
    return [0.0] * 1536

async def insert_docs_to_db(conn, docs_data: list, openai_client: AsyncOpenAI):
    """Generates embeddings and inserts parsed documentation snippets into the database."""
    if not docs_data:
        return 0

    insert_tuples = []
    for i, d in enumerate(docs_data):
        text_to_embed = (
            f"Title: {d.get('title','')}\n"
            f"Description: {d.get('description','')}\n"
            f"Language: {d.get('language','')}\n"
            f"Code: {d.get('code','')}"
        ).strip()
        snippet_ref = f"{d.get('library_id')} snippet {i}"
        embedding_vector = await get_embedding(text_to_embed, openai_client, snippet_ref)
        insert_tuples.append((
            d.get("library_id"),
            d.get("title"),
            d.get("description"),
            d.get("source_url"),
            d.get("language"),
            d.get("code"),
            embedding_vector
        ))

    sql = """
        INSERT INTO context7_docs (library_id, title, description, source_url, language, code, embedding)
        VALUES %s
        ON CONFLICT (library_id, source_url) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, insert_tuples, template="(%s, %s, %s, %s, %s, %s, %s)")
            inserted_count = cur.rowcount
            conn.commit()
            return inserted_count
    except Exception as e:
        print(f"Database insertion error: {e}")
        conn.rollback()
        return 0

async def main():
    """Main function to crawl Context7 raw text endpoints, generate embeddings, and store in DB."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in .env file. Cannot generate embeddings.")
        return
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    progress.info(f"OpenAI client initialized. Using embedding model: {EMBEDDING_MODEL}")

    # --- Database Connection ---
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        progress.info("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    # --- Initialize Crawler ---
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        progress.info(f"Initialized crawler. Processing {len(CONTEXT7_LIBRARY_IDS)} libraries...")
        processed_count = 0
        failed_libs = []

        for library_id in CONTEXT7_LIBRARY_IDS:
            library_path = library_id.lstrip('/')
            target_url = f"{CONTEXT7_BASE_URL}/{library_path}/llms.txt"
            progress.info(f"Fetching {library_id} → {target_url}")

            try:
                result = await crawler.arun(url=target_url, config=run_config)
                fetched_content = result.markdown_v2.raw_markdown if result.success and result.markdown_v2 else None

                if fetched_content:
                    progress.info(f"Parsing snippets for {library_id}")
                    parsed_snippets = parse_llms_text(fetched_content, library_id)

                    if parsed_snippets:
                        progress.info(f"Embedding {library_id} ({len(parsed_snippets)} snippets)…")
                        insert_count = await insert_docs_to_db(conn, parsed_snippets, openai_client)
                        processed_count += insert_count
                        progress.info(f"Inserted {insert_count} snippets for {library_id}")
                    else:
                        progress.info(f"No snippets parsed for {library_id}")
                elif result.success:
                    progress.info(f"No text found at {target_url} for {library_id}")
                else:
                    progress.info(f"Failed to crawl {library_id}: {result.error_message}")
                    failed_libs.append(library_id)

            except Exception as e:
                progress.info(f"Error processing {library_id}: {e}")
                failed_libs.append(library_id)

            await asyncio.sleep(CRAWLER_DELAY_SECONDS)

    if conn:
        conn.close()
        progress.info("Database connection closed.")

    progress.info(f"--- Extraction Summary: {processed_count} snippets inserted; {len(failed_libs)} failures. ---")

if __name__ == "__main__":
    # Create a dummy .env file if missing
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Database Credentials\n")
            f.write("DB_NAME=context7_db\n")
            f.write("DB_USER=postgres\n")
            f.write("DB_PASSWORD=YOUR_SUPER_SECRET_PASSWORD\n")
            f.write("DB_HOST=localhost\n")
            f.write(f"DB_PORT={DB_PORT}\n")
            f.write("\n# OpenAI Credentials\n")
            f.write("OPENAI_API_KEY=YOUR_OPENAI_API_KEY\n")
            f.write("# EMBEDDING_MODEL=text-embedding-3-small # Optional: uncomment to override default\n")
        print("Created a sample .env file. Please update it with your database and OpenAI credentials.")

    if not OPENAI_API_KEY:
         print("Error: OPENAI_API_KEY missing in .env file.")
    elif not DB_PASSWORD:
         print("Error: DB_PASSWORD missing in .env file.")
    else:
        asyncio.run(main())
