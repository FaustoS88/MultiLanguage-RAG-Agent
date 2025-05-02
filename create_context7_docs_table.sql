-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table to store Context7 documentation
CREATE TABLE context7_docs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library_id VARCHAR(255) NOT NULL,
    title TEXT,
    description TEXT,
    source_url VARCHAR(1024) NOT NULL, -- Increased length and made NOT NULL for constraint
    language VARCHAR(50),
    code TEXT,
    embedding VECTOR(1536), -- Assuming a common embedding dimension like OpenAI's
    CONSTRAINT unique_library_source UNIQUE (library_id, source_url) -- Add unique constraint
);

-- Create an index on library_id for faster lookups (optional, covered by unique constraint index)
-- CREATE INDEX idx_context7_docs_library_id ON context7_docs (library_id);

-- Create an index on source_url (optional, covered by unique constraint index)
-- CREATE INDEX idx_context7_docs_source_url ON context7_docs (source_url);

-- Optional: Create an index on the embedding column for vector search (replace with appropriate index type for your needs, e.g., ivfflat, hnsw)
-- CREATE INDEX idx_context7_docs_embedding ON context7_docs USING ivfflat (embedding vector_ops) WITH (lists = 100);
