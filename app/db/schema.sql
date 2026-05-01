CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS topics (
    topic_id    TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384),
    entity_type TEXT DEFAULT 'other',
    confidence  FLOAT DEFAULT 0.0,
    timestamps  FLOAT[] DEFAULT '{}',
    access_count INT DEFAULT 0,
    decay_param  FLOAT DEFAULT 0.5,
    last_activation FLOAT DEFAULT 0.0,
    created_at  FLOAT
);

CREATE INDEX IF NOT EXISTS topics_embedding_idx
    ON topics USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS topics_user_idx
    ON topics (user_id);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    active_session TEXT,
    retrieval_threshold FLOAT DEFAULT -1.0,
    noise_param FLOAT DEFAULT 0.25,
    speech_in_weight FLOAT DEFAULT 0.6
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    active_users TEXT[] DEFAULT '{}',
    created_at FLOAT,
    updated_at FLOAT
);