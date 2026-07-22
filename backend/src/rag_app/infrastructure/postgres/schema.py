SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS knowledge_bases (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        embedding_model TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id),
        title TEXT NOT NULL,
        file_name TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        source_object_key TEXT NOT NULL,
        status TEXT NOT NULL,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id),
        knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id),
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        page_number INTEGER,
        section_path TEXT,
        qdrant_point_id TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id),
        title TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id),
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        citations JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS conversation_id TEXT REFERENCES conversations(id)",
    """
    INSERT INTO conversations (id, knowledge_base_id, title, created_at, updated_at)
    SELECT
        'legacy-' || md5(knowledge_base_id),
        knowledge_base_id,
        '历史对话',
        min(created_at),
        max(created_at)
    FROM messages
    WHERE conversation_id IS NULL
    GROUP BY knowledge_base_id
    ON CONFLICT (id) DO NOTHING
    """,
    """
    UPDATE messages
    SET conversation_id = 'legacy-' || md5(knowledge_base_id)
    WHERE conversation_id IS NULL
    """,
    "ALTER TABLE messages ALTER COLUMN conversation_id SET NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(knowledge_base_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_kb_updated ON conversations(knowledge_base_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id)",
)
