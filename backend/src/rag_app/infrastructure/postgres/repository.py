from __future__ import annotations

import json
from typing import Any

from sqlalchemy import create_engine, text

from ...domain.ids import vector_point_id
from ...domain.models import ParsedChunk
from .schema import SCHEMA_STATEMENTS


class PostgresRepository:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 3})

    def close(self) -> None:
        self.engine.dispose()

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(text(statement))

    def knowledge_base_exists(self, knowledge_base_id: str) -> bool:
        return self.get_knowledge_base(knowledge_base_id) is not None

    def create_knowledge_base(self, knowledge_base_id: str, name: str, description: str, embedding_model: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO knowledge_bases (id, name, description, embedding_model) VALUES (:id, :name, :description, :embedding_model)"),
                {"id": knowledge_base_id, "name": name, "description": description, "embedding_model": embedding_model},
            )
        return self.get_knowledge_base(knowledge_base_id) or {}

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT * FROM knowledge_bases ORDER BY created_at DESC")).mappings().all()
        return [dict(row) for row in rows]

    def get_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM knowledge_bases WHERE id = :id"), {"id": knowledge_base_id}).mappings().first()
        return dict(row) if row else None

    def create_document(self, item: dict[str, Any]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text("""
                    INSERT INTO documents
                    (id, knowledge_base_id, title, file_name, mime_type, source_object_key, status)
                    VALUES (:id, :knowledge_base_id, :title, :file_name, :mime_type, :source_object_key, :status)
                """),
                item,
            )

    def update_document(self, document_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = []
        parameters: dict[str, Any] = {"id": document_id}
        for key, value in fields.items():
            assignments.append(f"{key} = :{key}")
            parameters[key] = value
        assignments.append("updated_at = now()")
        with self.engine.begin() as connection:
            connection.execute(text(f"UPDATE documents SET {', '.join(assignments)} WHERE id = :id"), parameters)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM documents WHERE id = :id"), {"id": document_id}).mappings().first()
        return dict(row) if row else None

    def list_documents(self, knowledge_base_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM documents WHERE knowledge_base_id = :id ORDER BY created_at DESC"),
                {"id": knowledge_base_id},
            ).mappings().all()
        return [dict(row) for row in rows]

    def replace_chunks(self, document_id: str, knowledge_base_id: str, chunks: list[ParsedChunk]) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM document_chunks WHERE document_id = :id"), {"id": document_id})
            for chunk in chunks:
                chunk_id = f"{document_id}:{chunk.index}"
                connection.execute(
                    text("""
                        INSERT INTO document_chunks
                        (id, document_id, knowledge_base_id, chunk_index, text, page_number, section_path, qdrant_point_id)
                        VALUES (:id, :document_id, :knowledge_base_id, :chunk_index, :text, :page_number, :section_path, :qdrant_point_id)
                    """),
                    {
                        "id": chunk_id,
                        "document_id": document_id,
                        "knowledge_base_id": knowledge_base_id,
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "section_path": chunk.section_path,
                        "qdrant_point_id": vector_point_id(chunk_id),
                    },
                )

    def create_conversation(self, conversation_id: str, knowledge_base_id: str, title: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO conversations (id, knowledge_base_id, title) VALUES (:id, :kb, :title)"),
                {"id": conversation_id, "kb": knowledge_base_id, "title": title},
            )
        return self.get_conversation(conversation_id) or {}

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM conversations WHERE id = :id"),
                {"id": conversation_id},
            ).mappings().first()
        return dict(row) if row else None

    def list_conversations(self, knowledge_base_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text("""
                    SELECT
                        conversations.*,
                        (SELECT count(*) * 2 FROM messages WHERE conversation_id = conversations.id) AS message_count,
                        (SELECT question FROM messages WHERE conversation_id = conversations.id ORDER BY id DESC LIMIT 1) AS last_message
                    FROM conversations
                    WHERE knowledge_base_id = :kb
                    ORDER BY updated_at DESC, created_at DESC
                """),
                {"kb": knowledge_base_id},
            ).mappings().all()
        return [dict(row) for row in rows]

    def add_message(self, conversation_id: str, knowledge_base_id: str, question: str, answer: str, citations: list[dict[str, Any]]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text("""
                    INSERT INTO messages (knowledge_base_id, conversation_id, question, answer, citations)
                    VALUES (:kb, :conversation, :question, :answer, CAST(:citations AS JSONB))
                """),
                {
                    "kb": knowledge_base_id,
                    "conversation": conversation_id,
                    "question": question,
                    "answer": answer,
                    "citations": json.dumps(citations, ensure_ascii=False),
                },
            )
            connection.execute(
                text("UPDATE conversations SET updated_at = now() WHERE id = :id"),
                {"id": conversation_id},
            )

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT id, question, answer, citations, created_at FROM messages WHERE conversation_id = :conversation ORDER BY id"),
                {"conversation": conversation_id},
            ).mappings().all()

        messages = []
        for row in rows:
            messages.extend(
                (
                    {
                        "id": f"{row['id']}:user",
                        "role": "user",
                        "content": row["question"],
                        "created_at": row["created_at"],
                    },
                    {
                        "id": f"{row['id']}:assistant",
                        "role": "assistant",
                        "content": row["answer"],
                        "citations": row["citations"] or [],
                        "created_at": row["created_at"],
                    },
                )
            )
        return messages
