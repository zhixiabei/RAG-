import json
import unittest

from rag_app.infrastructure.postgres.repository import PostgresRepository


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    def execute(self, statement, parameters):
        query = str(statement)
        self.calls.append((query, parameters))
        if query.lstrip().startswith("SELECT"):
            return FakeResult(self.rows)
        return None


class FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeEngine:
    def __init__(self, connection):
        self.connection = connection

    def begin(self):
        return FakeTransaction(self.connection)


class PostgresRepositoryMessageTest(unittest.TestCase):
    @staticmethod
    def repository(connection):
        repository = PostgresRepository.__new__(PostgresRepository)
        repository.engine = FakeEngine(connection)
        return repository

    def test_add_message_serializes_metrics(self):
        connection = FakeConnection()
        repository = self.repository(connection)
        metrics = {"responseTimeMs": 12.5, "tokenUsage": {"available": True, "total_tokens": 42}}

        repository.add_message("conversation-1", "kb-1", "question", "answer", [], metrics)

        insert_query, parameters = connection.calls[0]
        self.assertIn("metrics", insert_query)
        self.assertEqual(json.loads(parameters["metrics"]), metrics)

    def test_list_messages_restores_assistant_metrics(self):
        metrics = {"responseTimeMs": 12.5, "serverResponseTimeMs": 12.5, "tokenUsage": {"available": True}}
        connection = FakeConnection([{
            "id": 7,
            "question": "question",
            "answer": "answer",
            "citations": [],
            "metrics": metrics,
            "created_at": "2026-08-25T00:00:00Z",
        }])
        repository = self.repository(connection)

        messages = repository.list_messages("conversation-1")

        self.assertIn("metrics", connection.calls[0][0])
        self.assertEqual(messages[1]["metrics"], metrics)


if __name__ == "__main__":
    unittest.main()
