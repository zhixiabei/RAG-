import unittest

from rag_app.application.auth import create_session, credentials_match, verify_session


class AuthTest(unittest.TestCase):
    def test_signed_session_round_trip(self):
        token = create_session("admin", "personal", "secret", 60)

        payload = verify_session(token, "secret")

        self.assertEqual(payload["sub"], "admin")
        self.assertEqual(payload["owner_id"], "personal")

    def test_rejects_tampered_and_expired_sessions(self):
        token = create_session("admin", "personal", "secret", 60)
        expired = create_session("admin", "personal", "secret", -1)

        self.assertIsNone(verify_session(token + "x", "secret"))
        self.assertIsNone(verify_session(expired, "secret"))

    def test_credentials_require_both_exact_values(self):
        self.assertTrue(credentials_match("admin", "password", "admin", "password"))
        self.assertFalse(credentials_match("admin", "wrong", "admin", "password"))
        self.assertFalse(credentials_match("other", "password", "admin", "password"))


if __name__ == "__main__":
    unittest.main()
