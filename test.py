import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from database import db_service
from services import fusion_engine, text_prediction


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.original_db_name = db_service.DB_NAME
        db_service.DB_NAME = self.db_path
        self._create_tables()

    def tearDown(self):
        db_service.DB_NAME = self.original_db_name
        self.temp_dir.cleanup()

    def _create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_text TEXT,
            face_emotion TEXT,
            mental_state TEXT,
            insight TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def test_register_and_login_hashes_password(self):
        self.assertTrue(
            db_service.register_user("Test User", "tester", "t@example.com", "secret123")
        )

        conn = sqlite3.connect(self.db_path)
        stored_password = conn.execute(
            "SELECT password FROM users WHERE username='tester'"
        ).fetchone()[0]
        conn.close()

        self.assertTrue(stored_password.startswith("pbkdf2_sha256$"))
        user = db_service.login_user("tester", "secret123")
        self.assertEqual(user[1], "Test User")

    def test_save_prediction_and_latest_emotion(self):
        db_service.register_user("Test User", "tester", "t@example.com", "secret123")
        user = db_service.login_user("tester", "secret123")

        db_service.save_prediction(user[0], "sample text", "Sad", "Stress", "Needs a break")
        db_service.save_prediction(user[0], "sample text 2", "Happy", "Positive", "Doing well")

        history = db_service.get_user_history(user[0])
        self.assertEqual(len(history), 2)
        self.assertEqual(db_service.get_latest_emotion(user[0]), "Positive")


class ServiceTests(unittest.TestCase):
    def test_combine_predictions_without_face(self):
        result = fusion_engine.combine_predictions(
            {"label": "Stress", "confidence": 0.88},
            None
        )
        self.assertEqual(result["mental_state"], "Stress")
        self.assertEqual(result["face_emotion"], "Not Provided")

    def test_text_prediction_validates_empty_input(self):
        with self.assertRaises(ValueError):
            text_prediction.predict_text_mental_state("   ")

    @mock.patch("services.text_prediction._load_assets")
    def test_text_prediction_returns_label_and_confidence(self, mock_load_assets):
        class DummyVectorizer:
            def transform(self, texts):
                class DummyArray:
                    def toarray(self_inner):
                        return [[0.1, 0.2, 0.3]]
                return DummyArray()

        class DummyModel:
            def predict(self, features, verbose=0):
                return [[0.1, 0.7, 0.2]]

        class DummyEncoder:
            classes_ = ["Anxiety", "Positive", "Stress"]

            def inverse_transform(self, values):
                return [self.classes_[values[0]]]

        mock_load_assets.return_value = (DummyModel(), DummyVectorizer(), DummyEncoder())
        result = text_prediction.predict_text_mental_state("I feel okay")

        self.assertEqual(result["label"], "Positive")
        self.assertGreater(result["confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
