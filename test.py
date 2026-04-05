import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from database import db_service
from services import fusion_engine, text_prediction, safety_guard


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
            text_confidence REAL,
            face_confidence REAL,
            support_level TEXT,
            input_source TEXT,
            urgency_score REAL,
            triage_level TEXT,
            triage_reason TEXT,
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

        db_service.save_prediction(
            user[0], "sample text", "Sad", "Stress", "Needs a break",
            text_confidence=0.84, face_confidence=0.67, support_level="Moderate", input_source="Text + Face",
            urgency_score=0.72, triage_level="Needs Attention", triage_reason="hopeless"
        )
        db_service.save_prediction(
            user[0], "sample text 2", "Happy", "Positive", "Doing well",
            text_confidence=0.91, face_confidence=0.76, support_level="Low", input_source="Text + Face",
            urgency_score=0.20, triage_level="Routine", triage_reason="No immediate high-risk phrases detected"
        )

        history = db_service.get_user_history(user[0])
        self.assertEqual(len(history), 2)
        self.assertEqual(db_service.get_latest_emotion(user[0]), "Positive")
        self.assertEqual(history[0][6], "Low")
        self.assertEqual(history[0][7], "Text + Face")
        self.assertEqual(history[0][9], "Routine")

        recent = db_service.get_recent_predictions(user[0], limit=1)
        self.assertEqual(recent[0][6], "Routine")


class ServiceTests(unittest.TestCase):
    def test_combine_predictions_without_face(self):
        result = fusion_engine.combine_predictions(
            {"label": "Stress", "raw_label": "Stress", "confidence": 0.88, "is_uncertain": False},
            None
        )
        self.assertEqual(result["mental_state"], "Stress")
        self.assertEqual(result["face_emotion"], "Not Provided")
        self.assertEqual(result["support_level"], "Moderate")
        self.assertEqual(result["input_source"], "Text Only")

    def test_combine_predictions_marks_uncertain_when_text_is_low_confidence(self):
        result = fusion_engine.combine_predictions(
            {"label": "Uncertain", "raw_label": "Positive", "confidence": 0.31, "is_uncertain": True},
            {"label": "Happy", "raw_label": "Happy", "confidence": 0.82, "is_uncertain": False}
        )
        self.assertEqual(result["mental_state"], "Uncertain")
        self.assertEqual(result["support_level"], "Review")

    def test_safety_guard_detects_high_risk_language(self):
        result = safety_guard.assess_text_risk("I want to die and hurt myself")
        self.assertEqual(result["triage_level"], "High Alert")
        self.assertGreater(result["urgency_score"], 0.9)

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
        self.assertEqual(len(result["top_predictions"]), 3)

    @mock.patch("services.text_prediction._load_assets")
    def test_text_prediction_returns_uncertain_below_threshold(self, mock_load_assets):
        class DummyVectorizer:
            def transform(self, texts):
                class DummyArray:
                    def toarray(self_inner):
                        return [[0.1, 0.2, 0.3]]
                return DummyArray()

        class DummyModel:
            def predict(self, features, verbose=0):
                return [[0.34, 0.33, 0.33]]

        class DummyEncoder:
            classes_ = ["Anxiety", "Positive", "Stress"]

            def inverse_transform(self, values):
                return [self.classes_[values[0]]]

        mock_load_assets.return_value = (DummyModel(), DummyVectorizer(), DummyEncoder())
        result = text_prediction.predict_text_mental_state("I feel okay")

        self.assertEqual(result["label"], "Uncertain")
        self.assertTrue(result["is_uncertain"])


if __name__ == "__main__":
    unittest.main()
