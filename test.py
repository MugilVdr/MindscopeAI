import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

from database import db_service
from services import face_prediction, fusion_engine, pdf_report, text_prediction, safety_guard


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
            text_raw_label TEXT,
            face_raw_label TEXT,
            text_top_predictions TEXT,
            face_top_predictions TEXT,
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
            urgency_score=0.72, triage_level="Needs Attention", triage_reason="hopeless",
            text_raw_label="Stress", face_raw_label="Sad",
            text_top_predictions=[{"label": "Stress", "score": 0.84}],
            face_top_predictions=[{"label": "Sad", "score": 0.67}]
        )
        db_service.save_prediction(
            user[0], "sample text 2", "Happy", "Positive", "Doing well",
            text_confidence=0.91, face_confidence=0.76, support_level="Low", input_source="Text + Face",
            urgency_score=0.20, triage_level="Routine", triage_reason="No immediate high-risk phrases detected",
            text_raw_label="Positive", face_raw_label="Happy",
            text_top_predictions=[{"label": "Positive", "score": 0.91}],
            face_top_predictions=[{"label": "Happy", "score": 0.76}]
        )

        history = db_service.get_user_history(user[0])
        self.assertEqual(len(history), 2)
        self.assertEqual(db_service.get_latest_emotion(user[0]), "Positive")
        self.assertEqual(history[0][6], "Low")
        self.assertEqual(history[0][7], "Text + Face")
        self.assertEqual(history[0][9], "Routine")

        recent = db_service.get_recent_predictions(user[0], limit=1)
        self.assertEqual(recent[0][6], "Routine")
        self.assertEqual(recent[0][7], "Positive")

    def test_build_weekly_summary(self):
        db_service.register_user("Test User", "tester", "t@example.com", "secret123")
        user = db_service.login_user("tester", "secret123")

        db_service.save_prediction(
            user[0], "sample 1", "Happy", "Positive", "Doing well",
            text_confidence=0.91, face_confidence=0.80, support_level="Low", input_source="Text + Face",
            urgency_score=0.20, triage_level="Routine", triage_reason="none",
            text_raw_label="Positive", face_raw_label="Happy",
            text_top_predictions=[{"label": "Positive", "score": 0.91}],
            face_top_predictions=[{"label": "Happy", "score": 0.80}]
        )
        db_service.save_prediction(
            user[0], "sample 2", "Uncertain", "Uncertain", "Mixed signals",
            text_confidence=0.42, face_confidence=0.30, support_level="Review", input_source="Text + Face",
            urgency_score=0.35, triage_level="Routine", triage_reason="none",
            text_raw_label="Stress", face_raw_label="Neutral",
            text_top_predictions=[{"label": "Stress", "score": 0.42}],
            face_top_predictions=[{"label": "Neutral", "score": 0.30}]
        )

        summary = db_service.build_weekly_summary(user[0], days=7)
        self.assertEqual(summary["total_checkins"], 2)
        self.assertIn(summary["trend_direction"], {"Improving", "Worsening", "Mixed"})
        self.assertEqual(summary["uncertain_sessions"], 1)
        self.assertEqual(summary["raw_label_mismatch_count"], 2)
        self.assertEqual(len(summary["change_events"]), 1)
        self.assertIn(summary["change_events"][0]["movement"], {"Improving", "Worsening", "Stable"})

    def test_change_event_detection(self):
        rows = [
            ("2026-04-01 10:00:00", "Positive", 0.90, "Happy", 0.80, "Low", 0.20, "Routine", "Text + Face", "Positive", "Happy", "[]", "[]"),
            ("2026-04-02 10:00:00", "Stress", 0.60, "Sad", 0.70, "Moderate", 0.50, "Needs Attention", "Text + Face", "Stress", "Sad", "[]", "[]"),
            ("2026-04-03 10:00:00", "Uncertain", 0.35, "Uncertain", 0.30, "Review", 0.55, "Routine", "Text + Face", "Stress", "Neutral", "[]", "[]"),
        ]
        events = db_service.build_change_events_from_rows(rows)
        self.assertEqual(len(events), 2)
        self.assertTrue(any("Urgency spiked" in trigger for trigger in events[0]["triggers"]))
        self.assertTrue(any("Session became uncertain" in trigger for trigger in events[1]["triggers"]))


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

    @mock.patch("services.face_prediction._prepare_image")
    @mock.patch("services.face_prediction._load_model")
    def test_face_prediction_returns_uncertain_when_confidence_is_low(self, mock_load_model, mock_prepare_image):
        class DummyModel:
            def predict(self, features, verbose=0):
                return [[0.22, 0.08, 0.10, 0.18, 0.17, 0.14, 0.11]]

        mock_load_model.return_value = DummyModel()
        mock_prepare_image.return_value = (np.zeros((1, 64, 64, 3)), False, "No clear face detected. Using center crop fallback.")

        result = face_prediction.predict_face_emotion(b"fake-image")
        self.assertEqual(result["label"], "Uncertain")
        self.assertFalse(result["face_detected"])

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

    def test_pdf_report_line_helpers(self):
        weekly_summary = {
            "total_checkins": 3,
            "dominant_state": "Stress",
            "trend_direction": "Mixed",
            "average_text_confidence": 0.61,
            "average_urgency": 0.42,
            "uncertain_sessions": 1,
            "highest_urgency_state": "Anxiety",
            "highest_urgency_score": 0.75,
            "agreement_count": 1,
            "disagreement_count": 2,
            "raw_label_mismatch_count": 2,
        }
        diagnostics = [
            (
                "2026-04-05 10:00:00", "Stress", "Uncertain", 0.61, 0.32,
                "Stress", "Neutral",
                '[{"label":"Stress","score":0.61}]',
                '[{"label":"Neutral","score":0.32}]'
            )
        ]
        changes = [
            {
                "date": "2026-04-05 11:00:00",
                "from_state": "Positive",
                "to_state": "Stress",
                "movement": "Worsening",
                "triggers": ["Urgency spiked"],
                "urgency_delta": 0.30,
                "text_conf_delta": -0.20,
            }
        ]

        summary_lines = pdf_report._line_items_from_summary(weekly_summary)
        diagnostics_lines = pdf_report._line_items_from_diagnostics(diagnostics)
        change_lines = pdf_report._line_items_from_changes(changes)
        self.assertTrue(any("Dominant state" in line for line in summary_lines))
        self.assertTrue(any("Text top 3" in line for line in diagnostics_lines))
        self.assertTrue(any("Worsening" in line for line in change_lines))


if __name__ == "__main__":
    unittest.main()
