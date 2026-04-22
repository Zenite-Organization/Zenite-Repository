import unittest

from scripts.train_meta_calibrator import _dedupe_latest_issue_rows


class TestTrainMetaCalibrator(unittest.TestCase):
    def test_dedupe_latest_issue_rows_keeps_most_recent_validation(self):
        rows = [
            {
                "issue_id": 101,
                "validation_run_id": "run-a",
                "predicted_at": "2026-04-14 19:30:00",
                "base_hours": 8.0,
            },
            {
                "issue_id": 101,
                "validation_run_id": "run-b",
                "predicted_at": "2026-04-14 20:30:00",
                "base_hours": 9.0,
            },
            {
                "issue_id": 202,
                "validation_run_id": "run-a",
                "predicted_at": "2026-04-14 19:35:00",
                "base_hours": 5.0,
            },
        ]

        deduped = _dedupe_latest_issue_rows(rows)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["issue_id"], 202)
        self.assertEqual(deduped[1]["issue_id"], 101)
        self.assertEqual(deduped[1]["validation_run_id"], "run-b")
        self.assertEqual(deduped[1]["base_hours"], 9.0)


if __name__ == "__main__":
    unittest.main()
