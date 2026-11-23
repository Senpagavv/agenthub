import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

CLI = Path(__file__).resolve().parents[1] / "hub.py"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class EvaluationTests(unittest.TestCase):
    def evaluate(self, predictions, reviews=()):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "tasks/first"
            first.mkdir(parents=True)
            write_jsonl(first / "results.jsonl", predictions)
            gold = root / "gold.jsonl"
            write_jsonl(gold, [{"id": "a", "label": True}, {"id": "b", "label": False}])
            cmd = [
                sys.executable,
                str(CLI),
                "eval",
                "--task",
                "first",
                "--gold",
                str(gold),
                "--field",
                "label",
            ]
            if reviews:
                second = root / "tasks/review"
                second.mkdir()
                write_jsonl(second / "jev_results.jsonl", reviews)
                cmd += ["--fallback", "review"]
            subprocess.run(
                cmd,
                env=dict(os.environ, AGENTHUB_WORKSPACE=tmp),
                capture_output=True,
                check=True,
            )
            return json.loads((root / "logs/evals.jsonl").read_text(encoding="utf-8"))

    def test_missing_prediction_stays_in_denominator(self):
        record = self.evaluate(
            [{"id": "a", "status": "ok", "output": {"label": True}, "cost_cny": 0.02}]
        )
        self.assertEqual(
            (record["n"], record["accuracy"], record["cost_per_item"]), (2, 0.5, 0.01)
        )

    def test_malformed_review_is_unscored_and_unknown_charge_is_preserved(self):
        record = self.evaluate(
            [{"id": "a", "status": "ok", "output": {"label": True}, "cost_cny": 0.02}],
            [{"id": "b", "status": "ok", "answers": {"label": {"type": "noul"}}}],
        )
        self.assertEqual(record["accuracy"], 0.5)
        self.assertIsNone(record["cost_per_item"])


if __name__ == "__main__":
    unittest.main()
