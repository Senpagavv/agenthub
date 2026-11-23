"""Run the actual AgentHub evaluator on four illustrative records, offline."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import shutil
import subprocess
import sys

EXAMPLES = Path(__file__).resolve().parent
CLI = EXAMPLES.parent / "hub.py"


def main():
    with TemporaryDirectory(prefix="agenthub-demo-") as temp:
        root = Path(temp)
        for name, source, destination in (
            ("first", "first-pass.jsonl", "results.jsonl"),
            ("review", "review.jsonl", "jev_results.jsonl"),
        ):
            folder = root / "tasks" / name
            folder.mkdir(parents=True)
            shutil.copyfile(EXAMPLES / source, folder / destination)
            (folder / "run_config.json").write_text(json.dumps({"executor": "example", "model": name}), encoding="utf-8")
        env = dict(os.environ, AGENTHUB_WORKSPACE=str(root), PYTHONIOENCODING="utf-8")
        args = [sys.executable, str(CLI), "eval", "--task", "first", "--gold", str(EXAMPLES / "gold.jsonl"), "--field", "label"]
        for title, extra in (("First pass", []), ("First pass + one review", ["--fallback", "review"])):
            subprocess.run(args + extra, env=env, check=True, stdout=subprocess.DEVNULL)
            rows = (root / "logs/evals.jsonl").read_text(encoding="utf-8").splitlines()
            result = json.loads(rows[-1])
            correct = round(result["accuracy"] * result["n"])
            print(f"{title}: {correct}/{result['n']} correct, CNY {result['cost_per_item']:.5f} per input")


if __name__ == "__main__":
    main()
