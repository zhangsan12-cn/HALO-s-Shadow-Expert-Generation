from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shadow_planner.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_cli_defaults_ignore_training_environment(self):
        config = json.loads((ROOT / "examples" / "sample_counts.json").read_text(encoding="utf-8"))
        config["time_model"]["row_bytes"] = None
        config.pop("shadow_cost")
        config.pop("numa")
        config.pop("planner")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            stream = io.StringIO()
            with patch.dict(os.environ, {
                "MOE_SHADOW_ROW_BYTES": "invalid",
                "MOE_SHADOW_COUNT_COST_ALPHA_MS": "invalid",
                "MOE_SHADOW_NUMA_RANK_DOMAINS": "invalid",
            }), redirect_stdout(stream):
                main([str(path), "--model", "numa"])
            result = json.loads(stream.getvalue())
            self.assertEqual(result["metrics"]["count_cost_params"]["attention_ms"], 7.636)
            self.assertEqual(result["metrics"]["tau_abs"], 0.5)

    def test_both_models_produce_json_without_training_dependencies(self):
        for model in ("cost", "numa"):
            result = subprocess.run(
                [sys.executable, "-m", "shadow_planner", "examples/sample_counts.json", "--model", model],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            output = json.loads(result.stdout)
            self.assertEqual(output["model"], model)
            self.assertEqual(len(output["shadow_ids"]), 4)
            self.assertEqual(output["metrics"]["planner_variant"], model)

    def test_output_file_and_slot_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            subprocess.run(
                [sys.executable, "-m", "shadow_planner", "examples/sample_counts.json",
                 "--slots", "0", "--output", str(path)],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            output = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(output["shadow_ids"], [[], [], [], []])
            self.assertEqual(output["metrics"]["additional_cost_final"], 0)

    def test_bad_input_returns_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "shadow_planner", str(path)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
