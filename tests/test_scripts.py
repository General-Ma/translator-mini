import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="translator scripts ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project with spaces"
        self.project.mkdir()
        repo = Path(__file__).resolve().parents[1]
        for script in ("setup.sh", "run.sh"):
            shutil.copy2(repo / script, self.project / script)
        self.env = dict(os.environ, PATH="/usr/bin:/bin")

    def executable(self, path, code):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!{sys.executable}\n{code}\n")
        path.chmod(0o755)

    def fake_python(self):
        self.executable(self.project / ".venv/bin/python",
                        "import json, os, sys; print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}))")

    def test_run_preserves_arguments_and_cwd_and_finds_local_whisper(self):
        self.fake_python()
        whisper = self.project / ".tools/whisper.cpp/build/bin/whisper-cli"
        self.executable(whisper, "pass")
        result = subprocess.run([str(self.project / "run.sh"), "--model", "my models/tiny.bin"],
                                cwd=self.root, env=self.env, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["cwd"], str(self.root))
        self.assertEqual(payload["args"], [str(self.project / "translator.py"), "--whisper-cli",
                                           str(whisper), "--model", "my models/tiny.bin"])

    def test_run_respects_explicit_executable(self):
        self.fake_python()
        self.executable(self.project / ".tools/whisper.cpp/build/bin/whisper-cli", "pass")
        result = subprocess.run([str(self.project / "run.sh"), "--whisper-cli=/some other/bin", "--help"],
                                env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["args"],
                         [str(self.project / "translator.py"), "--whisper-cli=/some other/bin", "--help"])

    def test_run_without_setup_has_actionable_error(self):
        result = subprocess.run([str(self.project / "run.sh")], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("setup.sh first", result.stderr)

    def test_setup_stops_when_python_dependency_install_fails(self):
        # --skip-system must not invoke a package manager; errors must not report success.
        self.executable(self.project / ".venv/bin/python", "import sys; sys.exit(7)")
        result = subprocess.run([str(self.project / "setup.sh"), "--skip-system"],
                                env=dict(self.env, PYTHON=sys.executable), capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("Setup complete", result.stdout)
        self.assertFalse((self.project / ".tools").exists())


if __name__ == "__main__":
    unittest.main()
