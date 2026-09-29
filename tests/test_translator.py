import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

import translator as app
from live_audio import SpeechChunk


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="translator test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.whisper_model = self.root / "ggml-base.bin"
        self.whisper_model.touch()

    def package(self, source="en", target="zh", prefix=""):
        root = self.root / f"{source}-{target}"
        (root / "model").mkdir(parents=True, exist_ok=True)
        for name in ("sentencepiece.model", "model/model.bin", "model/config.json"):
            (root / name).touch()
        (root / "metadata.json").write_text(json.dumps({
            "from_code": source, "to_code": target, "target_prefix": prefix,
        }))
        return app.translation_model(root, source, target)

    def whisper(self, document=None, exit_code=0):
        executable = self.root / "whisper-cli"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys\n"
            f"sys.exit({exit_code}) if {exit_code} else None\n"
            "args = sys.argv[1:]\n"
            "assert args[args.index('--language') + 1] == 'auto'\n"
            "assert '--translate' not in args and '--detect-language' not in args\n"
            "assert '--no-gpu' in args\n"
            "assert pathlib.Path(args[args.index('--model') + 1]).is_file()\n"
            "out = pathlib.Path(args[args.index('--output-file') + 1] + '.json')\n"
            f"out.write_text({json.dumps(document)!r}, encoding='utf-8')\n"
        )
        executable.chmod(0o755)
        return str(executable)

    def document(self, language="en", text=" Hello", multilingual=True):
        return {"model": {"multilingual": multilingual},
                "result": {"language": language}, "transcription": [{"text": text}]}

    def test_whisper_json_uses_detected_language_and_preserves_chinese(self):
        for language, text in (("en", "Hello"), ("zh", "你好世界")):
            with self.subTest(language=language):
                executable = self.whisper(self.document(language, text))
                self.assertEqual(app.transcribe(executable, self.whisper_model,
                                               self.root / "audio.wav", 2), (language, text))

    def test_rejects_english_only_model_and_malformed_output(self):
        for document, message in ((self.document(multilingual=False), "multilingual"),
                                  ({"result": {}}, "Unexpected")):
            with self.subTest(document=document), self.assertRaisesRegex(app.AppError, message):
                app.transcribe(self.whisper(document), self.whisper_model, self.root / "audio.wav", 2)

    def test_whisper_failure_and_timeout(self):
        with self.assertRaisesRegex(app.AppError, "exit 7"):
            app.transcribe(self.whisper(exit_code=7), self.whisper_model, self.root / "audio.wav", 2)
        with patch.object(app.subprocess, "run", side_effect=subprocess.TimeoutExpired("whisper", 300)):
            with self.assertRaisesRegex(app.AppError, "5 minutes"):
                app.transcribe("whisper", self.whisper_model, self.root / "audio.wav", 2)

    def test_package_validation_rejects_wrong_direction_and_missing_tokenizer(self):
        model = self.package()
        with self.assertRaisesRegex(app.AppError, "expected an zh"):
            app.translation_model(model.path, "zh", "en")
        (model.path / "sentencepiece.model").unlink()
        with self.assertRaisesRegex(app.AppError, "Missing"):
            app.translation_model(model.path, "en", "zh")

    def translation_modules(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = ["▁Hello"]
        tokenizer.decode.return_value = "你好"
        engine = MagicMock()
        engine.translate_batch.return_value = [SimpleNamespace(hypotheses=[["你好"]])]
        ct2 = SimpleNamespace(Translator=MagicMock(return_value=engine))
        sp = SimpleNamespace(SentencePieceProcessor=MagicMock(return_value=tokenizer))
        return {"ctranslate2": ct2, "sentencepiece": sp}, engine, tokenizer

    def test_local_translation_int8_and_model_release(self):
        model = self.package(prefix="<zh>")
        modules, engine, tokenizer = self.translation_modules()
        engine.translate_batch.return_value = [SimpleNamespace(hypotheses=[["<zh>", "你好"]])]
        with patch.dict(sys.modules, modules):
            self.assertEqual(app.translate("Hello", model, 2), "你好")
        modules["ctranslate2"].Translator.assert_called_once_with(
            str(model.path / "model"), device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)
        tokenizer.decode.assert_called_once_with(["你好"])
        engine.unload_model.assert_called_once()

    def test_translation_failure_still_releases_model(self):
        modules, engine, _ = self.translation_modules()
        engine.translate_batch.side_effect = RuntimeError("bad model")
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(app.AppError, "bad model"):
            app.translate("Hello", self.package(), 2)
        engine.unload_model.assert_called_once()

    def test_translation_cleans_argos_word_boundaries(self):
        modules, _, tokenizer = self.translation_modules()
        tokenizer.decode.return_value = "▁Where's the▁train▁station?"
        with patch.dict(sys.modules, modules):
            self.assertEqual(app.translate("火车站在哪里？", self.package("zh", "en"), 2),
                             "Where's the train station?")

    def test_long_translation_input_is_not_silently_truncated(self):
        modules, _, tokenizer = self.translation_modules()
        tokenizer.encode.return_value = ["x"] * 257
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(app.AppError, "too long"):
            app.translate("Hello", self.package(), 2)
        modules["ctranslate2"].Translator.assert_not_called()

    def live_microphone(self, chunks):
        mic = MagicMock()
        mic.__enter__.return_value = mic
        mic.next_chunk.side_effect = chunks + [KeyboardInterrupt()]
        return mic

    def test_writes_mono_wav(self):
        audio = self.root / "record.wav"
        app.write_audio(audio, b"\x01\x00" * 16000)
        with wave.open(str(audio)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()),
                             (1, 2, 16000, 16000))

    def test_live_session_routes_both_languages_and_removes_recordings(self):
        models = {"en": self.package(), "zh": self.package("zh", "en")}
        args = SimpleNamespace(interval=1, speech_threshold=300, continuous=True, keyboard_device=None, whisper_cli="whisper",
                               model=self.whisper_model, threads=2)
        chunks = [(SpeechChunk(i, b"\x01\x00" * 16000, True), 0) for i in range(4)]
        mic = self.live_microphone(chunks)
        results = iter([("en", "Hello"), ("zh", "你好"), ("fr", "Bonjour"), ("en", "")])
        paths = []

        def transcribe(executable, model, audio, threads):
            self.assertTrue(audio.exists())
            paths.append(audio)
            return next(results)

        with patch.object(app, "LiveMicrophone", return_value=mic), \
             patch.object(app, "transcribe", side_effect=transcribe), \
             patch.object(app, "translate", side_effect=["你好", "Hello"]) as translate, \
             contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(KeyboardInterrupt):
            app.run(args, models)
        self.assertEqual([call.args[1].source for call in translate.call_args_list], ["en", "zh"])
        self.assertIn("Chinese: 你好", output.getvalue())
        self.assertIn("English: Hello", output.getvalue())
        self.assertIn("Detected fr", output.getvalue())
        self.assertTrue(all(not path.parent.exists() for path in paths))
        mic.__exit__.assert_called_once()

    def test_partial_updates_before_final_and_reuses_identical_translation(self):
        args = SimpleNamespace(interval=1, speech_threshold=300, continuous=True, keyboard_device=None, whisper_cli="whisper",
                               model=self.whisper_model, threads=2)
        mic = self.live_microphone([(SpeechChunk(1, b"\0\0", False), 0),
                                    (SpeechChunk(1, b"\0\0", True), 2)])
        with patch.object(app, "LiveMicrophone", return_value=mic), \
             patch.object(app, "transcribe", return_value=("en", "Hello")), \
             patch.object(app, "translate", return_value="你好") as translate, \
             contextlib.redirect_stdout(io.StringIO()) as output, \
             contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(KeyboardInterrupt):
            app.run(args, {"en": self.package()})
        self.assertIn("[1 partial] Chinese: 你好", output.getvalue())
        self.assertIn("[1 final] Chinese: 你好", output.getvalue())
        self.assertIn("skipped 2", errors.getvalue())
        translate.assert_called_once()

    def test_recording_cleanup_on_error_and_interrupt(self):
        args = SimpleNamespace(interval=1, speech_threshold=300, continuous=True, keyboard_device=None, whisper_cli="whisper",
                               model=self.whisper_model, threads=2)
        paths = []
        for failure in (app.AppError("inference error"), KeyboardInterrupt()):
            mic = self.live_microphone([(SpeechChunk(1, b"\0\0", False), 0)])

            def transcribe(executable, model, audio, threads):
                paths.append(audio)
                raise failure

            with patch.object(app, "LiveMicrophone", return_value=mic), \
                 patch.object(app, "transcribe", side_effect=transcribe), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(type(failure)):
                app.run(args, {})
            mic.__exit__.assert_called_once()
        self.assertTrue(all(not path.parent.exists() for path in paths))

    def test_main_validates_paths_before_loading_dependencies(self):
        with patch.object(app, "dependencies") as deps, contextlib.redirect_stderr(io.StringIO()):
            code = app.main(["--model", str(self.root / "missing.bin")])
        self.assertEqual(code, 1)
        deps.assert_not_called()

    def test_complete_cli_with_local_fixtures(self):
        self.package()
        self.package("zh", "en")
        elsewhere = self.root / "another-directory"
        elsewhere.mkdir()
        executable = self.whisper(self.document())
        modules, engine, _ = self.translation_modules()
        modules["sounddevice"] = SimpleNamespace()
        mic = self.live_microphone([(SpeechChunk(1, b"\x01\x00" * 16000, False), 0)])
        with patch.dict(sys.modules, modules), patch.object(app, "LiveMicrophone", return_value=mic), \
             patch.object(app, "MODELS_DIR", self.root), contextlib.chdir(elsewhere), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = app.main([
                "--model", str(self.whisper_model), "--whisper-cli", executable,
                "--interval", "1", "--continuous",
            ])
        self.assertEqual(code, 130)
        self.assertIn("Chinese: 你好", output.getvalue())
        engine.unload_model.assert_called_once()

    def test_default_cli_starts_muted_with_push_to_talk(self):
        self.package()
        self.package("zh", "en")
        executable = self.whisper(self.document())
        mic = self.live_microphone([])
        controls = MagicMock()
        with patch.object(app, "MODELS_DIR", self.root), patch.object(app, "dependencies"), \
             patch.object(app, "LiveMicrophone", return_value=mic) as live_mic, \
             patch.object(app, "PushToTalk", return_value=controls) as push_to_talk, \
             contextlib.redirect_stdout(io.StringIO()):
            code = app.main(["--model", str(self.whisper_model), "--whisper-cli", executable])
        self.assertEqual(code, 130)
        live_mic.assert_called_once_with(1.0, 300, recording=False)
        push_to_talk.assert_called_once_with(mic, None)
        controls.__enter__.return_value.check.assert_called_once()
        controls.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
