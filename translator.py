#!/usr/bin/env python3
"""Offline English/Chinese microphone translator for Linux."""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

from live_audio import AudioError, LiveMicrophone, SAMPLE_RATE
from keyboard_input import KeyboardError, PushToTalk


LANGUAGES = {"en": "English", "zh": "Chinese"}
MODELS_DIR = Path(__file__).resolve().parent / "models"


class AppError(Exception):
    """An actionable error to show without a traceback."""


@dataclass(frozen=True)
class TranslationModel:
    path: Path
    source: str
    target: str
    target_prefix: str = ""


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AppError(f"Cannot read JSON from {path}: {exc}") from exc


def translation_model(path, source, target):
    """Validate an already extracted Argos SentencePiece package, without installing it."""
    path = path.expanduser().resolve()
    for name in ("metadata.json", "sentencepiece.model", "model/model.bin", "model/config.json"):
        if not (path / name).is_file():
            raise AppError(f"Missing {path / name}. Restore the project's bundled translation models.")
    metadata = read_json(path / "metadata.json")
    if not isinstance(metadata, dict) or (
        metadata.get("from_code"), metadata.get("to_code")
    ) != (source, target):
        raise AppError(f"{path}: expected an {source} → {target} translation model.")
    prefix = metadata.get("target_prefix", "")
    if not isinstance(prefix, str):
        raise AppError(f"{path}: target_prefix must be a string.")
    return TranslationModel(path, source, target, prefix)


def dependencies():
    for name in ("sounddevice", "ctranslate2", "sentencepiece"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as exc:
            raise AppError(
                f"Cannot load {name}: {exc}. Install requirements.txt; "
                "sounddevice also needs the system PortAudio library."
            ) from exc


def write_audio(path, pcm):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm)


def transcribe(executable, model, audio, threads):
    output = audio.parent / "transcript"
    command = [
        executable, "--model", str(model), "--file", str(audio),
        "--language", "auto", "--output-json", "--output-file", str(output),
        "--threads", str(threads), "--no-gpu", "--beam-size", "1", "--best-of", "1",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300, check=False, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise AppError("whisper.cpp exceeded 5 minutes. Try a smaller multilingual model.") from exc
    except OSError as exc:
        raise AppError(f"Cannot run whisper.cpp: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise AppError(f"whisper.cpp failed (exit {result.returncode}): {detail}")
    document = read_json(output.with_suffix(".json"))
    try:
        if document["model"]["multilingual"] is not True:
            raise AppError("Use a multilingual Whisper model; English-only (.en) models cannot detect Chinese.")
        language = document["result"]["language"]
        segments = document["transcription"]
        if not isinstance(language, str) or not isinstance(segments, list):
            raise TypeError("invalid language or transcription")
        text = "".join(segment["text"] for segment in segments).strip()
    except (KeyError, TypeError) as exc:
        raise AppError("Unexpected whisper.cpp JSON. Use a recent whisper-cli with --output-json support.") from exc
    return language, text


def translate(text, model, threads):
    """Run one local Argos model directly, avoiding Argos sentence-splitter dependencies."""
    import ctranslate2
    import sentencepiece

    engine = None
    try:
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(model.path / "sentencepiece.model")
        )
        tokens = tokenizer.encode(text, out_type=str)
        if len(tokens) > 256:
            raise AppError("Phrase is too long for this MVP. Record a shorter sentence.")
        if not tokens:
            raise AppError("No translatable text was recognized.")
        engine = ctranslate2.Translator(
            str(model.path / "model"), device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=threads,
        )
        result = engine.translate_batch(
            [tokens], target_prefix=[[model.target_prefix]] if model.target_prefix else None,
            beam_size=1, replace_unknowns=True, max_input_length=256,
            max_decoding_length=256,
        )[0]
        translated_tokens = result.hypotheses[0]
        if len(translated_tokens) >= 256:
            raise AppError("Translation reached its length limit. Record a shorter sentence.")
        if model.target_prefix and translated_tokens[:1] == [model.target_prefix]:
            translated_tokens = translated_tokens[1:]
        # Argos vocabularies can leave SentencePiece word-boundary markers in
        # decoded tokens, particularly when translating Chinese into English.
        translated = tokenizer.decode(translated_tokens).replace("▁", " ").strip()
        if not translated:
            raise AppError("The translation model returned no text. Try recording again.")
        return translated
    except AppError:
        raise
    except Exception as exc:
        raise AppError(f"Cannot translate using {model.path}: {exc}") from exc
    finally:
        if engine is not None:
            engine.unload_model()


def run(args, models):
    previous = None
    with LiveMicrophone(args.interval, args.speech_threshold, recording=args.continuous) as microphone, \
         (nullcontext() if args.continuous else PushToTalk(microphone, args.keyboard_device)) as controls:
        if args.continuous:
            print("Listening continuously. Speak English or Chinese; Ctrl+C to quit.", flush=True)
        else:
            print("Hold Space or Enter to talk; release to stop. Ctrl+C to quit.", flush=True)
        print("Partial translations update during speech and may change as more words arrive.", flush=True)
        while True:
            if controls is not None:
                controls.check()
            chunk, dropped = microphone.next_chunk()
            if chunk is None:
                continue
            if dropped:
                print(
                    f"Warning: inference fell behind; skipped {dropped} older speech window(s). "
                    "Try a smaller model or pause between sentences.", file=sys.stderr, flush=True,
                )
            with tempfile.TemporaryDirectory(prefix="translator-mini-") as temporary:
                audio = Path(temporary) / "microphone.wav"
                write_audio(audio, chunk.pcm)
                language, text = transcribe(args.whisper_cli, args.model, audio, args.threads)
            if not text:
                continue
            if language not in models:
                if chunk.final:
                    print(f"Detected {language}; only English and Chinese are supported.", flush=True)
                continue
            model = models[language]
            key = (chunk.sequence, language, text)
            if previous is not None and previous[0] == key:
                if not chunk.final:
                    continue
                translated = previous[1]
            else:
                translated = translate(text, model, args.threads)
                previous = (key, translated)
            state = "final" if chunk.final else "partial"
            print(f"[{chunk.sequence} {state}] {LANGUAGES[language]}: {text}", flush=True)
            print(f"[{chunk.sequence} {state}] {LANGUAGES[model.target]}: {translated}", flush=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local multilingual whisper.cpp .bin model")
    parser.add_argument("--whisper-cli", default="whisper-cli", help="whisper-cli executable path (default: PATH lookup)")
    parser.add_argument("--interval", type=float, default=1.0, help="partial update interval, 0.5–4 seconds (default: 1)")
    parser.add_argument("--speech-threshold", type=positive_int, default=300,
                        help="microphone RMS speech threshold, 1–32767 (default: 300)")
    parser.add_argument("--threads", type=positive_int, default=2, help="CPU threads per model (default: 2)")
    parser.add_argument("--continuous", action="store_true", help="listen continuously instead of holding Space/Enter")
    parser.add_argument("--keyboard-device", help="Linux keyboard event device for push-to-talk (default: auto-detect)")
    args = parser.parse_args(argv)
    if not 0.5 <= args.interval <= 4:
        parser.error("--interval must be between 0.5 and 4")
    if args.speech_threshold > 32767:
        parser.error("--speech-threshold must be between 1 and 32767")
    try:
        args.model = args.model.expanduser().resolve()
        if not args.model.is_file():
            raise AppError(f"Whisper model does not exist: {args.model}")
        executable = shutil.which(os.path.expanduser(args.whisper_cli))
        if executable is None:
            raise AppError("whisper-cli not found. Supply --whisper-cli /path/to/whisper-cli.")
        args.whisper_cli = str(Path(executable).resolve())
        models = {
            "en": translation_model(MODELS_DIR / "en-zh", "en", "zh"),
            "zh": translation_model(MODELS_DIR / "zh-en", "zh", "en"),
        }
        dependencies()
        run(args, models)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except (AppError, AudioError, KeyboardError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
