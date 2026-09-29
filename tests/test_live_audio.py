from array import array
import contextlib
import io
from pathlib import Path
import sys
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import live_audio as live
import translator as app


VOICE = array("h", [1000] * live.FRAME_SAMPLES).tobytes()
SILENCE = b"\0\0" * live.FRAME_SAMPLES


class LiveAudioTests(unittest.TestCase):
    def test_emits_partial_during_uninterrupted_speech_then_final_after_pause(self):
        windows = live.SpeechWindows(interval=1)
        chunks = [windows.feed(VOICE) for _ in range(100)]
        partials = [chunk for chunk in chunks if chunk]
        self.assertEqual(len(partials), 2)
        self.assertTrue(all(not chunk.final for chunk in partials))
        self.assertLess(len(partials[0].pcm), len(partials[1].pcm))
        finals = [windows.feed(SILENCE) for _ in range(25)]
        self.assertTrue(finals[-1].final)
        self.assertEqual(finals[-1].sequence, partials[0].sequence)
        self.assertFalse(windows.audio)

    def test_silence_and_click_do_not_trigger_inference(self):
        windows = live.SpeechWindows()
        for frame in [SILENCE] * 100 + [VOICE] + [SILENCE] * 100:
            self.assertIsNone(windows.feed(frame))

    def test_continuous_speech_has_bounded_windows(self):
        windows = live.SpeechWindows()
        chunks = []
        for _ in range(850):
            chunk = windows.feed(VOICE)
            if chunk:
                chunks.append(chunk)
            self.assertLessEqual(len(windows.audio), live.MAX_WINDOW_SECONDS * live.SAMPLE_RATE * 2)
        self.assertEqual(sum(chunk.final for chunk in chunks), 2)
        self.assertEqual(chunks[-1].sequence, 3)

    def test_pending_work_coalesces_partials_and_reports_dropped_windows(self):
        pending = live.PendingChunks()
        for size in range(1, 20):
            pending.put(live.SpeechChunk(1, b"x" * size, False))
        pending.put(live.SpeechChunk(1, b"final", True))
        self.assertEqual(len(pending.items), 1)
        self.assertTrue(pending.items[0].final)
        pending.put(live.SpeechChunk(2, b"second", True))
        pending.put(live.SpeechChunk(3, b"third", False))
        chunk, dropped = pending.take()
        self.assertEqual((chunk.sequence, dropped), (2, 1))
        self.assertEqual(len(pending.items), 1)

    def microphone_module(self):
        stream = MagicMock()
        stream.active = True
        sd = SimpleNamespace(RawInputStream=MagicMock(return_value=stream),
                             PortAudioError=RuntimeError, CallbackAbort=type("CallbackAbort", (Exception,), {}))
        return sd, stream

    def test_capture_continues_while_app_is_transcribing(self):
        sd, stream = self.microphone_module()
        inference_started, captured_during_inference = Event(), Event()
        stop = Event()
        failures = []

        def capture():
            try:
                callback = sd.RawInputStream.call_args.kwargs["callback"]
                for _ in range(50):
                    callback(VOICE, 320, None, None)
                if not inference_started.wait(2):
                    raise AssertionError("inference never started")
                for _ in range(50):
                    callback(VOICE, 320, None, None)
                captured_during_inference.set()
                stop.wait(2)
            except BaseException as exc:
                failures.append(exc)

        worker = Thread(target=capture, daemon=True)
        stream.start.side_effect = worker.start
        stream.abort.side_effect = stop.set

        def transcribe(*args):
            inference_started.set()
            self.assertTrue(captured_during_inference.wait(2), "capture stalled during inference")
            self.assertTrue(stream.active)
            return "en", "Hello"

        args = SimpleNamespace(interval=1, speech_threshold=300, continuous=True, keyboard_device=None, whisper_cli="whisper",
                               model=Path("model.bin"), threads=2)
        model = SimpleNamespace(target="zh")
        try:
            with patch.dict(sys.modules, {"sounddevice": sd}), \
                 patch.object(app, "transcribe", side_effect=transcribe), \
                 patch.object(app, "translate", side_effect=KeyboardInterrupt), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                app.run(args, {"en": model})
        finally:
            stop.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertIsNone(sd.RawInputStream.call_args.kwargs["device"])
        stream.close.assert_called_once()

    def test_callback_errors_reach_main_thread_and_close_microphone(self):
        sd, stream = self.microphone_module()
        with patch.dict(sys.modules, {"sounddevice": sd}):
            with self.assertRaisesRegex(live.AudioError, "overflow"):
                with live.LiveMicrophone(1, 300) as mic:
                    with self.assertRaises(sd.CallbackAbort):
                        mic.callback(VOICE, 320, None, "input overflow")
                    mic.next_chunk()
        stream.close.assert_called_once()

    def test_startup_failure_closes_stream(self):
        sd, stream = self.microphone_module()
        stream.start.side_effect = RuntimeError("no input")
        with patch.dict(sys.modules, {"sounddevice": sd}), self.assertRaisesRegex(live.AudioError, "default microphone"):
            with live.LiveMicrophone(1, 300):
                self.fail("must not enter")
        stream.close.assert_called_once()

    def test_hold_to_talk_discards_idle_audio_and_flushes_on_release(self):
        sd, stream = self.microphone_module()
        with patch.dict(sys.modules, {"sounddevice": sd}):
            with live.LiveMicrophone(1, 300, recording=False) as mic:
                for _ in range(60):
                    mic.callback(VOICE, 320, None, None)
                self.assertEqual(len(mic.pending.items), 0)
                self.assertFalse(mic.windows.preroll)
                mic.set_recording(True)
                for _ in range(10):
                    mic.callback(VOICE, 320, None, None)
                mic.set_recording(False)
                chunk, dropped = mic.next_chunk()
                self.assertTrue(chunk.final)
                self.assertEqual(len(chunk.pcm), len(VOICE) * 10)
                self.assertEqual(dropped, 0)
                for _ in range(60):
                    mic.callback(VOICE, 320, None, None)
                self.assertFalse(mic.pending.items)
                self.assertFalse(mic.windows.audio)


if __name__ == "__main__":
    unittest.main()
