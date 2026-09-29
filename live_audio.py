"""Bounded microphone streaming and simple RMS-based speech segmentation."""

from array import array
from collections import deque
from dataclasses import dataclass
import math
from threading import Condition, Lock


SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms
MAX_WINDOW_SECONDS = 8


class AudioError(Exception):
    pass


@dataclass(frozen=True)
class SpeechChunk:
    sequence: int
    pcm: bytes
    final: bool


class SpeechWindows:
    """Emit growing partials during speech and close a window after 0.5 s silence."""

    def __init__(self, interval=1.0, threshold=300):
        self.interval_bytes = int(interval * SAMPLE_RATE) * 2
        self.threshold = threshold
        self.preroll = deque(maxlen=10)  # 200 ms avoids clipping speech onset.
        self.sequence = 0
        self.audio = bytearray()
        self.last_emitted = 0
        self.silent_samples = 0
        self.voiced_samples = 0

    def finish(self):
        """Flush speech immediately on key release and discard idle pre-roll."""
        chunk = None
        if self.audio and self.voiced_samples >= 3 * FRAME_SAMPLES:
            chunk = SpeechChunk(self.sequence, bytes(self.audio), True)
        self.audio.clear()
        self.preroll.clear()
        self.last_emitted = self.silent_samples = self.voiced_samples = 0
        return chunk

    def feed(self, pcm):
        samples = array("h", pcm)
        if not samples:
            return None
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        voiced = rms >= self.threshold
        if not self.audio:
            self.preroll.append(pcm)
            if not voiced:
                return None
            self.sequence += 1
            self.audio.extend(b"".join(self.preroll))
            self.preroll.clear()
        else:
            self.audio.extend(pcm)
        self.silent_samples = 0 if voiced else self.silent_samples + len(samples)
        if voiced:
            self.voiced_samples += len(samples)
        final = (self.silent_samples >= SAMPLE_RATE // 2 or
                 len(self.audio) >= MAX_WINDOW_SECONDS * SAMPLE_RATE * 2)
        chunk = None
        if final or len(self.audio) - self.last_emitted >= self.interval_bytes:
            # Ignore isolated clicks shorter than 60 ms.
            if self.voiced_samples >= 3 * FRAME_SAMPLES:
                chunk = SpeechChunk(self.sequence, bytes(self.audio), final)
            self.last_emitted = len(self.audio)
        if final:
            self.audio.clear()
            self.last_emitted = self.silent_samples = self.voiced_samples = 0
        return chunk


class PendingChunks:
    """Keep two speech windows, replacing superseded partials of the same window."""

    def __init__(self):
        self.items = deque()
        self.condition = Condition()
        self.dropped = 0

    def put(self, chunk):
        with self.condition:
            if self.items and self.items[-1].sequence == chunk.sequence:
                self.items[-1] = chunk
            else:
                if len(self.items) == 2:
                    self.items.popleft()
                    self.dropped += 1
                self.items.append(chunk)
            self.condition.notify()

    def take(self, timeout=0.1):
        with self.condition:
            if not self.items:
                self.condition.wait(timeout)
            if not self.items:
                return None, 0
            dropped, self.dropped = self.dropped, 0
            return self.items.popleft(), dropped


class LiveMicrophone:
    """PortAudio captures in its callback thread while the caller runs inference."""

    def __init__(self, interval, threshold, *, recording=True):
        import sounddevice

        self.sd = sounddevice
        self.windows = SpeechWindows(interval, threshold)
        self.pending = PendingChunks()
        self.error = None
        self.stream = None
        self.recording = recording
        self.state_lock = Lock()

    def set_recording(self, recording):
        with self.state_lock:
            if self.recording and not recording:
                chunk = self.windows.finish()
                if chunk is not None:
                    self.pending.put(chunk)
            self.recording = recording

    def callback(self, data, frames, timing, status):
        try:
            if status:
                raise AudioError(f"Microphone stream error: {status}")
            with self.state_lock:
                if self.recording:
                    chunk = self.windows.feed(bytes(data))
                    if chunk is not None:
                        self.pending.put(chunk)
        except Exception as exc:
            self.error = exc
            raise self.sd.CallbackAbort from exc

    def __enter__(self):
        try:
            self.stream = self.sd.RawInputStream(
                device=None, samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, callback=self.callback,
            )
            self.stream.start()
        except BaseException as exc:
            if self.stream is not None:
                self.stream.close()
            if isinstance(exc, self.sd.PortAudioError):
                raise AudioError(
                    f"Cannot open the default microphone: {exc}. "
                    "Check Linux sound settings; 16 kHz mono input is required."
                ) from exc
            raise
        return self

    def __exit__(self, *exc):
        try:
            self.stream.abort()
        finally:
            self.stream.close()

    def next_chunk(self):
        if self.error:
            raise AudioError(str(self.error)) from self.error
        if not self.stream.active:
            raise AudioError("The default microphone stopped unexpectedly.")
        return self.pending.take()
