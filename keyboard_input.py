"""Linux key-down/key-up push-to-talk without running the app as root."""

import fcntl
import os
from pathlib import Path
import select
import struct
import sys
import termios
from threading import Event, Thread
import tty


KEYS = {28, 57, 96}  # Enter, Space, keypad Enter
INPUT_EVENT = struct.Struct("@llHHi")


class KeyboardError(Exception):
    pass


def keyboard_devices():
    """Find keyboards through readable sysfs metadata, including inaccessible nodes."""
    word_bits = struct.calcsize("L") * 8
    for event in sorted(Path("/sys/class/input").glob("event*")):
        try:
            words = (event / "device/capabilities/key").read_text().split()
            bits = sum(int(word, 16) << (word_bits * i) for i, word in enumerate(reversed(words)))
            if bits & (1 << 57) and bits & (1 << 28):
                yield Path("/dev/input") / event.name, (event / "device/name").read_text().strip()
        except (OSError, ValueError):
            continue


class PushToTalk:
    def __init__(self, microphone, device=None):
        self.microphone = microphone
        self.device = device
        self.fds = []
        self.pressed = set()
        self.stopping = Event()
        self.thread = None
        self.error = None
        self.terminal = None
        self.terminal_settings = None

    def handle_event(self, fd, event_type, code, value):
        if event_type == 0 and code == 3:  # SYN_DROPPED: a release might have been lost.
            raise KeyboardError("Keyboard events were lost. Restart push-to-talk.")
        if event_type != 1 or code not in KEYS or value not in (0, 1):
            return  # Ignore other keys and auto-repeat (value 2).
        before = bool(self.pressed)
        if value:
            self.pressed.add((fd, code))
        else:
            self.pressed.discard((fd, code))
        after = bool(self.pressed)
        if before != after:
            self.microphone.set_recording(after)
            print("Recording…" if after else "Released. Finishing translation…", flush=True)

    def read_events(self):
        try:
            while not self.stopping.is_set():
                ready, _, _ = select.select([*self.fds, self.terminal], [], [], 0.05)
                for fd in ready:
                    if fd == self.terminal:
                        # Drain ordinary terminal characters so they cannot reach the shell later.
                        if not os.read(fd, 1024):
                            raise KeyboardError("Terminal input closed.")
                        continue
                    data = os.read(fd, INPUT_EVENT.size * 64)
                    if not data:
                        raise KeyboardError("Keyboard disconnected.")
                    for _, _, kind, code, value in INPUT_EVENT.iter_unpack(data):
                        self.handle_event(fd, kind, code, value)
        except Exception as exc:
            self.error = exc
        finally:
            self.microphone.set_recording(False)

    def __enter__(self):
        try:
            if not sys.stdin.isatty():
                raise KeyboardError("Push-to-talk needs an interactive local terminal.")
            devices = [Path(self.device).expanduser()] if self.device else [p for p, _ in keyboard_devices()]
            for path in devices:
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                except PermissionError:
                    if self.device:
                        raise KeyboardError(f"Cannot read {path}. See the README keyboard-access command.")
                    continue
                try:
                    keys = bytearray(96)
                    # Linux EVIOCGBIT(EV_KEY, sizeof(keys)).
                    fcntl.ioctl(fd, (2 << 30) | (len(keys) << 16) | (ord('E') << 8) | 0x21, keys)
                    if not (keys[57 // 8] & (1 << (57 % 8)) and keys[28 // 8] & (1 << (28 % 8))):
                        raise KeyboardError(f"{path} is not a keyboard with Space and Enter.")
                except BaseException:
                    os.close(fd)
                    raise
                self.fds.append(fd)
            if not self.fds:
                raise KeyboardError(
                    "No readable local keyboard. Run ./setup.sh for the device access instructions, "
                    "or select --keyboard-device /dev/input/eventN."
                )
            self.terminal = sys.stdin.fileno()
            self.terminal_settings = termios.tcgetattr(self.terminal)
            tty.setcbreak(self.terminal)
            self.thread = Thread(target=self.read_events, name="push-to-talk", daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        if self.error:
            raise KeyboardError(str(self.error)) from self.error

    def __exit__(self, *exc):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        self.microphone.set_recording(False)
        try:
            if self.terminal_settings is not None:
                termios.tcsetattr(self.terminal, termios.TCSAFLUSH, self.terminal_settings)
        finally:
            for fd in self.fds:
                os.close(fd)
            self.fds.clear()
