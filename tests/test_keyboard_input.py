import contextlib
import io
import os
import pty
import sys
import termios
from threading import Event
import unittest
from unittest.mock import MagicMock, call, patch

import keyboard_input as keyboard


class KeyboardTests(unittest.TestCase):
    def test_space_enter_release_and_autorepeat(self):
        mic = MagicMock()
        controls = keyboard.PushToTalk(mic)
        with contextlib.redirect_stdout(io.StringIO()):
            controls.handle_event(1, 1, 57, 1)  # Space down
            controls.handle_event(1, 1, 57, 2)  # Repeat must not toggle
            controls.handle_event(1, 1, 28, 1)  # Enter also held
            controls.handle_event(1, 1, 57, 0)  # Enter keeps capture active
            mic.set_recording.assert_called_once_with(True)
            controls.handle_event(1, 1, 28, 0)
            controls.handle_event(1, 1, 30, 1)  # Unrelated key
            controls.handle_event(1, 1, 96, 1)  # Keypad Enter
            controls.handle_event(1, 1, 96, 0)
        self.assertEqual(mic.set_recording.call_args_list,
                         [call(True), call(False), call(True), call(False)])

    def test_lost_events_fail_instead_of_leaving_recording_enabled(self):
        with self.assertRaisesRegex(keyboard.KeyboardError, "lost"):
            keyboard.PushToTalk(MagicMock()).handle_event(1, 0, 3, 0)

    def test_background_release_and_terminal_restoration(self):
        master, slave = pty.openpty()
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        original = termios.tcgetattr(slave)
        stdin = MagicMock()
        stdin.isatty.return_value = True
        stdin.fileno.return_value = slave
        down, up = Event(), Event()
        mic = MagicMock()
        mic.set_recording.side_effect = lambda enabled: (down if enabled else up).set()
        real_ioctl = keyboard.fcntl.ioctl

        def ioctl(fd, request, bitmap):
            if request & 0xFF == 0x21:
                for code in (28, 57):
                    bitmap[code // 8] |= 1 << (code % 8)
                return 0
            return real_ioctl(fd, request, bitmap)

        with patch.object(sys, "stdin", stdin), patch.object(keyboard.fcntl, "ioctl", side_effect=ioctl), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with keyboard.PushToTalk(mic, f"/proc/self/fd/{read_fd}") as controls:
                    os.write(write_fd, keyboard.INPUT_EVENT.pack(0, 0, 1, 57, 1))
                    self.assertTrue(down.wait(2))
                    os.write(write_fd, keyboard.INPUT_EVENT.pack(0, 0, 1, 57, 0))
                    # No main-loop polling: release is handled even while inference blocks.
                    self.assertTrue(up.wait(2))
                    controls.check()
                    raise RuntimeError("inference failed")
        self.assertEqual(termios.tcgetattr(slave), original)
        self.assertFalse(controls.thread.is_alive())
        self.assertEqual(controls.fds, [])

    def test_no_keyboard_has_actionable_error(self):
        stdin = MagicMock()
        stdin.isatty.return_value = True
        with patch.object(sys, "stdin", stdin), patch.object(keyboard, "keyboard_devices", return_value=[]), \
             self.assertRaisesRegex(keyboard.KeyboardError, "No readable local keyboard"):
            with keyboard.PushToTalk(MagicMock()):
                self.fail("should not enter")


if __name__ == "__main__":
    unittest.main()
