# translator-mini

Offline English ↔ Chinese microphone translation for Linux. Python controls
whisper.cpp speech recognition and the bundled Argos/CTranslate2 translation
models. Partial translations appear while you speak. No runtime downloads.

## Setup and run

On Fedora or 64-bit Raspberry Pi OS / Debian / Ubuntu:

```sh
./setup.sh
./run.sh --model /path/to/ggml-tiny.bin
```

Setup installs system packages using `dnf` or `apt-get` (via sudo when needed),
creates `.venv`, and builds [whisper.cpp v1.9.4](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.4)
in `.tools/` if `whisper-cli` is not already available. It downloads software,
**never models**. Rerunning setup reuses the environment and existing build.
Use `./setup.sh --skip-system` if system dependencies are already installed.
`PYTHON=python3.11 ./setup.sh` selects Python; `BUILD_JOBS=1` reduces build memory.

Requires Python 3.11+, PortAudio, and 64-bit x86 or AArch64 Linux. Old x86 CPUs
need SSE 4.1. Raspberry Pi OS must be **64-bit** (`uname -m`: `aarch64`). Set your
microphone as the default input; it must support 16 kHz mono recording. Setup
installs Fedora's `pipewire-alsa` bridge for the system default input.

Supply a local **multilingual** Whisper `.bin` model. Start with `tiny` on slow
hardware; English-only `.en` models cannot support both languages. The translation
models are already bundled in `models/` (~170 MB); only the Whisper model path is
configurable. See [model sources and attribution](models/README.md).

Copy the app and `models/` to a Pi, then run setup there. Recreate `.venv` and
`.tools` on the target machine; Fedora x86 builds will not run on ARM.

## Hold to talk

**Hold Space or Enter to record; release to stop.** Either key works, including
keypad Enter. Auto-repeat is ignored. If both keys are held, release both to stop.
Partial translations continue during the press; release submits the remaining
audio immediately. **Ctrl+C** exits and restores the terminal.

True release detection uses Linux keyboard input events. Setup lists your
keyboard devices and, if needed, prints a command such as:

```sh
sudo setfacl -m u:"$(id -un)":r /dev/input/eventN
./run.sh --model /path/to/ggml-tiny.bin --keyboard-device /dev/input/eventN
```

Replace `eventN` with your keyboard from setup's output. This grants read access
to **raw keyboard events**, not just Space/Enter; use it only for your own account
and keyboard. The app uses only Space/Enter and never grabs or logs other keys.
Access may need granting again after reboot/reconnection. Run the app as your
normal user. This works with a **locally attached keyboard**, including Wayland;
Space/Enter also trigger it when another window has focus. Remote SSH keystrokes
cannot provide these local key-release events.

For continuous listening, including use over SSH, keyboard access is unnecessary:

```sh
./run.sh --model /path/to/ggml-tiny.bin --continuous
```

## Options and limits

- `--interval 0.5`: submit partials more often (0.5–4 seconds; default 1).
- `--speech-threshold 150`: detect quieter speech (RMS threshold; default 300).
  Increase it if background noise triggers recognition.
- `--threads 1`: reduce inference CPU usage (default 2).
- `--whisper-cli /path/to/whisper-cli`: override the automatically located binary.

Latency is the audio interval plus model loading/inference time. Models run
sequentially to conserve memory while microphone/key handling continues in the
background. Partial text may change. A pause or eight-second window produces a
final update; window boundaries may split words. The simple energy gate can miss
quiet speech or mistake noise for speech. Output is text only.

Audio buffers are bounded, outdated partials are replaced, and at most two
windows wait for inference. Dropped windows produce a warning when processing
cannot keep up. Temporary audio/transcription files are removed after inference,
including on errors and Ctrl+C. In hold-to-talk mode, idle audio is discarded.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover language routing, streaming, real key event handling, terminal cleanup,
model loading, and script argument handling with simulated hardware. Bundled
models were also tested with real translations in both directions. Microphone
accuracy, latency, and Raspberry Pi execution need testing on the target hardware.
