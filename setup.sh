#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
skip_system=false
case "${1:-}" in
    --skip-system) skip_system=true; shift ;;
    -h|--help)
        echo "Usage: ./setup.sh [--skip-system]"
        echo "Install system/Python dependencies and build whisper-cli if absent. No model downloads."
        echo "Set PYTHON=python3.11 to select Python; BUILD_JOBS=1 reduces build memory use."
        exit 0 ;;
esac
if (( $# )); then
    echo "Unknown argument: $1 (see ./setup.sh --help)" >&2
    exit 2
fi
case "$(uname -m)" in
    x86_64|aarch64) ;;
    *) echo "Use 64-bit Linux (x86-64 or AArch64); Raspberry Pi OS must be 64-bit." >&2; exit 1 ;;
esac

if ! "$skip_system"; then
    elevate=()
    if (( EUID != 0 )); then
        command -v sudo >/dev/null || { echo "Install sudo or use --skip-system after installing system packages." >&2; exit 1; }
        elevate=(sudo)
    fi
    . /etc/os-release
    case " ${ID:-} ${ID_LIKE:-} " in
        *fedora*|*rhel*)
            "${elevate[@]}" dnf install -y python3 python3-pip portaudio pipewire-alsa acl git cmake gcc-c++ make ;;
        *debian*|*ubuntu*|*raspbian*)
            "${elevate[@]}" apt-get update
            "${elevate[@]}" apt-get install -y python3 python3-venv python3-pip libportaudio2 acl git cmake g++ make ;;
        *) echo "Unsupported distro. Install the README prerequisites, then use --skip-system." >&2; exit 1 ;;
    esac
fi

python_command="${PYTHON:-python3}"
"$python_command" -c 'import sys; sys.exit("Python 3.11+ is required") if sys.version_info < (3, 11) else None'
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    "$python_command" -m venv "$project_dir/.venv"
fi
"$project_dir/.venv/bin/python" -m pip install --disable-pip-version-check -r "$project_dir/requirements.txt"
"$project_dir/.venv/bin/python" - "$project_dir" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import translator
translator.dependencies()
for source, target in (("en", "zh"), ("zh", "en")):
    translator.translation_model(Path(sys.argv[1]) / "models" / f"{source}-{target}", source, target)
print("Python dependencies and bundled translation model files are ready.")
PY

whisper_source="$project_dir/.tools/whisper.cpp"
if command -v whisper-cli >/dev/null; then
    echo "Using whisper-cli from PATH."
elif [[ -x "$whisper_source/build/bin/whisper-cli" ]]; then
    echo "Using the existing local whisper-cli build."
else
    for tool in git cmake g++ make; do
        command -v "$tool" >/dev/null || { echo "Missing $tool; rerun setup without --skip-system." >&2; exit 1; }
    done
    mkdir -p "$project_dir/.tools"
    if [[ ! -d "$whisper_source" ]]; then
        git clone --depth 1 --branch v1.9.4 https://github.com/ggml-org/whisper.cpp.git "$whisper_source"
    fi
    cmake -S "$whisper_source" -B "$whisper_source/build" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DWHISPER_CURL=OFF \
        -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF
    cmake --build "$whisper_source/build" --target whisper-cli --parallel "${BUILD_JOBS:-2}"
fi

echo "Setup complete. Run: ./run.sh --model /path/to/ggml-tiny.bin"
"$project_dir/.venv/bin/python" - "$project_dir" <<'PY'
import getpass
import os
import shlex
import sys
sys.path.insert(0, sys.argv[1])
from keyboard_input import keyboard_devices
devices = list(keyboard_devices())
readable = [(path, name) for path, name in devices if os.access(path, os.R_OK)]
if not devices:
    print("No local keyboard detected. Attach a keyboard for hold-to-talk, or use --continuous.")
elif not readable:
    print("Hold-to-talk needs read access to your keyboard. Choose your keyboard below:")
    for path, name in devices:
        print(f"  {path}: {name}")
        print(f"    sudo setfacl -m u:{shlex.quote(getpass.getuser())}:r {shlex.quote(str(path))}")
    print("This grants raw keyboard access for this device until its permissions reset (e.g. reboot).")
    print("The translator itself should run as your normal user.")
else:
    print("Readable keyboards for hold-to-talk:")
    for path, name in readable:
        print(f"  {path}: {name}")
PY
