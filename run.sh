#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    echo "Run $project_dir/setup.sh first." >&2
    exit 1
fi

whisper_args=()
explicit_whisper=false
for argument in "$@"; do
    case "$argument" in --whisper-cli|--whisper-cli=*) explicit_whisper=true ;; esac
done
if ! "$explicit_whisper" && ! command -v whisper-cli >/dev/null; then
    local_whisper="$project_dir/.tools/whisper.cpp/build/bin/whisper-cli"
    if [[ -x "$local_whisper" ]]; then
        whisper_args=(--whisper-cli "$local_whisper")
    fi
fi
# Keep the caller's working directory so relative model paths remain meaningful.
exec "$project_dir/.venv/bin/python" "$project_dir/translator.py" "${whisper_args[@]}" "$@"
