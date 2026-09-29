# Project instructions

Build a minimal Linux microphone translator in Python for an older CPU with 4 GB RAM.

- Use the system default microphone and whisper.cpp for multilingual transcription.
- Detect English or Chinese and translate into the other language.
- Accept a configurable path only for the Whisper model. Load the bundled Argos 1.9 translation models from models/en-zh and models/zh-en relative to the app, independent of the working directory.
- Never download models or access model registries at runtime.
- Support Fedora and future deployment to 64-bit Raspberry Pi OS; document both package managers.
- Default to real hold-to-talk: Space or Enter down enables capture; release flushes the final update. Use Linux input key-up events, ignore auto-repeat, and keep keyboard handling responsive during inference. Keep continuous mode available via --continuous.
- Provide setup.sh for Fedora and 64-bit Raspberry Pi OS and run.sh for launching through the project virtual environment. Setup may install dependencies/build whisper.cpp but must never download models.
- Keep dependencies, CPU use, memory use, and the CLI small. Run inference sequentially while microphone capture continues independently.
- Bound audio buffers and pending work. Coalesce outdated partials and explicitly report dropped speech windows if inference cannot keep up.
- Use temporary audio files and remove them after processing. Do not retain recordings.
- Provide one command to run the app after setup, with clear errors for missing prerequisites.
- Verify language routing, local model loading, subprocess failures, live capture during inference, backpressure, and recording cleanup.
- Finish with a concise README covering setup, model layout, usage, and limitations.
