# Bundled translation models

The app automatically reads `en-zh/` and `zh-en/` from this directory.
Both are Argos Translate 1.9 packages, derived from OPUS-MT, downloaded from
[Argos's model repository](https://data.argosopentech.com/argospm/v1/).
Together the retained files occupy about **170 MB (162 MiB)**.

The original package README and attribution are preserved in each directory.
The upstream models are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in those
README files; the application's license does not replace the model license.
Authors: Jörg Tiedemann and Santhosh Thottingal, *OPUS-MT — Building open
translation services for the World*, EAMT 2020, Lisbon, Portugal.

Only the inference model, tokenizer, metadata, and README were retained.
Unused Stanza sentence-splitting models were omitted. `manifest.json` records
the source URLs, original archive SHA-256 checksums, and retained file checksums.
The app never downloads or modifies these models. Keep this directory with the
Python files when deploying to another machine.
