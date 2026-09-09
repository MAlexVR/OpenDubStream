# Third-party notices

## Included portions

- The Qt palette/startup portions identified in `src/opendubstream/ui/app.py` derive
  from BluCast. Preserve [BluCast's MIT notice](LICENSES/BluCast-MIT.txt), including
  Andrei9383 and MAlexVR fork copyright notices.
- The original OpenDubStream SVG has its own [CC0 declaration](assets/OPENDUBSTREAM-ICON-PROVENANCE.md).
- No BluCast artwork, NVIDIA Maxine SDK, Python environment or model weight is bundled.

OpenDubStream source is licensed under [MIT](LICENSE), copyright 2026 Mauricio Vargas.
These notices preserve the distinct terms of included and downloaded components.

## Downloaded separately

The provisioning manifest, `scripts/local-model-sources.json`, identifies exact sources
and revisions. Consult each upstream repository/model card for its applicable terms;
this project does not relicense these dependencies or models:

- Silero VAD: https://github.com/snakers4/silero-vad
- Faster Whisper: https://github.com/SYSTRAN/faster-whisper
- ASR model: https://huggingface.co/Systran/faster-distil-whisper-large-v3
- Translation model: https://huggingface.co/Helsinki-NLP/opus-mt-en-es
- Kokoro ONNX runtime/assets: https://github.com/thewh1teagle/kokoro-onnx
- PySide6/Qt licensing: https://doc.qt.io/qtforpython-6/licenses.html

Python dependencies retain their own licenses. NVIDIA driver/CUDA licensing is separate;
GPU software is not installed by the RPM. This file records provenance, not legal advice.
