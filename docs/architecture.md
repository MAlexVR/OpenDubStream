# Local caption and dubbing flow

The desktop UI starts a bounded application session. Audio capture, inference and playback
are separated so a slow model does not directly block reading the next audio chunk.

```text
Chrome stream → PipeWire capture → pause-aware phrases → VAD → ASR → EN→ES translation
                                                               ↓          ↓
                                                        captions/UI   optional TTS
                                                                          ↓
                                                               bounded playback + mix
```

| Layer | Responsibility |
|---|---|
| `domain` | Pipeline contracts and immutable values |
| `application` | Session lifecycle, cancellation, mode selection and ownership rules |
| `infrastructure/audio` | Discovery, capture, routing journal, playback and phrase segmentation |
| `infrastructure/models` | Local VAD, ASR, translation and TTS adapters |
| `pipeline` | Scheduling, bounded work and stage coordination |
| `ui` | Qt workspace, bilingual copy, floating captions and tray |

CC mode passively monitors only the selected stream and does not use dubbing routing,
TTS or playback. Dubbing owns temporary routing resources, preserves their identities
in a recovery journal and must restore them safely. It must not manipulate unrelated
streams or guess ownership from reused numeric IDs.

Models load lazily from local assets. Explicit provisioning is a separate, network-using
operation; a normal session is intended to use local inference. Queue bounds prevent
unbounded latency but can discard stale work. That tradeoff is exposed in diagnostics,
not presented as a guarantee of complete real-time translation.
