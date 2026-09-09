# Diagnose a session without losing audio

Start the application in a terminal and expand **Troubleshooting details** after a
failure. Include the exact last error, app version, selected mode and stage timings.
Do not infer the cause from an unrelated initialization warning.

| Symptom | Next check |
|---|---|
| No captions | Confirm a single playing Chrome stream is selected. Check the last error. |
| Original audio changes in CC mode | Stop and report it: CC should not route or attenuate Chrome. |
| Dubbing starts then stops | Share stage timings and queue/drop counters; inference and playback have separate delays. |
| `audio recovery is pending` | Preserve the journal. An old journal must be reconciled with live audio identities, not blindly deleted. |
| Capture timeout or stalled video | Record whether the browser paused, changed stream or output; share the last error. |
| UI/browser freeze | Check available RAM/swap and concurrent heavy workloads; do not assume CUDA warnings are the cause. |
| Floating captions behind Chrome | Report desktop, Wayland/X11 session and whether XWayland is available. |

The runtime may report ONNX placement warnings during model initialization. These alone
do not explain a failed session. CPU fallback in one adapter does not mean the complete
pipeline supports CPU-only hardware.

## Safe test

Use a short video you are allowed to process. Begin with CC, verify that original audio
continues, pause/resume the video, then stop. Test dubbing separately and verify that
stopping restores original audio. Do not repeatedly restart a failing routed session.

No personal recordings or local audio-state journals belong in public issues. Review
screenshots and terminal output for usernames, file paths and private content first.
