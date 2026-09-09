# Contributing

This is an experimental project. Start with a focused issue describing the user-visible
problem and a minimal reproduction without private audio or host data.

1. Use Python 3.12 and the test setup in the README.
2. Add a failing regression test before changing runtime behavior.
3. Keep fake/offline tests separate from explicitly consented hardware validation.
4. Include tests and user-facing documentation with the behavior change.

Use conventional commit messages, with no AI attribution or `Co-Authored-By` trailers.
Keep code, identifiers and technical documentation in English. Preserve the existing
English/Spanish UI translations. Never claim that an offline green suite proves real
Chrome audio, latency or recovery correctness.

Never commit virtual environments, model weights, recordings, credentials, recovery
journals or machine-specific benchmark dumps. Avoid broad changes to audio ownership
without targeted regression coverage. Do not submit external code without its applicable
license and attribution. Project contributions are distributed under the MIT License; retain third-party notices.
