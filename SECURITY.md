# Security

Only the current experimental prerelease is under active development; no security support
window is promised. Avoid processing sensitive audio with an unvalidated alpha.

Do not post credentials, private audio or exploitable security details in a public issue.
Use GitHub private vulnerability reporting if enabled for the repository. If unavailable,
open a minimal issue requesting a private contact without disclosing the vulnerability.

Runtime dependencies and models are downloaded only by explicit provisioning. Direct
versions/revisions are pinned; transitive dependencies are not fully locked, and recorded
local SHA-256 manifests are not trusted upstream signatures. Review third-party terms
and origins before running setup.
