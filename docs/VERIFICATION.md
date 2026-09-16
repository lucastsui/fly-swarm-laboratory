# Packaging verification — September 15, 2026

Verified in the new isolated repository snapshot, without modifying the original project or resuming its training/demo:

- Clean `npm ci` installation completed (679 packages).
- `npm test`: **22 passed**, no failures.
- `npm run typecheck`: passed.
- `npm run build`: passed. Vite reported a large-client-chunk warning and vinext an unclassified route; neither prevented the build.
- Python source compilation: passed.
- `python -m unittest discover -s engine -t . -p 'test_*.py' -q` with `CUDA_VISIBLE_DEVICES=-1`: **418 tests, 12 skipped, no failures**. These are fixture/unit tests, not new whole-line training evidence.
- Artifact verification: all **8** allowlisted runtime files matched size/SHA-256.
- The reference checkpoint's gain-array hash matched all four paired evidence files. The viewer's own evidence validator recovered 62/64 completed factories and 190 products.
- The experimental phase manifest passed its own checkpoint, shape, finiteness, interface and hash checks, retaining validation status **pending**.
- Private deployment identifiers were removed from distributed source/reports. Pattern scans found no original SSH passwords, GitHub tokens, private-key blocks or personal deployment addresses in the staged source. Ignored runtime/dependency paths were confirmed absent from Git.

The initial broad test attempt exposed a missing-data guard in a full-graph test. This snapshot now skips it unless both CUDA and its prepared source tables are available; the scientific test itself was not weakened. GPU/data-dependent tests are explicitly skipped rather than claimed as verified.

Environment: Windows, Node 22.23.2, Python 3.11.15, original installed PyTorch 2.11.0+cu128. The repository's hosted Linux CI has not yet been observed running at packaging time. Source builds and fixtures do not prove behavioral reproduction, distributed performance, or a fresh complete-machine installation.

The companion runtime archive contains inference artifacts only, not optimizer state or exact live factory/neural-state snapshots. Scientific metrics in the README come from retained reports; no training or long-horizon frozen evaluation was restarted for packaging.
