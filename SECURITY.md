# Security and deployment

- Never commit SSH passwords, API tokens, `.env` files, `.runtime`, private keys, credential files or personal host inventories.
- The dashboard viewers are intentionally bound to `127.0.0.1`. Their control routes are not a public multi-user service. CORS is not authentication.
- Additional trusted dashboard origins can be configured with `FLY_DASHBOARD_ORIGINS` (comma-separated). Do not use a wildcard and do not expose viewer ports directly to the Internet.
- Experimental cluster recovery uses an explicit private `FLY_COORDINATOR_HOST`, TLS and bearer tokens. Keep the coordinator behind a trusted private network/firewall. It is not a hardened public service. All workers must share the same endpoint setting, pinned certificate, interface, graph and compatible code.
- Generate fresh per-experiment credentials; no original credentials are distributed. `engine.layout_recovery_security` requires OpenSSL and does not overwrite an existing credential directory. Windows ACLs require separate review; POSIX chmod is not a complete Windows access-control policy.
- Load checkpoints only from trusted sources. NPZ readers use `allow_pickle=False`; historical PyTorch files should not be treated as safe arbitrary downloads.
- Do not run operational verification commands against a live learner casually: some legacy tools pause, resume or checkpoint training. The documented smoke suite does not use the live services.
- This snapshot removes personal infrastructure identifiers and old hosting history. It does not convert all research scripts into a production deployment framework.

Report sensitive issues privately to the repository owner; do not put credentials in GitHub issues.
