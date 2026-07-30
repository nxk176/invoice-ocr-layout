# Repository instructions

- Never commit real invoices, ground truth, rendered pages, crops, predictions, logs,
  secrets, model checkpoints, or caches.
- Test data must be synthetic and must not contain real patient, hospital, supplier, or
  invoice information.
- Missing dependencies, checkpoints, and annotations must produce actionable errors;
  never fabricate successful model output.
- Keep model integrations behind adapters. Do not vendor third-party repositories.
- Use English for code, type hints, and log messages. The user-facing README is Vietnamese.
- Run Ruff, mypy, pytest, and the CPU integration smoke test before pushing.
- Do not force-push.

