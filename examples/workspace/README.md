# Example workspace

This directory is the **sandbox** that the coding agent operates inside.
Every `read_file`, `write_file`, `list_dir` and `run_shell` call is rooted
here and rejected if the resolved path escapes.

Drop your own files in here before running the agent against a real task,
or let the smoke test (`examples/01_smoke_test_fake.py`) seed it.
