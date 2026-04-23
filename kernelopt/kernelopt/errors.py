"""User-facing errors (CLI prints a message, no traceback)."""


class KerneloptUserError(Exception):
    """Expected configuration or environment problem; not a bug."""
