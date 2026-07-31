"""Shared type aliases for the compiler subpackage."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

SchedulerNodeList = list[Any]
CustomPassCallable = Callable[[Any], Any]
