"""One exception type per reason the tool can decline, because the CLI exit code
is part of the contract and a bare ValueError cannot carry it.

`Unanswerable` is deliberately separate from `ToolError`: the first means the tool
worked and the question cannot be answered from this evidence, the second means the
tool itself broke. Collapsing them is how a monitoring tool ends up reporting a crash
as a clean result.
"""

from __future__ import annotations


class FepError(Exception):
    """Base class, so a caller can catch everything this package raises."""


class ToolError(FepError):
    """The tool failed: bad input, missing fixture, malformed file. Exit code 3."""


class Unanswerable(FepError):
    """The tool worked and the question cannot be answered from this evidence.

    Exit code 2. Carries the diagnosis the user needs in order to make the question
    answerable, because "refused" with no remedy is indistinguishable from a crash.
    """

    def __init__(self, message: str, *, remedy: str) -> None:
        super().__init__(message)
        self.remedy = remedy
