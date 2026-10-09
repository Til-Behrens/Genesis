"""Progress events shared by all long-running pipelines."""
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["info", "progress", "complete", "error"]


@dataclass(frozen=True)
class Event:
    """One progress update; `message` is ready to show, `data` carries results for callers.

    Pipelines are generators of events and stop after the first `error` or `complete`.
    """

    status: Status
    message: str
    progress: tuple[int, int] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.progress:
            return f"[{self.progress[0]}/{self.progress[1]}] {self.message}"
        return self.message


def info(message: str, **data: Any) -> Event:
    """Status message without progress, e.g. while loading a model."""
    return Event("info", message, data=data)


def progress(current: int, total: int, message: str, **data: Any) -> Event:
    """Message for item `current` of `total`."""
    return Event("progress", message, progress=(current, total), data=data)


def complete(message: str, **data: Any) -> Event:
    """Final event of a successful run; `data` holds its results."""
    return Event("complete", message, data=data)


def error(message: str, **data: Any) -> Event:
    """Final event of a failed run."""
    return Event("error", message, data=data)
