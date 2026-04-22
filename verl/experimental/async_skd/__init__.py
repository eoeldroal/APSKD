"""Utilities for bounded asynchronous SKD."""

__all__ = [
    "AsyncSkdAgentLoopManager",
    "AsyncSkdDataSource",
    "AsyncSkdAgentLoopWorker",
    "AsyncSkdSample",
    "SkdPartialState",
]


def __getattr__(name):
    if name == "AsyncSkdAgentLoopManager":
        from .manager import AsyncSkdAgentLoopManager

        return AsyncSkdAgentLoopManager
    if name == "AsyncSkdDataSource":
        from .data_source import AsyncSkdDataSource

        return AsyncSkdDataSource
    if name == "AsyncSkdAgentLoopWorker":
        from .worker import AsyncSkdAgentLoopWorker

        return AsyncSkdAgentLoopWorker
    if name in {"AsyncSkdSample", "SkdPartialState"}:
        from .state import AsyncSkdSample, SkdPartialState

        return {"AsyncSkdSample": AsyncSkdSample, "SkdPartialState": SkdPartialState}[name]
    raise AttributeError(name)
