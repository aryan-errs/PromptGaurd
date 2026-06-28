"""@guard.middleware decorator (stub)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from promptguard.pipeline import Pipeline
from promptguard.types import Verdict


def middleware(pipeline: Pipeline) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that runs the pipeline on the first positional str argument."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            text = args[0] if args else kwargs.get("text", "")
            verdict: Verdict = pipeline.run(str(text))
            if verdict.blocked:
                raise PermissionError(f"PromptGuard blocked input (score={verdict.score:.2f})")
            return fn(*args, **kwargs)

        return wrapper

    return decorator
