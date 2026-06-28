"""Entry point for ``python -m promptguard.server``."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run:\n  pip install promptguard[server]",
            file=sys.stderr,
        )
        sys.exit(1)

    from promptguard.server.config import get_settings

    cfg = get_settings()

    print("\n  PromptGuard server")
    print("  ─────────────────")
    print(f"  Profile  : {cfg.profile_name} ({cfg.risk_tier} risk)")
    print(f"  Auth     : {'API key required' if cfg.api_key else 'disabled'}")
    print(
        f"  Rate limit: {cfg.rate_limit_rpm} req/min"
        if cfg.rate_limit_rpm
        else "  Rate limit: disabled"
    )
    print(f"  Listening: http://{cfg.host}:{cfg.port}\n")

    uvicorn.run(
        "promptguard.server.app:app",
        host=cfg.host,
        port=cfg.port,
        workers=cfg.workers,
        log_level=cfg.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
