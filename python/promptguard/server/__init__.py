"""PromptGuard HTTP service — expose the detection pipeline as a REST API.

Install deps:   pip install promptguard[server]
Run:            python -m promptguard.server
                uvicorn promptguard.server.app:app --host 0.0.0.0 --port 8000

Environment variables (all prefixed PROMPTGUARD_):
  PROFILE_NAME                default profile name (default: "default")
  RISK_TIER                   low | medium | high (default: "medium")
  ALLOW_SECURITY_DISCUSSION   true | false (default: false)
  TEMPLATE_DELIMITERS         comma-separated list (default: "")
  TOOLS_ENABLED               true | false (default: false)
  API_KEY                     shared secret for X-API-Key auth (default: disabled)
  RATE_LIMIT_RPM              requests/minute per IP; 0 = disabled (default: 0)
  MAX_REQUEST_BYTES           body size cap in bytes (default: 65536)
  CLASSIFIER_MODEL_PATH       path to trained Backend A .pkl artifact
  HOST                        bind host (default: 0.0.0.0)
  PORT                        bind port (default: 8000)
  WORKERS                     uvicorn worker count (default: 1)
"""

from promptguard.server.app import create_app

__all__ = ["create_app"]
