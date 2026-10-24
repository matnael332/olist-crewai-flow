"""Serve the API with waitress (works on Windows and Linux).

uv run pulse-api            # http://127.0.0.1:8000/health (same as python -m pulse_api.serve)
PORT=8080 uv run pulse-api  # e.g. on Railway (binds 0.0.0.0 when PORT is set)
"""

import os

from waitress import serve

from pulse_api.app import create_app


def main() -> None:
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    port = int(os.getenv("PORT", "8000"))
    app = create_app()
    bundle = app.config["MODEL_BUNDLE"]
    print(f"model: {'ok' if bundle.ok else 'DEGRADED - ' + str(bundle.error)}")
    print(f"serving on http://{host}:{port}")
    serve(app, host=host, port=port, threads=4)


if __name__ == "__main__":
    main()
