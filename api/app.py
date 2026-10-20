"""Flask app factory: GET /health and POST /predict.

The model is loaded once at startup with scikit-learn version mismatches treated as errors; if
anything is wrong the API starts in a degraded mode (/health says why, /predict answers 503)
instead of crashing.
"""

from __future__ import annotations

import os
from pathlib import Path

import sklearn
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from api.model import ModelBundle

ROOT = Path(__file__).resolve().parents[1]


def create_app(
    artifacts_dir: str | Path | None = None, lookups_dir: str | Path | None = None
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    art = Path(artifacts_dir or os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
    lookups = Path(lookups_dir or ROOT / "data" / "lookups")
    bundle = ModelBundle(art, lookups)
    app.config["MODEL_BUNDLE"] = bundle

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify(error=exc.name.lower().replace(" ", "_"), detail=exc.description), exc.code

    @app.get("/health")
    def health():
        if not bundle.ok:
            return jsonify(status="degraded", reason=bundle.error), 200
        meta = bundle.meta
        return jsonify(
            status="ok",
            model=meta["model_name"],
            estimator=meta["estimator"],
            scikit_learn=sklearn.__version__,
            n_features=len(meta["model_features"]),
            threshold=meta["threshold"],
            top10_risk_cutoff=round(bundle.top10_cutoff, 6),
            contract_version=meta["contract_version"],
            contract_sha256=meta["contract_sha256"],
        )

    @app.post("/predict")
    def predict():
        if not bundle.ok:
            return jsonify(error="model_unavailable", detail=bundle.error), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "records" not in payload:
            return jsonify(
                error="invalid_request", detail="body must be a JSON object with a 'records' list"
            ), 400
        predictions, errors, warnings = bundle.predict(payload["records"])
        if errors:
            status = 400 if any(e["index"] is None for e in errors) else 422
            return jsonify(error="validation_failed", details=errors), status
        return jsonify(
            model={
                "name": bundle.meta["model_name"],
                "threshold": bundle.meta["threshold"],
                "contract_version": bundle.meta["contract_version"],
            },
            predictions=predictions,
            warnings=warnings,
        )

    return app
