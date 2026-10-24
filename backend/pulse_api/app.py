"""Flask app factory: GET /health, POST /predict, POST /predict/batch and GET /risk-list.

The full contract (requests, responses, error codes) is in docs/api.md.

The model is loaded once at startup with scikit-learn version mismatches treated as errors; if
anything is wrong the API starts in a degraded mode (/health says why, /predict answers 503)
instead of crashing.
"""

from __future__ import annotations

import os
from pathlib import Path

import sklearn
from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

from pulse_api.batch import BatchError, score_csv, score_records
from pulse_api.model import ModelBundle
from pulse_api.replay import ReplayError, backtest, load_clean, orders_for_day, parse_day
from pulse_api.risk import LEVELS
from pulse_core.paths import repo_root

ROOT = repo_root()


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
        if exc.code == 413:
            return jsonify(error="too_large", detail="the request body is larger than 2 MB"), 413
        return jsonify(error=exc.name.lower().replace(" ", "_"), detail=exc.description), exc.code

    def model_block() -> dict:
        return {
            "name": bundle.meta["model_name"],
            "threshold": bundle.meta["threshold"],
            "top10_cutoff": round(bundle.top10_cutoff, 6),
            "contract_version": bundle.meta["contract_version"],
        }

    def unavailable():
        return jsonify(error="model_unavailable", detail=bundle.error), 503

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
            risk_levels=bundle.policy.describe(),
        )

    @app.post("/predict")
    def predict():
        if not bundle.ok:
            return unavailable()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "records" not in payload:
            return jsonify(
                error="invalid_request", detail="body must be a JSON object with a 'records' list"
            ), 400
        predictions, errors, warnings = bundle.predict(payload["records"])
        if errors:
            status = 400 if any(e["index"] is None for e in errors) else 422
            return jsonify(error="validation_failed", details=errors), status
        return jsonify(model=model_block(), predictions=predictions, warnings=warnings)

    @app.post("/predict/batch")
    def predict_batch():
        """CSV in (text/csv body or multipart 'file'), ranked risk list out; partial success."""
        if not bundle.ok:
            return unavailable()
        upload = request.files.get("file")
        data = upload.read() if upload else request.get_data()
        if not data:
            return jsonify(
                error="no_file",
                detail="send the CSV as the request body (text/csv) or as a multipart 'file'",
            ), 400
        try:
            result = score_csv(bundle, data)
        except BatchError as exc:
            return jsonify(exc.to_dict()), exc.status
        if not result.predictions:
            return jsonify(
                error="no_valid_rows", detail="no row passed validation", errors=result.errors
            ), 422
        if request.args.get("format") == "csv":
            return Response(
                result.to_csv(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=scored_orders.csv"},
            )
        return jsonify(
            model=model_block(),
            summary=result.summary,
            predictions=result.predictions,
            errors=result.errors,
            warnings=result.warnings,
        )

    @app.get("/risk-list")
    def risk_list():
        """Replay a test-window day: its orders scored and ranked, optionally with the outcome."""
        if not bundle.ok:
            return unavailable()
        min_level = request.args.get("min_level", "elevated")
        if min_level not in LEVELS:
            return jsonify(error="invalid_level", detail=f"min_level must be one of {LEVELS}"), 400
        try:
            day = parse_day(request.args.get("date"))
            records, outcomes = orders_for_day(load_clean(art), day)
        except ReplayError as exc:
            return jsonify(exc.to_dict()), exc.status
        except OSError:
            return jsonify(error="data_unavailable", detail="clean_data.csv not found"), 503
        result = score_records(bundle, records)
        keep = LEVELS[: LEVELS.index(min_level) + 1]
        predictions = [p for p in result.predictions if p["risk_level"] in keep]
        with_outcome = request.args.get("include_outcome", "false").lower() in {"1", "true", "yes"}
        if with_outcome:
            for prediction in predictions:
                prediction["actual_late"] = bool(outcomes.get(prediction["order_id"], 0))
        body = {
            "model": model_block(),
            "date": day.isoformat(),
            "summary": result.summary,
            "predictions": predictions,
            "errors": result.errors,
        }
        if with_outcome:
            body["backtest"] = backtest(result.predictions, outcomes)
        return jsonify(body)

    return app
