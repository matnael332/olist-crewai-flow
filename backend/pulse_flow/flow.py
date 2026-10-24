"""PulseFlow: the CrewAI Flow that hands off from the Data Analyst crew to the Data Scientist crew.

The dataset spec (``--spec`` / PIPELINE_DATASET_SPEC) is loaded inside check_inputs, so a missing
spec is a graceful failure, not an import error.

prepare_run -> check_inputs -> analyst stage (Crew 1) -> gate_contract -> [HANDOFF]
-> feature phase (Crew 2, phase 1) -> gate_required_features -> modeling phase (Crew 2, phase 2)
-> gate_artifacts -> promote_and_finish            (any gate or step failure -> fail_gracefully)

Every step catches its own exceptions and records them in the state (CrewAI 1.14.5 has no error
hook and kickoff() would otherwise raise); routers turn errors and gate reports into labels.
"""

import time
import traceback
from pathlib import Path
from typing import Literal

from crewai.flow import Flow, listen, router, start

from pulse_core import faults, stages
from pulse_core.io import read_json, sha256_file, write_json_lf, write_text_lf
from pulse_core.logging_setup import Redactor, get_logger
from pulse_core.manifest import build_manifest, write_manifest
from pulse_core.paths import repo_root
from pulse_core.promotion import PromotionError, promote
from pulse_core.raw import check_raw_schema, verify_manifest
from pulse_core.spec import DatasetSpec, load_spec
from pulse_core.status import STATUS_FILE, read_status, utcnow
from pulse_core.validators.artifacts_gate import validate_artifacts
from pulse_core.validators.contract_gate import validate_clean_data_against_contract
from pulse_core.validators.features_gate import validate_features
from pulse_core.validators.report import GateReport
from pulse_flow.crews import runner
from pulse_flow.llm import has_gemini_key
from pulse_flow.state import PipelineState

log = get_logger("flow")
REPO_ROOT = repo_root()
START_LABELS = {
    "": "inputs_ok",
    "analyst": "inputs_ok",
    "features": "contract_ok",
    "modeling": "features_ok",
}


class GateFailed(RuntimeError):
    """A gate run inside check_inputs (when resuming) failed."""


def is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return (
        isinstance(exc, faults.InjectedQuotaError)
        or "RESOURCE_EXHAUSTED" in text
        or (
            "429" in text
            and ("quota" in text.lower() or "rate" in text.lower() or "exhaust" in text.lower())
        )
    )


class PulseFlow(Flow[PipelineState]):
    name = "PulseFlow"
    tracing = False
    _skip_auto_memory: bool = True
    _dataset_spec: DatasetSpec | None = None

    def _spec(self) -> DatasetSpec:
        """The dataset spec, loaded once (a method: the Flow metaclass getattr's every attribute)."""
        if self._dataset_spec is None:
            self._dataset_spec = load_spec(self.state.spec_ref or None, self.state.spec_params)
        return self._dataset_spec

    # ---------------------------------------------------------------- helpers
    def _art(self) -> Path:
        return Path(self.state.run_dir) / "artifacts"

    def _sync_status(self, **extra) -> None:
        path = Path(self.state.run_dir) / STATUS_FILE
        data = read_status(path) or {}
        data.update(
            run_id=self.state.run_id,
            mode=self.state.mode,
            inject_fault=self.state.inject_fault,
            status=self.state.status,
            stage=self.state.stage,
            stages_done=list(self.state.stages_done),
            errors=list(self.state.errors),
            warnings=list(self.state.warnings),
            narrative_sources=dict(self.state.narrative_sources),
            started_at=self.state.started_at,
            finished_at=self.state.finished_at,
            **extra,
        )
        write_json_lf(path, data, ndigits=None)

    def _run_step(self, stage: str, work) -> bool:
        self.state.stage = stage
        self._sync_status(stage_started_at=utcnow())
        log.info("STAGE %s START", stage)
        started = time.perf_counter()
        try:
            work()
            ok = True
        except Exception as exc:  # noqa: BLE001 - every failure becomes a graceful, recorded error
            ok = False
            reason = (
                f"quota exhausted: {exc}" if is_quota_error(exc) else f"{type(exc).__name__}: {exc}"
            )
            self.state.errors.append(f"{stage}: {reason}")
            trace = Path(self.state.run_dir) / "traceback.txt"
            write_text_lf(trace, Redactor()(traceback.format_exc()))
            log.error("STAGE %s ERROR %s (traceback in %s)", stage, reason, trace.name)
        elapsed = time.perf_counter() - started
        log.info("STAGE %s END status=%s elapsed=%.1fs", stage, "ok" if ok else "failed", elapsed)
        if ok:
            self.state.stages_done.append(stage)
        self._sync_status()
        return ok

    def _record_gate(self, report: GateReport) -> bool:
        self.state.stage = report.gate
        self.state.gates = {**self.state.gates, report.gate: report.to_dict()}
        for check in report.checks:
            log.info(
                "GATE %s %s %s %s",
                report.gate,
                check["name"],
                "PASS" if check["passed"] else "FAIL",
                check["detail"],
            )
        log.info("GATE %s RESULT %s", report.gate, report.summary)
        if not report.passed:
            for error in report.errors:
                self.state.errors.append(error)
        else:
            self.state.stages_done.append(report.gate)
        self._sync_status()
        return report.passed

    def _narrate(self, phase: str) -> None:
        if self.state.inject_fault == "crew_error" and phase == "analyst":
            log.warning("FAULT INJECTED crew_error (simulated 429 from Gemini)")
            raise faults.InjectedQuotaError("429 RESOURCE_EXHAUSTED (injected fault)")
        if self.state.mode != "llm":
            return
        try:
            result = runner.narrate(phase, self._art(), self.state.model_dump(), self._spec())
            self.state.narrative_sources = {**self.state.narrative_sources, **result["sources"]}
            for warning in result.get("warnings", []):
                self.state.warnings.append(warning)
                log.warning("NARRATIVE %s", warning)
            usage = dict(self.state.token_usage)
            for key, value in (result.get("token_usage") or {}).items():
                usage[key] = usage.get(key, 0) + int(value)
            self.state.token_usage = usage
            log.info(
                "CREW phase=%s narratives=%s tokens=%s",
                phase,
                result["sources"],
                (result.get("token_usage") or {}).get("total_tokens"),
            )
        except Exception as exc:
            if is_quota_error(exc):
                raise
            message = f"narrative_degraded ({phase}): {type(exc).__name__}: {exc}"
            self.state.warnings.append(message)
            log.warning("NARRATIVE %s", message)

    def _fault(self, applied: bool) -> None:
        if applied:
            log.warning("FAULT INJECTED %s", self.state.inject_fault)

    def _write_validation_report(self) -> None:
        write_json_lf(
            self._art() / "validation_report.json",
            {"gates": dict(self.state.gates), "errors": list(self.state.errors)},
            ndigits=None,
        )

    def _manifest(self) -> dict:
        return build_manifest(
            self.state.model_dump(),
            Path(self.state.run_dir),
            self._art(),
            REPO_ROOT,
            Path(self.state.raw_dir),
        )

    # ------------------------------------------------------------------ steps
    @start()
    def prepare_run(self):
        Path(self.state.run_dir, "artifacts").mkdir(parents=True, exist_ok=True)
        Path(self.state.run_dir, "crew_logs").mkdir(parents=True, exist_ok=True)
        self.state.started_at = utcnow()
        self.state.stage = "prepare_run"
        self._sync_status()
        log.info(
            "RUN %s START mode=%s fault=%s seed=%s start_at=%s",
            self.state.run_id,
            self.state.mode,
            self.state.inject_fault or "none",
            self.state.seed,
            self.state.start_at or "analyst",
        )

    def _check_inputs(self) -> None:
        art = self._art()
        spec = self._spec()  # SpecNotConfigured becomes a graceful check_inputs failure
        log.info(
            "INPUT dataset spec %s (%s) params=%s",
            spec.name,
            self.state.spec_ref or "from env",
            self.state.spec_params or "none",
        )
        if self.state.start_at in ("features", "modeling"):
            needed = ["clean_data.csv", "dataset_contract.json"]
            if self.state.start_at == "modeling":
                needed += ["features.csv", "feature_spec.json"]
            missing = [n for n in needed if not (art / n).exists()]
            if missing:
                raise FileNotFoundError(
                    f"cannot resume from {self.state.from_run}: missing {missing}"
                )
            report = (
                validate_clean_data_against_contract(
                    art / "clean_data.csv", art / "dataset_contract.json"
                )
                if self.state.start_at == "features"
                else validate_features(
                    art / "features.csv", art / "feature_spec.json", art / "dataset_contract.json"
                )
            )
            if not self._record_gate(report):
                raise GateFailed(report.errors[0])
            log.info("RESUME from %s at %s", self.state.from_run, self.state.start_at)
            return
        raw_dir = Path(self.state.raw_dir)
        if self.state.inject_fault == "bad_input_hash":
            raw_dir = faults.tampered_raw_copy(raw_dir, Path(self.state.run_dir))
            log.warning("FAULT INJECTED bad_input_hash (one byte appended to a raw file copy)")
        verify_manifest(raw_dir, log=lambda message: log.info("INPUT %s", message))
        schema = check_raw_schema(raw_dir, required_tables=spec.required_tables)
        if not schema.passed:
            raise ValueError("; ".join(schema.errors))
        missing = [
            name
            for name in spec.required_lookups
            if not (Path(self.state.lookups_dir) / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"missing lookups {missing} in {self.state.lookups_dir}")
        if self.state.mode == "llm" and not has_gemini_key():
            raise RuntimeError("llm mode requires GEMINI_API_KEY (or run with --no-llm)")
        log.info("INPUT raw tables, schema and lookups verified")

    @router(prepare_run)
    def check_inputs(self) -> Literal["inputs_ok", "contract_ok", "features_ok", "failed"]:
        if not self._run_step("check_inputs", self._check_inputs):
            return "failed"
        return START_LABELS.get(self.state.start_at, "inputs_ok")

    @listen("inputs_ok")
    def run_analyst_stage(self):
        def work() -> None:
            summary = stages.run_analyst_stage(
                self._spec(), self.state.raw_dir, self.state.lookups_dir, self._art()
            )
            self.state.cleaning_counts = {
                k: v for k, v in summary.items() if isinstance(v, int | float)
            }
            self._narrate("analyst")
            self._fault(faults.after_analyst(self.state.inject_fault, self._art()))

        self._run_step("run_analyst_stage", work)

    @router(run_analyst_stage)
    def gate_contract(self) -> Literal["contract_ok", "failed"]:
        if self.state.errors:
            return "failed"
        art = self._art()
        report = validate_clean_data_against_contract(
            art / "clean_data.csv", art / "dataset_contract.json"
        )
        if not self._record_gate(report):
            return "failed"
        contract = read_json(art / "dataset_contract.json")
        required = [f["name"] for f in contract["required_features"]]
        split = contract["target"]["split_policy"]
        digest = {
            "contract_version": str(contract["contract_version"]),
            "contract_sha256": sha256_file(art / "dataset_contract.json"),
            "row_count": str(contract["row_count"]),
            "target_definition": contract["target"]["definition"],
            "required_features": ", ".join(required),
            "forbidden_features": ", ".join(contract["forbidden_features"]),
            "split_policy": f"train {split['train'][0]}..{split['train'][1]}; "
            f"test {split['test'][0]}..{split['test'][1]}",
        }
        self.state.contract_digest = digest
        log.info(
            "HANDOFF analyst->scientist contract_version=%s contract_sha256=%s rows=%s "
            "required=%d forbidden=%d",
            digest["contract_version"],
            digest["contract_sha256"][:12],
            digest["row_count"],
            len(required),
            len(contract["forbidden_features"]),
        )
        return "contract_ok"

    @listen("contract_ok")
    def run_feature_phase(self):
        def work() -> None:
            stages.run_feature_stage(self._spec(), self._art())
            self._narrate("features")
            self._fault(faults.after_features(self.state.inject_fault, self._art()))

        self._run_step("run_feature_phase", work)

    @router(run_feature_phase)
    def gate_required_features(self) -> Literal["features_ok", "failed"]:
        if self.state.errors:
            return "failed"
        art = self._art()
        report = validate_features(
            art / "features.csv", art / "feature_spec.json", art / "dataset_contract.json"
        )
        return "features_ok" if self._record_gate(report) else "failed"

    @listen("features_ok")
    def run_modeling_phase(self):
        def work() -> None:
            summary = stages.run_modeling_stage(self._spec(), self._art(), seed=self.state.seed)
            self.state.training_summary = {
                "selected_model": summary["selected_model"],
                **{
                    k: v
                    for k, v in summary["test"].items()
                    if k in ("roc_auc", "pr_auc", "precision_at_top10pct", "prevalence")
                },
            }
            self._narrate("modeling")
            self._fault(faults.after_modeling(self.state.inject_fault, self._art()))

        self._run_step("run_modeling_phase", work)

    @router(run_modeling_phase)
    def gate_artifacts(self) -> Literal["artifacts_ok", "failed"]:
        if self.state.errors:
            return "failed"
        return "artifacts_ok" if self._record_gate(validate_artifacts(self._art())) else "failed"

    @listen("artifacts_ok")
    def promote_and_finish(self):
        self.state.stage = "promote_and_finish"
        art, run_dir = self._art(), Path(self.state.run_dir)
        self._write_validation_report()
        self.state.status = "success"
        self.state.finished_at = utcnow()
        self.state.stages_done.append("promote_and_finish")
        self._sync_status()
        write_json_lf(art / STATUS_FILE, read_json(run_dir / STATUS_FILE), ndigits=None)
        # CrewAI writes its transcripts in text mode (CRLF on Windows): store LF copies.
        for transcript in sorted((run_dir / "crew_logs").glob("*.json")):
            text = transcript.read_text(encoding="utf-8", errors="replace")
            write_text_lf(art / "logs" / transcript.name, text.replace("\r\n", "\n"))
        write_manifest(art / "run_manifest.json", self._manifest())
        if self.state.promote:
            extra = {"logs/pipeline.log": run_dir / "pipeline.log"}
            for handler in log.parent.handlers if log.parent else []:
                handler.flush()
            log.info("PROMOTE START %s -> %s", art.as_posix(), self.state.artifacts_dir)
            try:
                promote(art, Path(self.state.artifacts_dir), self.state.run_id, extra=extra)
                log.info("PROMOTE DONE")
            except PromotionError as exc:
                self.state.status = "promotion_failed"
                self.state.errors.append(f"promote: {exc}")
                log.error("PROMOTE FAILED %s (previous artifacts kept)", exc)
                write_json_lf(
                    art / STATUS_FILE,
                    {**read_json(art / STATUS_FILE), "status": "promotion_failed"},
                    ndigits=None,
                )
        self._sync_status()
        write_manifest(run_dir / "run_manifest.json", self._manifest())
        log.info("RUN %s END status=%s", self.state.run_id, self.state.status)
        return self.state.status

    @listen("failed")
    def fail_gracefully(self):
        self.state.status = "failed"
        self.state.finished_at = utcnow()
        run_dir = Path(self.state.run_dir)
        self._write_validation_report()
        self._sync_status()
        write_json_lf(self._art() / STATUS_FILE, read_json(run_dir / STATUS_FILE), ndigits=None)
        write_manifest(run_dir / "run_manifest.json", self._manifest())
        first = self.state.errors[0] if self.state.errors else "unknown error"
        log.error("RUN %s FAILED at %s: %s", self.state.run_id, self.state.stage, first)
        for error in self.state.errors[1:]:
            log.error("  - %s", error)
        log.error("Nothing was promoted; %s is unchanged.", self.state.artifacts_dir)
        log.error("Details: %s", (run_dir / STATUS_FILE).as_posix())
        log.info("RUN %s END status=failed", self.state.run_id)
        return "failed"
