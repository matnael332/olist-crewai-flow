"""The dataset spec: the one extension point between the generic pipeline and a dataset.

Producers (stages, the contract builder, the feature builder, the Flow, the crews) receive a
``DatasetSpec`` explicitly. Gates and readers never see it: everything they need is written into
``dataset_contract.json`` and ``feature_spec.json`` by the producers, so a gate depends on the
artifacts only.

Conventions a spec author must honour (the generic renderers rely on them):

- ``clean(tables, lookups_dir)`` returns ``(clean_df, report)``; ``report`` has ``rows_in``,
  ``rows_out`` and ``rules`` (each ``{rule, description, rows_before, rows_after, rows_removed,
  detail}``), plus an optional ``definitions`` mapping.
- ``eda_summary(tables, clean, report)`` returns a JSON-friendly dict with ``positive_rate`` (a
  float in [0, 1]) and ``key_numbers`` (a list of ``{"label": str, "value": number}``). Any other
  key is free-form and reaches the crews' tools as is.
- Every callable must be deterministic (seeded, stably sorted): two runs must be byte-identical.
- ``time_column`` is a real timestamp (``DATETIME_FORMAT``), because splits are day-granular.
- ``splits`` must contain ``train`` and ``test``; other names are excluded from the metrics.

This module imports leaf modules only, so it can never take part in an import cycle.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pulse_core.columns import Column
from pulse_core.narratives import CleaningNarrative, ContractNarrative, Insights

SPEC_ENV = "PIPELINE_DATASET_SPEC"


class SpecNotConfigured(RuntimeError):
    """No dataset spec was given (``--spec`` or the PIPELINE_DATASET_SPEC variable)."""


@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    # identity and provenance
    name: str
    source: str
    license: str
    grain: str
    snapshot_date: str
    contract_version: str = "1.0"
    seed: int = 42
    required_lookups: tuple[str, ...] = ()

    # row semantics
    id_column: str
    time_column: str
    split_column: str = "split"
    decision_stage: str
    entity: str
    decision_time: str

    # target
    target: str
    target_definition: str
    target_label: str
    rejected_target: str | None = None

    # time split
    splits: Mapping[str, tuple[str, str]]
    split_note: str = ""
    cv_note: str = ""

    # clean layer
    columns: tuple[Column, ...]
    clean: Callable[[dict[str, pd.DataFrame], Path], tuple[pd.DataFrame, dict]]
    patterns: Mapping[str, str] = field(default_factory=dict)
    bounds: Mapping[str, dict] = field(default_factory=dict)
    constraints: Sequence[dict] = ()
    allowed_values: Mapping[str, Sequence | str] = field(default_factory=dict)
    population: Mapping[str, object] = field(default_factory=dict)
    privacy: Mapping[str, object] = field(default_factory=dict)

    # features
    feature_definitions: tuple[dict, ...]
    required_features: tuple[dict, ...]
    forbidden_features: tuple[str, ...]
    derive_features: Callable[[pd.DataFrame], pd.DataFrame]

    # exploratory analysis
    eda_summary: Callable[[dict[str, pd.DataFrame], pd.DataFrame, dict], dict]
    eda_figures: Callable[[dict, pd.DataFrame, Path], list[str]] | None = None

    # narrative templates (None: the generic templates in pulse_core.reporting.templates)
    cleaning_template: Callable[[DatasetSpec, dict], CleaningNarrative] | None = None
    insights_template: Callable[[DatasetSpec, dict], Insights] | None = None
    contract_template: Callable[[DatasetSpec, dict, dict], ContractNarrative] | None = None
    model_card_items: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = {"train", "test"} - set(self.splits)
        if missing:
            raise ValueError(f"splits must define train and test; missing {sorted(missing)}")
        names = {c.name for c in self.columns}
        for column in (self.id_column, self.time_column, self.target):
            if column not in names:
                raise ValueError(f"column {column!r} is not in the column catalogue")

    @property
    def dtypes(self) -> dict[str, str]:
        return {c.name: c.dtype for c in self.columns}

    @property
    def model_features(self) -> list[str]:
        return [f["name"] for f in self.feature_definitions]


def load_spec(ref: str | None = None) -> DatasetSpec:
    """Import ``pkg.module:ATTR`` (or the PIPELINE_DATASET_SPEC variable) and return the spec."""
    ref = (ref or os.getenv(SPEC_ENV, "")).strip()
    if not ref:
        raise SpecNotConfigured(
            f"no dataset spec configured: pass --spec pkg.module:ATTR or set {SPEC_ENV}"
        )
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        raise SpecNotConfigured(f"spec reference {ref!r} must look like pkg.module:ATTR")
    try:
        module = importlib.import_module(module_name)
        spec = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise SpecNotConfigured(f"cannot load dataset spec {ref!r}: {exc}") from exc
    if not isinstance(spec, DatasetSpec):
        raise SpecNotConfigured(f"{ref} is not a DatasetSpec")
    return spec
