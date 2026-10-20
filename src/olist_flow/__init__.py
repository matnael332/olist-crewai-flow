"""CrewAI layer of the Olist late-delivery project (Flow, crews, tools, CLI)."""

import os

# Must run before any crewai import: telemetry and tracing are read when crewai is imported, and
# OpenMP reads the thread count when numpy/scikit-learn load.
for _key, _value in {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "OMP_NUM_THREADS": "2",
    "MPLBACKEND": "Agg",
}.items():
    os.environ.setdefault(_key, _value)

__version__ = "0.1.0"
