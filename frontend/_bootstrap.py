"""Make the backend packages importable from the Streamlit frontend.

Streamlit adds only the entrypoint's folder, frontend/, to
sys.path. The dashboard also reuses the backend's scoring code (pulse_api, pulse_core), so the
backend folder is added here. Nothing from CrewAI is ever imported by the frontend.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
