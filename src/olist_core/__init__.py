"""Pure data and ML library for the Olist late-delivery project.

This package never imports CrewAI, so the Streamlit app, the Flask API and the tests can use it
without the LLM stack.
"""

import os

# Thread defaults must be set before numpy/scikit-learn load their OpenMP runtimes.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "0.1.0"
