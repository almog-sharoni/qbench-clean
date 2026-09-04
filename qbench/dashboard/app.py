"""Standalone workbench. Run with qbench-dashboard after installation."""
from pathlib import Path
import runpy

import streamlit as st

st.set_page_config(page_title="QBench · Model Workbench", page_icon="🔬", layout="wide")
st.title("QBench")
st.caption("Inspect executed operations. Verify simulator routing. Measure output fidelity.")
st.info(
    "CPU inspection verifies routing with quantization disabled. "
    "A supported verdict is scoped to captured scenarios, not hardware fidelity."
)
runpy.run_path(
    str(Path(__file__).with_name("workbench.py")),
    init_globals={"st": st, "tab_workbench": st.container()},
)
