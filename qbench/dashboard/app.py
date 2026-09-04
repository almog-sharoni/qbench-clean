"""Standalone workbench. Run with qbench-dashboard after installation."""
from pathlib import Path
import os
import runpy

import streamlit as st

st.set_page_config(page_title="QBench · Model Workbench", page_icon="🔬", layout="wide")
st.title("QBench")
st.caption("Inspect executed operations. Verify simulator routing. Measure output fidelity.")
st.info(
    "CPU inspection verifies routing with quantization disabled. "
    "A supported verdict is scoped to captured scenarios, not hardware fidelity."
)
if os.environ.get("QBENCH_SINGLE_USER") == "1":
    st.warning("Single-user local mode: authentication is disabled. Do not expose this mode to other users.")
    runpy.run_path(
        str(Path(__file__).with_name("workbench.py")),
        init_globals={"st": st, "tab_workbench": st.container()},
    )
else:
    from qbench.dashboard.platform.views import render

    render(st)
