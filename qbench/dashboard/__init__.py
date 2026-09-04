"""Optional standalone Streamlit workbench."""
from pathlib import Path
import sys


def main():
    """Launch the installed dashboard without importing experiment code."""
    try:
        from streamlit.web import cli
    except ImportError:
        print('Install the dashboard extra: pip install "qbench[dashboard]"', file=sys.stderr)
        return 1
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    return cli.main()
