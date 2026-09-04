"""Optional standalone Streamlit workbench."""
from pathlib import Path
import argparse
import os
import sys


def main():
    """Launch authenticated mode by default; local legacy mode is explicit."""
    parser = argparse.ArgumentParser(prog="qbench-dashboard")
    parser.add_argument("--database", default=os.environ.get("QBENCH_PLATFORM_DB"))
    parser.add_argument("--host", "--server.address", default="127.0.0.1")
    parser.add_argument("--port", "--server.port", type=int, default=8501)
    parser.add_argument("--single-user", action="store_true", help="unauthenticated legacy UI, loopback only")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if args.single_user and args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Single-user mode may only bind to loopback")
    if not args.single_user and not args.database:
        print("Initialize with qbench-admin init, then pass --database to qbench-dashboard.", file=sys.stderr)
        return 1
    if not args.single_user:
        from .platform.store import PlatformError, Store

        try:
            Store(args.database)
        except (PlatformError, OSError) as exc:
            print(str(exc) if isinstance(exc, PlatformError) else "Platform database is unavailable.", file=sys.stderr)
            return 1
    try:
        from streamlit.web import cli
    except ImportError:
        print('Install the dashboard extra: pip install "qbench[dashboard]"', file=sys.stderr)
        return 1
    os.environ["QBENCH_SINGLE_USER"] = "1" if args.single_user else "0"
    if args.database:
        os.environ["QBENCH_PLATFORM_DB"] = str(Path(args.database).expanduser().absolute())
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")),
                "--server.address", args.host, "--server.port", str(args.port),
                "--server.headless", "true", "--server.enableXsrfProtection", "true",
                "--server.enableCORS", "true", "--browser.gatherUsageStats", "false",
                "--client.showErrorDetails", "none"]
    return cli.main()
