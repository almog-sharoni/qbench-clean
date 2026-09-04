"""Real-browser smoke against a disposable authenticated server and database.

Install the browser-test extra and run `python -m playwright install chromium`
first. This never starts a public listener or modifies a production account.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from zipfile import ZipFile

from playwright.sync_api import sync_playwright, expect

from qbench.dashboard.platform.store import Store


def login(page, name, password):
    page.get_by_label("Username", exact=True).fill(name)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/platform-browser")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    password, temporary, changed = (secrets.token_urlsafe(24) for _ in range(3))
    with tempfile.TemporaryDirectory(prefix="qbench-browser-") as directory:
        database = Path(directory) / "platform.sqlite3"
        store = Store.initialize(database, "owner", password)
        admin_token = store.login("owner", password)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        environment = dict(os.environ)
        environment.pop("QBENCH_DATASET_CATALOG", None)
        environment.pop("QBENCH_SINGLE_USER", None)
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(
                [sys.executable, "-c", "from qbench.dashboard import main; raise SystemExit(main())",
                 "--database", str(database), "--host", "127.0.0.1", "--port", str(port)],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 40
                while True:
                    try:
                        with urlopen(url + "/_stcore/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Disposable dashboard did not start; see server.log")
                    time.sleep(0.2)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    admin_context = browser.new_context(viewport={"width": 1440, "height": 1000})
                    user_context = browser.new_context(viewport={"width": 1440, "height": 1100})
                    admin, user = admin_context.new_page(), user_context.new_page()
                    for page in (admin, user):
                        page.set_default_timeout(60000)
                        page.goto(url)
                    login(admin, "owner", password)
                    admin.get_by_text("Administration", exact=True).click()
                    admin.get_by_role("tab", name="Create account", exact=True).click()
                    admin.get_by_label("New username", exact=True).fill("browser-user")
                    admin.get_by_label("Display name", exact=True).fill("Browser test user")
                    admin.get_by_label("Temporary password", exact=True).fill(temporary)
                    admin.get_by_role("button", name="Create account", exact=True).click()
                    expect(admin.get_by_text("Account created. The user must change the temporary password at first sign-in.")).to_be_visible()
                    login(user, "browser-user", temporary)
                    user.get_by_label("Current password", exact=True).fill(temporary)
                    user.get_by_label("New password", exact=True).fill(changed)
                    user.get_by_label("Confirm new password", exact=True).fill(changed)
                    user.get_by_role("button", name="Change password", exact=True).click()
                    login(user, "browser-user", changed)
                    expect(user.get_by_role("radio", name="Administration", exact=True)).to_have_count(0)
                    user.get_by_role("button", name="Inspect model", exact=True).click()
                    expect(user.get_by_text("Fully supported for captured scenarios", exact=True)).to_be_visible()
                    user.get_by_role("button", name="Build simulator", exact=True).click()
                    expect(user.get_by_text("Simulator built and verified.", exact=True)).to_be_visible()
                    user.get_by_role("button", name="Evaluate outputs", exact=True).click()
                    expect(user.get_by_text("Output MAE", exact=True)).to_be_visible()
                    user.get_by_role("button", name="Prepare artifact download", exact=True).click()
                    with user.expect_download() as pending:
                        user.get_by_role("button", name="Download artifacts ZIP", exact=True).click()
                    pending.value.save_as(output / "artifacts.zip")
                    with ZipFile(output / "artifacts.zip") as archive:
                        assert "evaluation.json" in archive.namelist()
                        assert "state.pt" not in archive.namelist()
                    user.screenshot(path=str(output / "evaluation-desktop.png"), full_page=True, animations="disabled")
                    user.get_by_test_id("stMain").evaluate("element => element.scrollTop = 0")
                    user.screenshot(path=str(output / "workbench-desktop.png"), full_page=True, animations="disabled")
                    user.get_by_test_id("stSidebar").hover()
                    user.get_by_test_id("stSidebarCollapseButton").get_by_role("button").click(timeout=10000)
                    expect(user.get_by_test_id("stSidebar")).to_have_attribute("aria-expanded", "false")
                    user.set_viewport_size({"width": 390, "height": 844})
                    user.wait_for_function("document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().right <= 0")
                    expect(user.get_by_role("heading", name="QBench", exact=True)).to_be_visible()
                    user.screenshot(path=str(output / "workbench-mobile.png"), full_page=True, animations="disabled")
                    user.set_viewport_size({"width": 1440, "height": 1100})
                    admin.get_by_role("tab", name="Users & feature access", exact=True).click()
                    admin.get_by_label("Manage user", exact=True).click()
                    admin.get_by_role("option", name="browser-user", exact=True).click()
                    admin.get_by_text("Account enabled", exact=True).click()
                    admin.get_by_role("button", name="Save access settings", exact=True).click()
                    expect(admin.get_by_label("Account enabled", exact=True)).not_to_be_checked()
                    user.get_by_role("button", name="Evaluate outputs", exact=True).click()
                    expect(user.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
                    login(user, "browser-user", changed)
                    expect(user.get_by_text("Sign-in failed. Check your credentials or try again in 15 minutes.")).to_be_visible()
                    admin.get_by_role("tab", name="Usage & audit", exact=True).click()
                    admin.screenshot(path=str(output / "administration.png"), full_page=True, animations="disabled")
                    totals = store.activity(admin_token, admin=True)["totals"]
                    actions = {row["action"] for row in totals if row["username"] == "browser-user" and row["status"] == "success"}
                    assert actions >= {"inspect", "convert", "evaluate", "downloads"}
                    browser.close()
                result = {"status": "passed", "browser": "chromium", "admin_account_creation": True,
                          "forced_password_change": True, "private_user_workflow": True,
                          "artifact_download": True, "account_disable_revokes_access": True,
                          "admin_usage_metrics": True, "desktop_and_mobile_screenshots": True}
                (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result))
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    main()
