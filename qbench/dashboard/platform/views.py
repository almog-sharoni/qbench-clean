"""Streamlit presentation; authorization lives in Store and PlatformService."""
from __future__ import annotations

from datetime import datetime, timezone
import os

from .service import CATALOG, FORMATS, ModelRequest, PlatformService, load_dataset_catalog
from .store import AccessDenied, DEFAULT_FEATURES, FEATURES, PlatformError, Store


def _clear(st, *, token=None, notice=None):
    for key in list(st.session_state):
        del st.session_state[key]
    if token:
        st.session_state["_platform_token"] = token
    if notice:
        st.session_state["_platform_notice"] = notice


def _utc(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if value else "Never"


def _login(st, store):
    st.subheader("Sign in to your workspace")
    st.caption("Accounts are created by your administrator. Refreshing the browser requires a new sign-in.")
    with st.form("platform_login", clear_on_submit=True):
        username = st.text_input("Username", max_chars=64, key="login_username")
        password = st.text_input("Password", type="password", max_chars=128, key="login_password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        try:
            token = store.login(username, password)
        except AccessDenied as exc:
            st.error(str(exc))
        except Exception:
            st.error("Sign-in is unavailable. Contact the host administrator.")
        else:
            _clear(st, token=token)
            st.rerun()


def _password_form(st, store, token, *, required=False):
    st.subheader("Change your password")
    if required:
        st.warning("Your account has a temporary password. Set a new password before accessing the dashboard.")
    with st.form("platform_password", clear_on_submit=True):
        current = st.text_input("Current password", type="password", max_chars=128)
        new = st.text_input("New password", type="password", max_chars=128, help="Use 12–128 characters.")
        confirm = st.text_input("Confirm new password", type="password", max_chars=128)
        submitted = st.form_submit_button("Change password", type="primary")
    if submitted:
        try:
            if new != confirm:
                raise PlatformError("New passwords do not match.")
            store.change_password(token, current, new)
        except PlatformError as exc:
            st.error(str(exc))
        else:
            _clear(st, notice="Password changed. All sessions were signed out. Sign in with your new password.")
            st.rerun()


def _run(st, label, call):
    try:
        with st.spinner(label):
            result = call()
    except AccessDenied:
        raise
    except PlatformError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:
        # Never render model/provider exception contents, which may contain
        # private tensor data, local paths or secret-bearing library messages.
        st.error(f"Operation failed ({type(exc).__name__}). Review your settings or contact the administrator.")
        return None
    return result


def _workbench(st, service, token, user):
    workspace = st.session_state.get("_platform_workspace")
    if workspace is None:
        workspace = service.new_workspace(token)
        st.session_state["_platform_workspace"] = workspace
    st.subheader("Model workbench")
    st.caption("1 · Inspect executed paths  →  2 · Build a verified simulator  →  3 · Compare outputs")
    with st.container(border=True):
        st.markdown("**1. Choose a model and inspect**")
        sources = list(CATALOG) if "vision" in user.features else ["demo"]
        left, right = st.columns(2)
        source = left.selectbox("Model source", sources, key="model_source")
        name = right.selectbox("Model", CATALOG[source], key=f"model_name_{source}")
        left, middle, right = st.columns(3)
        size = left.selectbox("Image size", [224, 256], disabled=source == "demo")
        batch = middle.number_input("Batch size", min_value=1, max_value=4, value=1, step=1)
        device = right.selectbox("Device", ["cpu", "cuda"] if "gpu" in user.features else ["cpu"])
        pretrained = st.checkbox("Use pretrained weights", disabled=source == "demo" or "pretrained" not in user.features)
        quantized = st.checkbox("Enable quantized execution", disabled=device != "cuda" or "quantization" not in user.features)
        quantization_type = st.selectbox("Quantization format", FORMATS, disabled=not quantized)
        fallback = st.checkbox("Allow partial FP32 fallback", disabled="fallback" not in user.features)
        # Disabled widgets can retain their prior value after another selector
        # changes; normalize before constructing a request, and still validate
        # everything again at the service boundary.
        pretrained = bool(pretrained and source != "demo" and "pretrained" in user.features)
        quantized = bool(quantized and device == "cuda" and "quantization" in user.features)
        fallback = bool(fallback and "fallback" in user.features)
        request = ModelRequest(source, name, int(size), int(batch), device, pretrained, quantized, quantization_type, fallback)
        st.info("The demo needs no downloads. Evaluation uses two deterministic synthetic batches, not a labeled accuracy dataset.")
        if st.button("Inspect model", type="primary", key="inspect_model"):
            st.session_state.pop("_platform_download", None)
            result = _run(st, "Capturing operations and verifying simulator routes…", lambda: service.inspect(token, workspace, request))
            if result is not None:
                st.success("Inspection complete. Review the qualified verdict below.")
    if workspace.inspection is None:
        st.caption("Your results are private to this signed-in browser session. Start with the demo above.")
        return
    changed = request != workspace.request
    if changed:
        st.warning("Settings changed. These results belong to your previous inspection; inspect again before continuing.")
    result = workspace.inspection
    support = result.support
    if result.fully_supported:
        st.success("Fully supported for captured scenarios")
    else:
        st.warning("Partial or unsupported for captured scenarios — review the gaps before conversion.")
    a, b, c, d = st.columns(4)
    a.metric("Capture", "Complete" if support.get("capture_complete") else "Incomplete")
    b.metric("Routing", "Verified" if support.get("strict_realization") else "Partial")
    c.metric("Quantization", "Verified" if support.get("quantized_execution_verified") else "Not verified")
    d.metric("Hardware fidelity", support.get("hardware_fidelity", {}).get("status", "missing_evidence").replace("_", " "))
    st.caption(f"{support.get('operation_count', 0)} captured operations · model: {workspace.request.name} · device: {workspace.request.device}")
    gaps_tab, modules_tab, scenarios_tab = st.tabs(["Support gaps", "Module summary", "Scenario coverage"])
    with gaps_tab:
        gaps = support.get("gaps", [])
        if gaps:
            search = st.text_input("Filter gaps", key="gap_filter").casefold()
            visible = [row for row in gaps if search in str(row).casefold()]
            st.dataframe(visible, width="stretch", hide_index=True)
        else:
            st.success("No unsupported executed operations were found in these scenarios.")
    with modules_tab:
        rows = [{"Module": row["path"] or "(root)", "Type": row["type"], "Status": row["status"],
                 "Operations": row["operation_count"]} for row in support.get("module_summary", [])]
        st.dataframe(rows, width="stretch", hide_index=True)
        if support.get("not_assessed_modules"):
            st.warning("Some modules did not execute and are not assessed.")
            st.json(support["not_assessed_modules"])
    with scenarios_tab:
        st.json(support.get("scenario_coverage", {}))
    with st.container(border=True):
        st.markdown("**2. Build the simulator**")
        allowed = "convert" in user.features
        if not allowed:
            st.caption("Simulator conversion is disabled for your account.")
        if st.button("Build simulator", disabled=not allowed or changed, key="build_simulator"):
            st.session_state.pop("_platform_download", None)
            verification = _run(st, "Building and auditing the simulator clone…", lambda: service.convert(token, workspace))
            if verification is not None:
                st.success("Simulator built and verified.")
    with st.container(border=True):
        st.markdown("**3. Compare outputs**")
        evaluation_sources = ["Synthetic samples"]
        if "datasets" in user.features:
            evaluation_sources += list(service.datasets)
        evaluation_source = st.selectbox("Evaluation data", evaluation_sources)
        dataset_name = None if evaluation_source == "Synthetic samples" else evaluation_source
        max_samples = st.selectbox("Dataset sample limit", [16, 32, 64], disabled=dataset_name is None)
        if dataset_name is not None:
            st.caption("Approved ImageNet subset · deterministic sample seed 7 · model-specific preprocessing.")
        detailed = st.checkbox("Detailed diagnostics", disabled="detailed" not in user.features)
        enabled = "evaluate" in user.features and workspace.simulator is not None and not changed
        if st.button("Evaluate outputs", disabled=not enabled, key="evaluate_outputs"):
            st.session_state.pop("_platform_download", None)
            report = _run(st, "Running paired reference and simulator evaluation…",
                          lambda: service.evaluate(token, workspace, detailed=bool(detailed and "detailed" in user.features),
                                                   dataset_name=dataset_name, max_samples=max_samples))
            if report is not None:
                st.success("Evaluation complete. " + ("ImageNet classification metrics are included." if dataset_name else "These are output comparisons, not task accuracy scores."))
        if workspace.evaluation is not None:
            report = workspace.evaluation
            st.caption("Displayed evaluation data: " + report.details.get("approved_dataset", "synthetic samples"))
            a, b, c = st.columns(3)
            a.metric("Output MAE", f"{report.metrics.get('mae', 0):.6g}")
            b.metric("Output MSE", f"{report.metrics.get('mse', 0):.6g}")
            c.metric("Batches", report.batches)
            with st.expander("All evaluation metrics"):
                st.json(report.to_dict())
    if "downloads" in user.features:
        if st.button("Prepare artifact download", key="prepare_download"):
            payload = _run(st, "Preparing your artifact bundle…", lambda: service.download(token, workspace))
            if payload is not None:
                st.session_state["_platform_download"] = payload
        payload = st.session_state.get("_platform_download")
        if payload is not None:
            service.store.require(token, "downloads")
            st.download_button("Download artifacts ZIP", payload, file_name="qbench-artifacts.zip", mime="application/zip")
    with st.expander("Advanced support report"):
        st.json(support)
    if st.button("Release workspace", key="release_workspace"):
        try:
            service.release(token, workspace)
        except PlatformError as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("_platform_download", None)
            st.rerun()


def _activity(st, store, token, *, admin=False):
    days = st.selectbox("Activity window (days)", [7, 30, 90], index=1, key=f"activity_days_{admin}")
    data = store.activity(token, days=days, admin=admin)
    compute_actions = {"inspect", "convert", "evaluate", "downloads"}
    rows = [row for row in data["totals"] if row["action"] in compute_actions]
    a, b, c = st.columns(3)
    a.metric("Successful operations", sum(row["count"] for row in rows if row["status"] == "success"))
    b.metric("Failed / denied operations", sum(row["count"] for row in rows if row["status"] in {"failed", "denied"}))
    c.metric("Operation time (seconds)", f"{sum(row['total_ms'] for row in rows) / 1000:.1f}")
    st.caption("Operation wall time includes setup and waiting within an action; it is not GPU utilization or model throughput.")
    st.dataframe(data["totals"], width="stretch", hide_index=True)
    st.markdown("**Recent activity**")
    st.dataframe([{**row, "at": _utc(row["at"])} for row in data["events"]], width="stretch", hide_index=True)
    st.caption("Audit data is retained for up to 90 days (pruned on sign-in). Latest 200 events shown; totals include the full selected window. No passwords or tensor values are recorded.")


def _admin(st, store, token):
    store.require(token, admin=True)
    st.subheader("Administration")
    users_tab, create_tab, metrics_tab = st.tabs(["Users & feature access", "Create account", "Usage & audit"])
    with users_tab:
        users = store.users(token)
        a, b, c = st.columns(3)
        a.metric("Total accounts", len(users))
        b.metric("Enabled accounts", sum(bool(row["enabled"]) for row in users))
        c.metric("Enabled administrators", sum(row["role"] == "admin" and bool(row["enabled"]) for row in users))
        st.dataframe([{k: (_utc(v) if k in {"created_at", "last_login"} else v) for k, v in row.items()
                       if k not in {"id", "features"}} for row in users], width="stretch", hide_index=True)
        selected = st.selectbox("Manage user", [row["username"] for row in users])
        target = next(row for row in users if row["username"] == selected)
        with st.form(f"manage_{target['id']}"):
            role = st.selectbox("Role", ["user", "admin"], index=int(target["role"] == "admin"))
            enabled = st.checkbox("Account enabled", value=bool(target["enabled"]))
            features = st.multiselect("Enabled features", list(FEATURES), default=target["features"], format_func=lambda key: FEATURES[key])
            submitted = st.form_submit_button("Save access settings", type="primary")
        st.caption("Saving access settings revokes every session for this user. Feature dependencies are validated server-side.")
        if submitted:
            try:
                store.update_user(token, target["id"], role=role, enabled=enabled, features=features)
            except PlatformError as exc:
                st.error(str(exc))
            else:
                st.success("Settings saved and sessions revoked.")
                st.rerun()
        if st.button("Revoke user sessions"):
            store.revoke_sessions(token, target["id"])
            st.success("User sessions revoked.")
        with st.form(f"reset_{target['id']}", clear_on_submit=True):
            password = st.text_input("New temporary password", type="password", max_chars=128)
            confirmed = st.checkbox("Confirm password reset")
            reset = st.form_submit_button("Reset password")
        if reset:
            try:
                if not confirmed:
                    raise PlatformError("Confirm the password reset first.")
                store.reset_password(token, target["id"], password)
            except PlatformError as exc:
                st.error(str(exc))
            else:
                st.success("Password reset. All sessions revoked; password change is required at sign-in.")
    with create_tab:
        with st.form("create_user", clear_on_submit=True):
            username = st.text_input("New username", max_chars=64)
            display_name = st.text_input("Display name", max_chars=80)
            password = st.text_input("Temporary password", type="password", max_chars=128, help="12–128 characters. Share through a trusted channel.")
            role = st.selectbox("New account role", ["user", "admin"])
            features = st.multiselect("Initial features", list(FEATURES), default=sorted(DEFAULT_FEATURES), format_func=lambda key: FEATURES[key])
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            try:
                store.create_user(token, username, password, display_name=display_name, role=role, features=features)
            except PlatformError as exc:
                st.error(str(exc))
            else:
                st.session_state["_platform_notice"] = "Account created. The user must change the temporary password at first sign-in."
                st.rerun()
    with metrics_tab:
        _activity(st, store, token, admin=True)


def render(st):
    database = os.environ.get("QBENCH_PLATFORM_DB")
    if not database:
        st.info("Multi-user setup required. Initialize a private database, then start the dashboard with --database.")
        st.code("qbench-admin init --database /private/path/platform.sqlite3 --username owner\nqbench-dashboard --database /private/path/platform.sqlite3", language="bash")
        return
    try:
        store = Store(database)
        service = PlatformService(store, max_workspaces=int(os.environ.get("QBENCH_MAX_WORKSPACES", "4")),
                                  datasets=load_dataset_catalog(os.environ.get("QBENCH_DATASET_CATALOG")))
    except (PlatformError, ValueError) as exc:
        st.error(str(exc))
        return
    notice = st.session_state.pop("_platform_notice", None)
    if notice:
        st.info(notice)
    token = st.session_state.get("_platform_token")
    if not token:
        _login(st, store)
        return
    try:
        user = store.me(token)
    except AccessDenied as exc:
        _clear(st, notice=str(exc))
        st.rerun()
        return
    except Exception:
        _clear(st)
        st.error("Account service is unavailable. Contact the host administrator.")
        return
    st.sidebar.text(user.display_name)
    st.sidebar.caption(f"{user.username} · {user.role}")
    if st.sidebar.button("Sign out"):
        store.logout(token)
        _clear(st, notice="Signed out. Your in-memory workspace has been released.")
        st.rerun()
    if user.must_change_password:
        _password_form(st, store, token, required=True)
        return
    pages = (["Workbench"] if "inspect" in user.features else []) + ["My activity", "Account"]
    if user.role == "admin":
        pages.append("Administration")
    page = st.sidebar.radio("Navigation", pages)
    st.sidebar.caption("Feature permissions are checked again for every server operation.")
    try:
        if page == "Workbench":
            _workbench(st, service, token, user)
        elif page == "My activity":
            st.subheader("My activity")
            _activity(st, store, token)
        elif page == "Administration":
            _admin(st, store, token)
        else:
            st.subheader("Your account")
            st.write("Enabled features: " + ", ".join(FEATURES[key] for key in sorted(user.features)))
            _password_form(st, store, token)
    except AccessDenied as exc:
        _clear(st, notice=str(exc))
        st.rerun()
    except Exception:
        st.error("This page is unavailable. Contact the host administrator.")
