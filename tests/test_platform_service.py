from dataclasses import replace
import hashlib
from io import BytesIO
import json
from zipfile import ZipFile

import pytest

from qbench.dashboard.platform.service import Busy, ModelRequest, PlatformService, _COMPUTE_LOCK, _WORKSPACES
from qbench.dashboard.platform.store import AccessDenied, PlatformError
from test_platform_store import accounts, add_user, USER_PASSWORD  # noqa: F401


@pytest.fixture
def platform(accounts):
    store, admin, _ = accounts
    uid, token = add_user(store, admin)
    service = PlatformService(store)
    return store, admin, uid, token, service, service.new_workspace(token)


def test_guided_workflow_downloads_and_two_forward_metrics(platform):
    store, admin, uid, token, service, workspace = platform
    result = service.inspect(token, workspace, ModelRequest())
    assert result.fully_supported
    assert service.convert(token, workspace).succeeded
    report = service.evaluate(token, workspace)
    assert report.batches == report.reference_forwards == report.simulator_forwards == 2
    assert report.metrics["mae"] == pytest.approx(0)
    archive = ZipFile(BytesIO(service.download(token, workspace)))
    assert set(archive.namelist()) == {"manifest.json", "support.json", "plan.json", "operations.jsonl.gz", "evaluation.json"}
    manifest = json.loads(archive.read("manifest.json"))
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == metadata["sha256"]
    totals = store.activity(admin, admin=True)["totals"]
    assert {row["action"] for row in totals if row["username"] == "researcher"} >= {"inspect", "convert", "evaluate", "downloads"}
    service.release(token, workspace)
    assert workspace.provider is workspace.inspection is None


def test_user_and_session_workspace_isolation(platform):
    store, admin, _, token, service, workspace = platform
    service.inspect(token, workspace, ModelRequest())
    _, other = add_user(store, admin, "other-user")
    for operation in (
        lambda: service.inspect(other, workspace, ModelRequest()),
        lambda: service.convert(other, workspace),
        lambda: service.download(other, workspace),
        lambda: service.release(other, workspace),
    ):
        with pytest.raises(AccessDenied):
            operation()
    assert workspace.inspection is not None
    same_user_other_session = store.login("researcher", USER_PASSWORD)
    with pytest.raises(AccessDenied):
        service.download(same_user_other_session, workspace)
    own = service.new_workspace(other)
    service.inspect(other, own, ModelRequest())
    one = next(workspace.provider.model.parameters())
    two = next(own.provider.model.parameters())
    assert one.data_ptr() != two.data_ptr()


@pytest.mark.parametrize("model_request", [
    ModelRequest(source="torchvision", name="resnet18"),
    ModelRequest(device="cuda"),
    ModelRequest(device="cuda", quantized=True),
    ModelRequest(fallback=True),
])
def test_disabled_features_are_denied_before_model_execution(platform, model_request, monkeypatch):
    _, _, _, token, service, workspace = platform
    called = []
    monkeypatch.setattr(service, "_provider", lambda spec: called.append(spec))
    with pytest.raises(AccessDenied):
        service.inspect(token, workspace, model_request)
    assert called == []


@pytest.mark.parametrize("changes", [
    {"source": "custom", "name": "os:system"}, {"name": "../../secret.pt"},
    {"batch_size": 999}, {"batch_size": True}, {"image_size": 1024},
    {"device": "cuda:7"}, {"quantized": True}, {"pretrained": "false"},
])
def test_catalog_and_resource_validation(platform, changes):
    _, _, _, token, service, workspace = platform
    with pytest.raises(PlatformError):
        service.inspect(token, workspace, replace(ModelRequest(), **changes))
    assert workspace.provider is None


def test_authorization_not_only_ui_and_revoked_results_cannot_be_downloaded(platform):
    store, admin, uid, token, service, workspace = platform
    service.inspect(token, workspace, ModelRequest())
    store.update_user(admin, uid, role="user", enabled=True, features={"inspect"})
    with pytest.raises(AccessDenied):
        service.download(token, workspace)
    fresh = store.login("researcher", USER_PASSWORD)
    clean = service.new_workspace(fresh)
    service.inspect(fresh, clean, ModelRequest())
    for call in (lambda: service.convert(fresh, clean), lambda: service.evaluate(fresh, clean), lambda: service.download(fresh, clean)):
        with pytest.raises(AccessDenied):
            call()


def test_detailed_metrics_need_separate_permission(platform):
    _, _, _, token, service, workspace = platform
    service.inspect(token, workspace, ModelRequest())
    service.convert(token, workspace)
    with pytest.raises(AccessDenied):
        service.evaluate(token, workspace, detailed=True)


def test_compute_serialization_error_audit_and_lock_release(platform, monkeypatch):
    store, admin, _, token, service, workspace = platform
    assert _COMPUTE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(Busy):
            service.inspect(token, workspace, ModelRequest())
    finally:
        _COMPUTE_LOCK.release()
    def bad_provider(_):
        raise RuntimeError("PRIVATE_TENSOR_CONTENTS_AND_PASSWORD")
    monkeypatch.setattr(service, "_provider", bad_provider)
    with pytest.raises(RuntimeError):
        service.inspect(token, workspace, ModelRequest())
    assert _COMPUTE_LOCK.acquire(blocking=False)
    _COMPUTE_LOCK.release()
    activity = store.activity(admin, admin=True)
    assert "PRIVATE_TENSOR_CONTENTS_AND_PASSWORD" not in json.dumps(activity)
    assert any(row["status"] == "failed" and row["action"] == "inspect" for row in activity["events"])


def test_revocation_during_job_prevents_result_delivery(platform, monkeypatch):
    store, admin, uid, token, service, workspace = platform
    provider = service._provider
    def revoking(request):
        result = provider(request)
        store.revoke_sessions(admin, uid)
        return result
    monkeypatch.setattr(service, "_provider", revoking)
    with pytest.raises(AccessDenied):
        service.inspect(token, workspace, ModelRequest())
    assert workspace.inspection is workspace.provider is None


def test_loaded_workspace_limit_and_reclamation(platform):
    store, admin, _, token, service, workspace = platform
    _WORKSPACES.clear()
    service.max_workspaces = 1
    service.inspect(token, workspace, ModelRequest())
    _, other = add_user(store, admin, "other-user")
    another = service.new_workspace(other)
    with pytest.raises(Busy, match="capacity"):
        service.inspect(other, another, ModelRequest())
    store.logout(token)
    service.inspect(other, another, ModelRequest())
    assert another.inspection.fully_supported and workspace.provider is None
