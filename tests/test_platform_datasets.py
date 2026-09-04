from dataclasses import replace
import json

import pytest
import torch

from qbench.dashboard.platform.service import ModelRequest, PlatformService, load_dataset_catalog
from qbench.dashboard.platform.store import AccessDenied, DEFAULT_FEATURES, PlatformError
from test_platform_store import accounts, add_user  # noqa: F401


def test_host_catalog_is_validated_and_never_accepts_browser_paths(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    catalog = tmp_path / "datasets.json"
    catalog.write_text(json.dumps({"imagenet-subset": str(root)}))
    assert load_dataset_catalog(catalog) == {"imagenet-subset": root.resolve()}
    for value in ([str(root)], {"../private": str(root)}, {"set": "relative/path"}, {"set": "/does-not-exist"}):
        catalog.write_text(json.dumps(value))
        with pytest.raises(PlatformError, match="Invalid host dataset catalog"):
            load_dataset_catalog(catalog)
    assert load_dataset_catalog(None) == {}


def test_approved_imagenet_evaluation_uses_labels_and_bounded_loader(accounts, tmp_path):
    from PIL import Image

    store, admin, _ = accounts
    _, token = add_user(store, admin, features=DEFAULT_FEATURES | {"vision", "datasets"})
    root = tmp_path / "imagenet"
    (root / "n01443537").mkdir(parents=True)
    Image.new("RGB", (256, 256), color=(10, 20, 30)).save(root / "n01443537" / "sample.png")
    service = PlatformService(store, datasets={"approved": root})
    workspace = service.new_workspace(token)
    spec = ModelRequest(source="torchvision", name="resnet18")
    assert service.inspect(token, workspace, spec).fully_supported
    service.convert(token, workspace)
    provider = service._dataset_provider(workspace, "approved", 1)
    inputs, target = next(iter(provider.loader))
    assert inputs.shape == (1, 3, 224, 224)
    assert target.tolist() == [1]
    report = service.evaluate(token, workspace, dataset_name="approved", max_samples=1)
    assert report.batches == 1
    assert report.details["approved_dataset"] == "approved"
    assert report.metrics["mae"] == pytest.approx(0)
    assert any("top1" in name for name in report.metrics)
    with pytest.raises(PlatformError, match="approved dataset ID"):
        service.evaluate(token, workspace, dataset_name=str(root))
    with pytest.raises(PlatformError, match="1–64"):
        service.evaluate(token, workspace, dataset_name="approved", max_samples=10000)


def test_dataset_permission_is_checked_before_filesystem_access(accounts, monkeypatch):
    store, admin, _ = accounts
    _, token = add_user(store, admin)
    service = PlatformService(store)
    workspace = service.new_workspace(token)
    service.inspect(token, workspace, ModelRequest())
    service.convert(token, workspace)
    calls = []
    monkeypatch.setattr(service, "_dataset_provider", lambda *args: calls.append(args))
    with pytest.raises(AccessDenied):
        service.evaluate(token, workspace, dataset_name="/etc/passwd")
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU platform integration")
def test_quantization_enabled_platform_workflow(accounts):
    store, admin, _ = accounts
    service = PlatformService(store)
    workspace = service.new_workspace(admin)
    result = service.inspect(admin, workspace, ModelRequest(device="cuda", quantized=True))
    assert result.fully_supported and result.support["quantized_execution_verified"]
    assert service.convert(admin, workspace).succeeded
    report = service.evaluate(admin, workspace)
    assert report.reference_forwards == report.simulator_forwards == 2
    assert report.metrics["mae"] is not None
