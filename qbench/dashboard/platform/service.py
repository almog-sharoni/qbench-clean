"""Authorized action boundary for the shared workbench, independent of Streamlit."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json
import re
import tempfile
import threading
import time
import weakref
from zipfile import ZipFile, ZIP_DEFLATED

from .store import AccessDenied, PlatformError, Store


CATALOG = {"demo": ("tiny",), "torchvision": ("resnet18", "vit_b_16"), "timm": ("mobilevit_s",)}
FORMATS = ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")
_COMPUTE_LOCK = threading.Lock()
_WORKSPACES = weakref.WeakValueDictionary()


def load_dataset_catalog(path):
    """Load host-controlled data roots. Browser users can select IDs, not paths."""
    if not path:
        return {}
    try:
        configured = json.loads(Path(path).read_text())
        if not isinstance(configured, dict) or len(configured) > 32:
            raise ValueError()
        result = {}
        for name, directory in configured.items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or not isinstance(directory, str):
                raise ValueError()
            root = Path(directory).expanduser()
            if not root.is_absolute() or not root.is_dir():
                raise ValueError()
            result[name] = root.resolve()
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise PlatformError("Invalid host dataset catalog. Ask the server administrator to check its configuration.") from exc


class Busy(PlatformError):
    pass


@dataclass(frozen=True)
class ModelRequest:
    source: str = "demo"
    name: str = "tiny"
    image_size: int = 224
    batch_size: int = 1
    device: str = "cpu"
    pretrained: bool = False
    quantized: bool = False
    quantization_type: str = "fp8_e4m3"
    fallback: bool = False

    def validate(self):
        if self.name not in CATALOG.get(self.source, ()):
            raise PlatformError("Choose a model from the approved catalog.")
        if type(self.image_size) is not int or self.image_size not in (224, 256):
            raise PlatformError("The shared platform supports image sizes 224 and 256.")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 4:
            raise PlatformError("Batch size must be between 1 and 4.")
        if self.device not in {"cpu", "cuda"} or self.quantization_type not in FORMATS:
            raise PlatformError("Unsupported device or quantization format.")
        if any(type(value) is not bool for value in (self.pretrained, self.quantized, self.fallback)):
            raise PlatformError("Invalid model options.")
        if self.quantized and self.device != "cuda":
            raise PlatformError("Quantization-enabled runs require CUDA.")
        if self.pretrained and self.source == "demo":
            raise PlatformError("The demo does not use pretrained weights.")
        if self.pretrained and self.name == "vit_b_16" and self.image_size != 224:
            raise PlatformError("The approved pretrained ViT configuration requires 224 pixels.")


@dataclass(eq=False)
class Workspace:
    owner_id: str
    session_id: str
    request: ModelRequest | None = None
    provider: object = None
    inspection: object = None
    simulator: object = None
    evaluation: object = None

    def clear(self):
        self.request = self.provider = self.inspection = self.simulator = self.evaluation = None


class PlatformService:
    def __init__(self, store: Store, *, max_workspaces=4, datasets=None):
        self.store = store
        if type(max_workspaces) is not int or not 1 <= max_workspaces <= 32:
            raise PlatformError("Maximum loaded workspaces must be between 1 and 32.")
        self.max_workspaces = max_workspaces
        self.datasets = dict(datasets or {})

    def new_workspace(self, token):
        user = self.store.require(token)
        return Workspace(user.id, user.session_id)

    def _owned(self, user, workspace):
        if not isinstance(workspace, Workspace) or (workspace.owner_id, workspace.session_id) != (user.id, user.session_id):
            raise AccessDenied("This workspace belongs to another session.")

    @contextmanager
    def _action(self, token, action, workspace=None, *, compute=True):
        user = self.store.require(token)
        started = time.monotonic()
        acquired = False
        status = "success"
        try:
            self.store.require(token, action)
            if workspace is not None:
                self._owned(user, workspace)
            if compute:
                acquired = _COMPUTE_LOCK.acquire(blocking=False)
                if not acquired:
                    raise Busy("Another model operation is running. Please retry shortly.")
            yield user
            # Revocation or a feature edit during a long run must prevent its
            # result from being delivered to the old session.
            self.store.require(token, action)
        except AccessDenied:
            status = "denied"
            if isinstance(workspace, Workspace) and (workspace.owner_id, workspace.session_id) == (user.id, user.session_id):
                workspace.clear()
            raise
        except Busy:
            status = "busy"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            if acquired:
                _COMPUTE_LOCK.release()
            self.store._record_operation(user.id, action, status, (time.monotonic() - started) * 1000)

    def _request_access(self, token, request):
        request.validate()
        for needed, enabled in (
            ("vision", request.source != "demo"), ("pretrained", request.pretrained),
            ("gpu", request.device == "cuda"), ("quantization", request.quantized),
            ("fallback", request.fallback),
        ):
            if enabled:
                self.store.require(token, needed)

    @staticmethod
    def _provider(request):
        import torch
        from qbench import DirectObjectProvider, Scenario

        if request.device == "cuda" and not torch.cuda.is_available():
            raise PlatformError("CUDA is unavailable on this server. Select CPU routing inspection.")
        # Never import a user-supplied package, factory, checkpoint or pathname.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            if request.source == "demo":
                model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
                shapes = [(request.batch_size, 4)] * 2
            else:
                if request.source == "torchvision":
                    from torchvision import models

                    kwargs = {"image_size": request.image_size} if request.name == "vit_b_16" else {}
                    model = models.get_model(request.name, weights="DEFAULT" if request.pretrained else None, **kwargs)
                else:
                    import timm

                    model = timm.create_model(request.name, pretrained=request.pretrained)
                shapes = [(request.batch_size, 3, request.image_size, request.image_size)] * 2
            inputs = [torch.randn(shape).to(request.device) for shape in shapes]
        model = model.eval().to(request.device)
        scenarios = [Scenario(f"sample-{i + 1}", (value,)) for i, value in enumerate(inputs)]
        return DirectObjectProvider(model, scenarios, loader=scenarios)

    def inspect(self, token, workspace, request):
        from qbench import InspectionConfig, QuantizationPolicy, inspect_provider

        with self._action(token, "inspect", workspace) as user:
            self._request_access(token, request)
            for (database, session_id), loaded in list(_WORKSPACES.items()):
                if database == str(self.store.path) and not self.store._session_live(session_id):
                    loaded.clear()
            # At most one loaded model per session and a bounded process-wide
            # collection. Models are never cached or reused across users.
            active = sum(w.provider is not None and w is not workspace for w in list(_WORKSPACES.values()))
            if active >= self.max_workspaces:
                raise Busy("Loaded-workspace capacity reached. Release an unused workspace and retry.")
            workspace.clear()
            provider = self._provider(request)
            config = InspectionConfig(
                device=request.device, quantization_enabled=request.quantized,
                allow_fp32_fallback=request.fallback, capture_callsites=False,
                enable_fx=False, enable_export=False,
                quantization_policy=QuantizationPolicy(quantization_type=request.quantization_type),
            )
            result = inspect_provider(provider, config)
            self.store.require(token, "inspect")
            self._request_access(token, request)
            workspace.request, workspace.provider, workspace.inspection = request, provider, result
            _WORKSPACES[(str(self.store.path), user.session_id)] = workspace
        return result

    def convert(self, token, workspace):
        from qbench import build_simulator

        with self._action(token, "convert", workspace):
            if workspace.inspection is None:
                raise PlatformError("Inspect a model before building a simulator.")
            self._request_access(token, workspace.request)
            result = workspace.inspection
            if not result.fully_supported and not workspace.request.fallback:
                raise PlatformError("Strict conversion is unavailable: inspect the support gaps first.")
            simulator = build_simulator(workspace.provider.build_model(), result.plan, strict=not workspace.request.fallback)
            verification = simulator.verify(workspace.provider.capture_scenarios())
            if not verification.succeeded:
                raise PlatformError("Converted verification failed. No simulator was saved.")
            self.store.require(token, "convert")
            workspace.simulator, workspace.evaluation = simulator, None
        return verification

    def _dataset_provider(self, workspace, dataset_name, max_samples):
        import torch
        from importlib.resources import files
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, Subset
        from qbench import DirectObjectProvider

        if dataset_name not in self.datasets:
            raise PlatformError("Choose an approved dataset ID; filesystem paths are not accepted.")
        if workspace.request.source == "demo":
            raise PlatformError("ImageNet evaluation requires an approved vision model.")
        if type(max_samples) is not int or not 1 <= max_samples <= 64:
            raise PlatformError("Shared dataset evaluations are limited to 1–64 samples.")
        spec = workspace.request
        if spec.source == "torchvision":
            from torchvision.models import get_model_weights

            transform = get_model_weights(spec.name).DEFAULT.transforms(
                crop_size=spec.image_size, resize_size=round(spec.image_size / 0.875),
            )
        else:
            from timm.data import create_transform, resolve_model_data_config

            config = resolve_model_data_config(workspace.provider.model)
            config["input_size"] = (3, spec.image_size, spec.image_size)
            transform = create_transform(**config, is_training=False)
        dataset = ImageFolder(str(self.datasets[dataset_name]), transform=transform)
        index = json.loads(files("qbench.data").joinpath("imagenet_class_index.json").read_text())
        canonical = {record[0]: int(number) for number, record in index.items()}
        if any(name not in canonical for name in dataset.classes):
            raise PlatformError("Approved ImageNet datasets must use canonical WordNet-ID class folders.")
        mapping = {local: canonical[name] for name, local in dataset.class_to_idx.items()}
        dataset.target_transform = mapping.__getitem__
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7))[:max_samples].tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=spec.batch_size, num_workers=0)

        def prepared_batches():
            for inputs, targets in loader:
                yield inputs.to(spec.device), targets.to(spec.device)

        return DirectObjectProvider(workspace.provider.model, workspace.provider.scenarios, loader=prepared_batches())

    def evaluate(self, token, workspace, *, detailed=False, dataset_name=None, max_samples=16):
        from qbench import EvaluationConfig, evaluate

        with self._action(token, "evaluate", workspace):
            if type(detailed) is not bool:
                raise PlatformError("Invalid evaluation options.")
            if detailed:
                self.store.require(token, "detailed")
            if dataset_name is not None:
                self.store.require(token, "datasets")
            if workspace.simulator is None:
                raise PlatformError("Build a verified simulator before evaluation.")
            self._request_access(token, workspace.request)
            provider = workspace.provider if dataset_name is None else self._dataset_provider(workspace, dataset_name, max_samples)
            report = evaluate(
                workspace.provider.build_model(), workspace.simulator, provider,
                EvaluationConfig(metrics="detailed" if detailed else "fast", max_batches=2 if dataset_name is None else 64,
                                 task="generic" if dataset_name is None else "classification", retain_activations=False),
            )
            self.store.require(token, "evaluate")
            if detailed:
                self.store.require(token, "detailed")
            if dataset_name is not None:
                self.store.require(token, "datasets")
                report.details["approved_dataset"] = dataset_name
            workspace.evaluation = report
        return report

    def download(self, token, workspace):
        from qbench.artifacts import write_artifacts

        with self._action(token, "downloads", workspace):
            if workspace.inspection is None:
                raise PlatformError("Inspect a model before exporting artifacts.")
            self._request_access(token, workspace.request)
            with tempfile.TemporaryDirectory(prefix="qbench-export-") as directory:
                write_artifacts(directory, workspace.inspection, evaluation=workspace.evaluation)
                output = BytesIO()
                with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
                    for path in sorted(Path(directory).iterdir()):
                        archive.write(path, arcname=path.name)
            payload = output.getvalue()
        return payload

    def release(self, token, workspace):
        user = self.store.require(token)
        self._owned(user, workspace)
        if not _COMPUTE_LOCK.acquire(blocking=False):
            raise Busy("Wait for the current operation before releasing a workspace.")
        try:
            workspace.clear()
        finally:
            _COMPUTE_LOCK.release()
