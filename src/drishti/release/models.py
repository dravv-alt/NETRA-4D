"""Explicit model registry used by every supported NETRA release path."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RegisteredModel:
    role: str
    path: Path
    provider: str
    required_classes: tuple[str, ...]
    release_approved: bool
    purpose: str

    def provenance(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "provider": self.provider,
            "required_classes": list(self.required_classes),
            "release_approved": self.release_approved,
            "purpose": self.purpose,
        }


class ModelRegistry:
    """Loads, validates, and fingerprints only explicitly configured weights."""

    def __init__(self, registry_path: Path | str, project_root: Path | str | None = None):
        self.registry_path = Path(registry_path).resolve()
        self.project_root = Path(project_root or self.registry_path.parent.parent).resolve()
        if not self.registry_path.is_file():
            raise FileNotFoundError(f"Model registry is missing: {self.registry_path}")
        payload = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        if payload.get("schema_version") != "1.0" or not isinstance(payload.get("models"), dict):
            raise ValueError("model registry must contain schema_version=1.0 and a models mapping")
        self._models: dict[str, RegisteredModel] = {}
        for role, value in payload["models"].items():
            path = Path(str(value["path"]))
            if not path.is_absolute():
                path = self.project_root / path
            self._models[role] = RegisteredModel(
                role=role,
                path=path.resolve(),
                provider=str(value["provider"]),
                required_classes=tuple(str(item) for item in value.get("required_classes", [])),
                release_approved=bool(value.get("release_approved", False)),
                purpose=str(value.get("purpose", "")),
            )

    def get(self, role: str, *, require_approved: bool = True) -> RegisteredModel:
        try:
            model = self._models[role]
        except KeyError as error:
            raise KeyError(f"Unknown model role {role!r}; configured roles: {sorted(self._models)}") from error
        if require_approved and not model.release_approved:
            raise PermissionError(
                f"Model role {role!r} is blocked from release use: {model.purpose}. "
                "Add held-out evaluation evidence and explicitly approve it in the registry first."
            )
        if not model.path.is_file():
            raise FileNotFoundError(f"Configured model for {role!r} is missing: {model.path}")
        return model

    def validate(self, *, include_unapproved: bool = False) -> dict[str, Any]:
        """Validate files and class names without running video inference."""
        from ultralytics import YOLO

        report: dict[str, Any] = {"schema_version": "1.0", "registry": str(self.registry_path), "models": {}}
        for role in sorted(self._models):
            configured = self._models[role]
            if not include_unapproved and not configured.release_approved:
                report["models"][role] = {
                    "path": str(configured.path), "release_approved": False,
                    "validation": "BLOCKED_PENDING_HELD_OUT_EVALUATION", "purpose": configured.purpose,
                }
                continue
            model = self.get(role, require_approved=False)
            loaded = YOLO(str(model.path))
            labels = {str(value).strip().lower() for value in loaded.names.values()}
            required = {value.strip().lower() for value in model.required_classes}
            missing = sorted(required - labels)
            report["models"][role] = {
                **model.provenance(), "labels": sorted(labels), "missing_required_classes": missing,
                "valid": not missing,
            }
            if missing:
                raise ValueError(f"{role} class-map mismatch; missing {missing} in {model.path.name}")
        return report

    def provenance(self, roles: tuple[str, ...] = ("pose", "objects")) -> dict[str, Any]:
        return {role: self.get(role).provenance() for role in roles}
