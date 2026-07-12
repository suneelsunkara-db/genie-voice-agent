from __future__ import annotations

import os
import sys
from pathlib import Path


def _path_variants(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = raw.removeprefix("dbfs:")
    variants = [cleaned]
    if cleaned.startswith("/Volumes/"):
        variants.append(f"/dbfs{cleaned}")
    return variants


def _package_root_from_config(config_path: str) -> str | None:
    path = Path(config_path.removeprefix("dbfs:"))
    parts = path.parts
    if "evaluations" not in parts:
        return None
    idx = parts.index("evaluations")
    training_root = str(Path(*parts[:idx]))
    return f"{training_root}/jobs/ml_asr_eval/package"


def _extract_worker_bootstrap_args(argv: list[str]) -> tuple[list[str], str | None]:
    cleaned = list(argv)
    package_root: str | None = None
    if "--package-root" in cleaned:
        idx = cleaned.index("--package-root")
        if idx + 1 < len(cleaned):
            package_root = cleaned[idx + 1]
        del cleaned[idx : idx + 2]
    if not package_root and "--config" in cleaned:
        idx = cleaned.index("--config")
        if idx + 1 < len(cleaned):
            package_root = _package_root_from_config(cleaned[idx + 1])
    return cleaned, package_root


def _bootstrap_sys_path(package_root: str | None) -> str:
    roots: list[str] = []
    if package_root:
        roots.append(package_root)
    env_root = os.environ.get("ML_ASR_PACKAGE_ROOT")
    if env_root:
        roots.append(env_root)
    jobs_dir = os.environ.get("ML_ASR_JOBS_DIR")
    if jobs_dir:
        roots.append(f"{jobs_dir.rstrip('/')}/package")

    tried: list[str] = []
    for root in roots:
        for variant in _path_variants(root):
            if variant in tried:
                continue
            tried.append(variant)
            if variant not in sys.path:
                sys.path.insert(0, variant)

    try:
        import genie_voice  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import genie_voice on serverless worker. "
            f"Tried package roots: {tried or roots}. "
            "Ensure jobs/ml_asr_eval/package was staged to the Volume."
        ) from exc

    return tried[0] if tried else (roots[0] if roots else "")


def main() -> None:
    os.environ.setdefault("ML_ASR_RUN_MODE", "serverless")
    worker_argv, package_root = _extract_worker_bootstrap_args(sys.argv[1:])
    if package_root:
        os.environ["ML_ASR_PACKAGE_ROOT"] = package_root
    sys.argv = [sys.argv[0], *worker_argv]
    _bootstrap_sys_path(package_root)
    from genie_voice.ml_asr.cli import main as cli_main

    try:
        code = int(cli_main())
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 1
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
