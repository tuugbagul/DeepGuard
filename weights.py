from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_WEIGHT_NAME = "fusion_finetuned.pth"
DEFAULT_RELEASE_URL = os.environ.get(
    "DEEPGUARD_WEIGHTS_URL",
    f"https://github.com/tuugbagul/DeepGuard/releases/latest/download/{DEFAULT_WEIGHT_NAME}",
)

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DOWNLOAD_DIR = _REPO_ROOT / "weights"


def _candidate_paths(weight_path: str | None) -> list[Path]:
    candidates: list[Path] = []

    if weight_path:
        requested = Path(weight_path).expanduser()
        candidates.append(requested)
        if not requested.is_absolute():
            candidates.append(_REPO_ROOT / requested)

    candidates.extend(
        [
            _REPO_ROOT / DEFAULT_WEIGHT_NAME,
            _DEFAULT_DOWNLOAD_DIR / DEFAULT_WEIGHT_NAME,
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)
    return unique_candidates


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=f"{destination.stem}_", suffix=destination.suffix, dir=destination.parent
    )
    os.close(fd)

    request = urllib.request.Request(url, headers={"User-Agent": "DeepGuard/1.0"})
    try:
        with urllib.request.urlopen(request) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        Path(temp_path).replace(destination)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise

    return destination


def resolve_demo_weights(
    weight_path: str | None = None,
    *,
    download: bool = True,
    download_url: str | None = None,
) -> Path:
    """
    Resolve the fusion model weights for the demo.

    Priority order:
    1. Explicit --weights path (absolute or repo-relative)
    2. Repo root `fusion_finetuned.pth`
    3. Repo `weights/fusion_finetuned.pth`
    4. Download from GitHub Release (or `DEEPGUARD_WEIGHTS_URL`)
    """

    attempted = _candidate_paths(weight_path)
    for candidate in attempted:
        if candidate.exists():
            return candidate

    if not download:
        raise FileNotFoundError(_missing_weight_message(attempted, download_url or DEFAULT_RELEASE_URL))

    target = _DEFAULT_DOWNLOAD_DIR / DEFAULT_WEIGHT_NAME
    url = download_url or DEFAULT_RELEASE_URL
    try:
        return _download_file(url, target)
    except urllib.error.HTTPError as exc:
        message = _missing_weight_message(attempted, url)
        raise RuntimeError(f"{message}\nDownload failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        message = _missing_weight_message(attempted, url)
        raise RuntimeError(f"{message}\nDownload failed: {exc.reason}.") from exc
    except Exception as exc:
        message = _missing_weight_message(attempted, url)
        raise RuntimeError(f"{message}\nDownload failed: {exc}.") from exc


def _missing_weight_message(attempted: list[Path], url: str) -> str:
    attempted_paths = "\n".join(f"  - {path}" for path in attempted)
    return (
        "Fusion model weights could not be resolved.\n"
        "Checked these locations:\n"
        f"{attempted_paths}\n"
        f"Download URL: {url}\n"
        "Fix one of these and rerun:\n"
        f"  1. Place `{DEFAULT_WEIGHT_NAME}` in the repo root.\n"
        f"  2. Place `{DEFAULT_WEIGHT_NAME}` in the `weights/` folder.\n"
        "  3. Pass a custom file with `python demo.py --weights path/to/file.pth`.\n"
        "  4. Set `DEEPGUARD_WEIGHTS_URL` to a direct downloadable file URL."
    )
