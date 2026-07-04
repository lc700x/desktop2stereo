from __future__ import annotations

import html
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import quote


APP_NAME = "infinidepth-hf-demo"
APP_TEMP_ROOT = Path(tempfile.gettempdir()) / APP_NAME
SUPERSPLAT_VIEWER_DIR = APP_TEMP_ROOT / "supersplat-viewer"
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SUPERSPLAT_VIEWER_DIR = REPO_ROOT / "assets" / "supersplat-viewer"

APP_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _candidate_source_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("SUPERSPLAT_SOURCE_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "third_party" / "supersplat",
            Path("/tmp/supersplat-inspect"),
        ]
    )

    resolved: list[Path] = []
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except FileNotFoundError:
            path = candidate
        if path not in resolved:
            resolved.append(path)
    return resolved


def _viewer_dist_is_ready(viewer_dir: Path) -> bool:
    return (viewer_dir / "index.html").exists() and (viewer_dir / "index.js").exists()


def _find_supersplat_source() -> Path:
    for candidate in _candidate_source_dirs():
        if (candidate / "package.json").exists():
            return candidate
    searched = "\n".join(f"- {path}" for path in _candidate_source_dirs())
    raise RuntimeError(
        "Local SuperSplat source was not found. Set SUPERSPLAT_SOURCE_DIR to a prepared checkout or place it in one of:\n"
        f"{searched}"
    )


def _patch_index_html(index_path: Path) -> None:
    text = index_path.read_text(encoding="utf-8")
    service_worker_block = '''        <!-- Service worker -->
        <script>
            const sw = navigator.serviceWorker;
            if (sw) {
                sw.register('./sw.js')
                    .then(reg => console.log('service worker registered', reg))
                    .catch(err => console.log('failed to register service worker', err));
            }
        </script>
'''
    if service_worker_block in text:
        text = text.replace(service_worker_block, "")
    text = text.replace('<base href="">', '<base href="./">')
    index_path.write_text(text, encoding="utf-8")


def _copy_viewer_dist(source_dist_dir: Path) -> Path:
    source_dist_dir = source_dist_dir.resolve()
    if not _viewer_dist_is_ready(source_dist_dir):
        raise RuntimeError(f"SuperSplat dist is incomplete: {source_dist_dir}")

    SUPERSPLAT_VIEWER_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(SUPERSPLAT_VIEWER_DIR, ignore_errors=True)
    shutil.copytree(source_dist_dir, SUPERSPLAT_VIEWER_DIR)
    _patch_index_html(SUPERSPLAT_VIEWER_DIR / "index.html")
    return SUPERSPLAT_VIEWER_DIR


def _build_supersplat_dist(source_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    dist_dir = source_dir / "dist"
    if _viewer_dist_is_ready(dist_dir):
        return dist_dir

    rollup_bin = source_dir / "node_modules" / "rollup" / "dist" / "bin" / "rollup"
    if not rollup_bin.exists():
        raise RuntimeError(
            "SuperSplat is available but its frontend bundle is not built. Install its Node dependencies and build it first, "
            "or point SUPERSPLAT_SOURCE_DIR to a prepared checkout."
        )

    result = subprocess.run(
        ["node", str(rollup_bin), "-c"],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env={**os.environ, "BUILD_TYPE": "release"},
    )
    if result.returncode != 0 or not _viewer_dist_is_ready(dist_dir):
        build_output = result.stdout.strip()
        message = "Failed to build the embedded SuperSplat viewer."
        if build_output:
            message += f"\n{build_output}"
        raise RuntimeError(message)

    return dist_dir


def ensure_supersplat_viewer_assets() -> Path:
    if _viewer_dist_is_ready(SUPERSPLAT_VIEWER_DIR):
        return SUPERSPLAT_VIEWER_DIR

    if _viewer_dist_is_ready(BUNDLED_SUPERSPLAT_VIEWER_DIR):
        return _copy_viewer_dist(BUNDLED_SUPERSPLAT_VIEWER_DIR)

    source_dir = _find_supersplat_source()
    dist_dir = source_dir / "dist"
    if not _viewer_dist_is_ready(dist_dir):
        dist_dir = _build_supersplat_dist(source_dir)
    return _copy_viewer_dist(dist_dir)


def _gradio_file_url(path: str | Path) -> str:
    return "/gradio_api/file=" + quote(str(Path(path).resolve()), safe="/")


def build_embedded_viewer_html(ply_path: str | Path, height_px: int = 700) -> str:
    viewer_dir = ensure_supersplat_viewer_assets()
    viewer_url = _gradio_file_url(viewer_dir / "index.html")
    splat_url = _gradio_file_url(ply_path)
    filename = Path(ply_path).name
    iframe_src = (
        f"{viewer_url}?load={quote(splat_url, safe='')}&filename={quote(filename, safe='')}"
    )
    escaped_src = html.escape(iframe_src, quote=True)
    return (
        f'<iframe src="{escaped_src}" title="SuperSplat Viewer" loading="eager" allowfullscreen '
        f'style="width:100%; height:{int(height_px)}px; border:0; border-radius:12px; background:#101418;"></iframe>'
        '<div style="margin-top:0.5rem; margin-bottom:0.75rem; font-size:0.92rem; line-height:1.5;">'
        'SuperSplat controls: drag to orbit, scroll to zoom, right-drag to pan. '
        f'<a href="{escaped_src}" target="_blank" rel="noopener noreferrer">Open GS viewer in a new tab</a>'
        '</div>'
    )


def build_viewer_error_html(message: str, ply_path: str | Path) -> str:
    escaped_message = html.escape(message)
    ply_url = html.escape(_gradio_file_url(ply_path), quote=True)
    return (
        '<div style="padding:1rem; border:1px solid #d0d7de; border-radius:12px; background:#fff;">'
        '<strong>Embedded SuperSplat viewer is unavailable.</strong>'
        f'<div style="margin-top:0.5rem; white-space:pre-wrap;">{escaped_message}</div>'
        f'<div style="margin-top:0.75rem;"><a href="{ply_url}" target="_blank" rel="noopener noreferrer">Open exported GS PLY</a></div>'
        '</div>'
    )
