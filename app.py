""" Main Flask entrypoint for our image/video upscaler."""
from __future__ import annotations
import os
import sys
import uuid
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, session, jsonify, abort
)
from werkzeug.utils import secure_filename

from jobs import manager

# Load environment variables
load_dotenv()


app = Flask(__name__)
app.secret_key = os.environ.get("DEV_SECRET", 'dev-secret')  # change this in production


# Where to store uploads, outputs, previews, and per-job scratch frames
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = BASE_DIR / "weights"
TMP_DIR = BASE_DIR / "tmp"                 # tmp/<job_id>/{original,upscaled,compare.png}
PREVIEW_DIR = OUTPUT_DIR / "previews"      # extracted video preview frames
for _d in (UPLOAD_DIR, OUTPUT_DIR, TMP_DIR, PREVIEW_DIR):
    _d.mkdir(exist_ok=True, parents=True)

# Pulls your Real-ESRGAN x4 model as specified in your .env or uses a default path
DEFAULT_MODEL = os.environ.get("REAL_ESRGAN_MODEL", "weights/RealESRGAN_x4plus.pth")

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "mkv"}

# Remembers each saved upload between /upload and /start (job_id -> details).
uploads_index: dict[str, dict] = {}


def get_available_models() -> list[str]:
    """Scan the weights directory for available model files (.pth)."""
    if not WEIGHTS_DIR.exists():
        return []
    return sorted(f.name for f in WEIGHTS_DIR.glob("*.pth"))


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def classify_kind(filename: str) -> Optional[str]:
    """Return 'image', 'video', or None for an upload filename."""
    ext = _ext(filename)
    if ext in ALLOWED_IMAGE:
        return "image"
    if ext in ALLOWED_VIDEO:
        return "video"
    return None


def _dir_size(path: Path) -> int:
    """Total size in bytes of all files under path."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _resolve_model() -> str:
    """Pick the model path from the submitted form, persisting the choice in session."""
    selected_model = request.form.get("model")
    if selected_model and selected_model in get_available_models():
        session["model"] = selected_model
        return str(WEIGHTS_DIR / selected_model)
    return DEFAULT_MODEL


def _resolve_tiles() -> Optional[int]:
    """Parse the optional tiles count from the form, persisting it in session."""
    raw = request.form.get("slice_tiles")
    tiles = int(raw) if raw and raw.isdigit() else None
    session["slice_tiles"] = tiles
    return tiles


@app.route("/")
def index():
    """Home page: upload form."""
    now = datetime.now()
    models = get_available_models()
    env_default = Path(DEFAULT_MODEL).name if DEFAULT_MODEL else None
    # Persisted user selection takes precedence over the env default.
    selected_model = session.get("model")
    if selected_model not in models:
        selected_model = env_default
    selected_tiles = session.get("slice_tiles")
    return render_template(
        "index.html",
        now=now,
        models=models,
        selected_model=selected_model,
        selected_tiles=selected_tiles,
        tmp_usage=_human_size(_dir_size(TMP_DIR)),
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Save an upload and return JSON with a preview so the user can confirm + start."""
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify(error="Please select a file to upload."), 400

    kind = classify_kind(file.filename)
    if kind is None:
        return jsonify(error="Unsupported file type. Images (PNG/JPG/WEBP/BMP) or video (MP4/MKV)."), 400

    filename = secure_filename(file.filename)
    job_id = uuid.uuid4().hex
    saved_name = f"{job_id}_{filename}"
    src_path = UPLOAD_DIR / saved_name
    file.save(str(src_path))

    if kind == "video":
        preview_name = f"{job_id}.png"
        try:
            _extract_preview_frame(src_path, PREVIEW_DIR / preview_name)
        except (subprocess.CalledProcessError, OSError) as exc:
            return jsonify(error=f"Could not read video: {exc}"), 400
        preview_url = url_for("preview_file", filename=preview_name)
    else:
        preview_url = url_for("upload_file", filename=saved_name)

    uploads_index[job_id] = {"kind": kind, "src_path": str(src_path),
                             "filename": filename, "ext": _ext(filename)}
    return jsonify(job_id=job_id, kind=kind, preview_url=preview_url, filename=filename)


def _extract_preview_frame(video_path: Path, dst: Path) -> None:
    """Grab a representative frame for the upload preview.

    Uses ffmpeg's `thumbnail` filter (scores a batch of frames by histogram and
    picks the most representative one) so we avoid black fade-in/intro frames
    that a plain first-frame grab would catch.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-vf", "thumbnail=n=300", "-frames:v", "1", "-q:v", "3", str(dst)],
        check=True,
    )


@app.route("/start/<job_id>", methods=["POST"])
def start(job_id: str):
    """Kick off the upscale job (image or video) for a previously uploaded file."""
    info = uploads_index.get(job_id)
    if not info:
        return jsonify(error="Upload not found. Please re-select your file."), 404

    model_path = _resolve_model()
    tiles = _resolve_tiles()
    src_path = info["src_path"]

    if info["kind"] == "video":
        out_name = f"{job_id}.{info['ext']}"
        job_dir = TMP_DIR / job_id
        cmd = [sys.executable, str(BASE_DIR / "video_upscaler.py"),
               "-m", model_path, "-i", src_path,
               "-o", str(OUTPUT_DIR / out_name), "--job-dir", str(job_dir)]
        if tiles:
            cmd += ["-s", str(tiles)]
    else:
        out_name = f"{job_id}.png"
        job_dir = None
        cmd = [sys.executable, str(BASE_DIR / "upscaler.py"),
               "-m", model_path, "-i", src_path, "-o", str(OUTPUT_DIR / out_name)]
        if tiles:
            cmd += ["-s", str(tiles)]

    try:
        manager.start(job_id, info["kind"], cmd, out_name, job_dir=job_dir)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409

    return jsonify(ok=True, job_id=job_id, kind=info["kind"])


@app.route("/status/<job_id>")
def status(job_id: str):
    """Return a JSON snapshot the GUI polls for progress + result links."""
    snap = manager.snapshot(job_id)
    if snap is None:
        return jsonify(error="Unknown job."), 404
    if snap["result_ready"]:
        info = uploads_index.get(job_id, {})
        out_name = f"{job_id}.png" if info.get("kind") != "video" else f"{job_id}.{info.get('ext')}"
        snap["result_url"] = url_for("output_file", filename=out_name)
        snap["download_url"] = url_for("download", filename=out_name)
        snap["compare_url"] = url_for("compare_file", job_id=job_id)
    elif snap["preview_seq"]:
        snap["compare_url"] = url_for("compare_file", job_id=job_id)
    return jsonify(snap)


@app.route("/pause/<job_id>", methods=["POST"])
def pause(job_id: str):
    """Pause a running video job (finishes the current frame)."""
    return _control(lambda: manager.pause(job_id))


@app.route("/resume/<job_id>", methods=["POST"])
def resume(job_id: str):
    """Resume a paused video job."""
    return _control(lambda: manager.resume(job_id))


@app.route("/terminate/<job_id>", methods=["POST"])
def terminate(job_id: str):
    """Hard-stop any job and clean up its temp files."""
    return _control(lambda: manager.terminate(job_id))


def _control(action):
    """Run a manager control action, mapping failures to a JSON 409."""
    try:
        action()
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(ok=True)


@app.route("/cleanup", methods=["POST"])
def cleanup():
    """Delete every per-job temp dir except the one belonging to the active job."""
    active = manager.active_job_id()
    freed = 0
    for child in TMP_DIR.iterdir():
        if child.is_dir() and child.name != active:
            freed += _dir_size(child)
            shutil.rmtree(child, ignore_errors=True)
    return jsonify(ok=True, freed_human=_human_size(freed),
                   remaining_human=_human_size(_dir_size(TMP_DIR)))


@app.route("/compare/<job_id>")
def compare_file(job_id: str):
    """Serve the live side-by-side preview image for a video job."""
    compare_path = TMP_DIR / job_id / "compare.png"
    if not compare_path.exists():
        abort(404)
    return send_from_directory(TMP_DIR / job_id, "compare.png")


@app.route("/preview/<path:filename>")
def preview_file(filename: str):
    """Serve an extracted video preview frame."""
    return send_from_directory(PREVIEW_DIR, filename)


@app.route("/uploads/<path:filename>")
def upload_file(filename: str):
    """Serve an uploaded image for its pre-upscale preview."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/result/<job_id>")
def result(job_id: str):
    """Standalone result page (image preview or video player) for direct links."""
    matches = sorted(OUTPUT_DIR.glob(f"{job_id}.*"))
    matches = [m for m in matches if m.parent == OUTPUT_DIR]
    if not matches:
        flash("That result is no longer available.")
        return redirect(url_for("index"))
    out_name = matches[0].name
    kind = "video" if _ext(out_name) in ALLOWED_VIDEO else "image"
    return render_template(
        "result.html",
        kind=kind,
        preview_url=url_for("output_file", filename=out_name),
        download_url=url_for("download", filename=out_name),
    )


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    """Serve generated results for inline preview/playback."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/download/<path:filename>")
def download(filename: str):
    """Trigger a browser download for the generated result."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/about")
def about():
    """A simple page that demonstrates headings, lists, links, and comments."""
    now = datetime.now()
    return render_template("about.html", now=now)


if __name__ == "__main__":
    # Run:  python app.py
    app.run(debug=True)
