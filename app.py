""" Main Flask entrypoint for our image-upscaler"""
from __future__ import annotations
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
from dotenv import load_dotenv

from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, session
)
from werkzeug.utils import secure_filename

# Expecting upscaler.py in the same folder with the functions below
from upscaler import upscale, upscale_slice

# Load environment variables
load_dotenv()


app = Flask(__name__)
app.secret_key =  os.environ.get("DEV_SECRET", 'dev-secret') # change this in production


# Where to store temporary uploads/outputs
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = BASE_DIR / "weights"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Pulls your Real-ESRGAN x4 model as specified in your .env or uses a default path
DEFAULT_MODEL = os.environ.get("REAL_ESRGAN_MODEL", "weights/RealESRGAN_x4plus.pth")


def get_available_models() -> list[str]:
    """Scan the weights directory for available model files (.pth)."""
    if not WEIGHTS_DIR.exists():
        return []
    models = [f.name for f in WEIGHTS_DIR.glob("*.pth")]
    return sorted(models)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _do_upscale(src_path: str, dst_path: str, *, slice_tiles: Optional[int] = None, model_path: Optional[str] = None) -> None:
    """Run your ESRGAN logic and write the upscaled image to dst_path."""
    model = model_path or DEFAULT_MODEL
    if slice_tiles:
        result_bgr = upscale_slice(model, src_path, slice_tiles)
    else:
        result_bgr = upscale(model, src_path)

    cv2.imwrite(dst_path, result_bgr)  # pylint: disable=no-member


@app.route("/")
def index():
    """Home page: upload form + current time."""
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
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload, run upscaler, then redirect to result page."""
    if "image" not in request.files:
        flash("No file part found in the request.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not file or file.filename == "":
        flash("Please select an image file to upload.")
        return redirect(url_for("index"))

    if not _allowed(file.filename):
        flash("Unsupported file type. Please upload PNG/JPG/JPEG/WEBP/BMP.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    job_id = uuid.uuid4().hex
    src_path = str(UPLOAD_DIR / f"{job_id}_{filename}")
    file.save(src_path)

    # Optional: allow a simple toggle for slicing (auto tiles)
    # You can expose this via a checkbox/select on the form if you want.
    slice_tiles = request.form.get("slice_tiles")
    tiles = int(slice_tiles) if slice_tiles and slice_tiles.isdigit() else None
    # Persist the tile setting so it's pre-filled on the next visit.
    session["slice_tiles"] = tiles

    # Get selected model
    selected_model = request.form.get("model")
    model_path = None
    if selected_model and selected_model in get_available_models():
        model_path = str(WEIGHTS_DIR / selected_model)
        # Persist the model choice so it's re-selected on the next visit.
        session["model"] = selected_model

    out_name = f"{job_id}.png"
    out_path = str(OUTPUT_DIR / out_name)

    try:
        _do_upscale(src_path, out_path, slice_tiles=tiles, model_path=model_path)
    except (RuntimeError, OSError, ValueError) as exc:
        flash(f"Upscale failed: {exc}")
        return redirect(url_for("index"))

    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id: str):
    """Show the upscaled image preview and a download button."""
    out_name = f"{job_id}.png"
    out_path = OUTPUT_DIR / out_name
    if not out_path.exists():
        flash("That result is no longer available.")
        return redirect(url_for("index"))

    # Pass a static URL for preview and a /download route for "save" prompt
    preview_url = url_for("output_file", filename=out_name)
    download_url = url_for("download", filename=out_name)
    return render_template("result.html", preview_url=preview_url, download_url=download_url)


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    """Serve generated images for inline preview."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/download/<path:filename>")
def download(filename: str):
    """Trigger browser download dialog for the generated image."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/about")
def about():
    """A simple page that demonstrates headings, lists, links, and comments."""
    now = datetime.now()
    return render_template("about.html", now=now)


if __name__ == "__main__":
    # Run:  python app.py
    app.run(debug=True)
