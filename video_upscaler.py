"""Video upscaler: extract frames, ESRGAN each frame, reassemble with ffmpeg.

Designed to run as a child process spawned by the Flask app (see jobs.py). It
prints a small structured protocol to stdout so the parent can both echo
progress to the server terminal and parse it for the browser:

    META fps=<float>
    META frames=<int>
    STAGE extracting | upscaling | encoding
    PROGRESS <done>/<total>
    PREVIEW
    DONE

Per-frame upscaling reuses upscale()/upscale_slice() from upscaler.py. The job
is resumable: already-upscaled frames are skipped, so Resume (and recovery after
a restart) just re-runs the same command. A SIGTERM handler finishes the
in-flight frame and exits cleanly so Pause never leaves a partial frame.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
from pathlib import Path

import cv2

from upscaler import upscale, upscale_slice

# --- Tunable constants -------------------------------------------------------
PREVIEW_EVERY_N_FRAMES = 24       # how often to refresh the side-by-side compare image
COMPARE_MAX_WIDTH = 1280          # cap the side-by-side preview width (keeps the GUI light)
H264_CRF = 18                     # libx264 quality (lower = better/larger)
FRAME_GLOB = "frame_*.png"        # zero-padded, lexically sortable
FRAME_NAME = "frame_%06d.png"     # ffmpeg image2 pattern
EXTRACT_DONE_MARKER = ".extract_complete"  # sentinel so a partial extraction is redone

# Set by the SIGTERM handler; checked between frames for a graceful pause.
_should_pause = False


def emit(msg: str) -> None:
    """Print one protocol/terminal line, unbuffered."""
    print(msg, flush=True)


def _handle_sigterm(_signum, _frame) -> None:
    """Request a graceful pause: finish the current frame, then exit."""
    global _should_pause
    _should_pause = True


def probe_fps(video_path: str) -> float:
    """Return the video's frame rate via ffprobe (parses 'num/den')."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        text=True,
    ).strip()
    num, _, den = out.partition("/")
    den_val = float(den) if den else 1.0
    return float(num) / den_val if den_val else float(num)


def extract_frames(video_path: str, original_dir: Path) -> None:
    """Extract all frames to original_dir as frame_000000.png ... (skip if done)."""
    marker = original_dir / EXTRACT_DONE_MARKER
    if marker.exists() and any(original_dir.glob(FRAME_GLOB)):
        return  # already extracted in a previous run
    # Clear any partial extraction before redoing it.
    for stale in original_dir.glob(FRAME_GLOB):
        stale.unlink()
    original_dir.mkdir(parents=True, exist_ok=True)
    emit("STAGE extracting")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_path,
            "-start_number", "0",
            str(original_dir / FRAME_NAME),
        ],
        check=True,
    )
    marker.touch()


def _write_atomic(img, dst: Path) -> None:
    """cv2.imwrite to a temp file then rename, so readers never see a partial frame.

    The temp name keeps the real image extension last (e.g. .frame_000000.tmp.png)
    so OpenCV can pick a writer, stays hidden (leading dot), and won't be matched by
    the frame_*.png glob or ffmpeg's %06d pattern.
    """
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    cv2.imwrite(str(tmp), img)  # pylint: disable=no-member
    os.replace(tmp, dst)


def _write_compare(original_path: Path, upscaled_img, compare_path: Path) -> None:
    """Write a side-by-side original-vs-upscaled preview (original scaled to match)."""
    orig = cv2.imread(str(original_path), cv2.IMREAD_COLOR)  # pylint: disable=no-member
    if orig is None:
        return
    height, width = upscaled_img.shape[:2]
    orig_resized = cv2.resize(orig, (width, height))  # pylint: disable=no-member
    side_by_side = cv2.hconcat([orig_resized, upscaled_img])  # pylint: disable=no-member
    if side_by_side.shape[1] > COMPARE_MAX_WIDTH:
        scale = COMPARE_MAX_WIDTH / side_by_side.shape[1]
        side_by_side = cv2.resize(  # pylint: disable=no-member
            side_by_side, (COMPARE_MAX_WIDTH, max(1, int(side_by_side.shape[0] * scale)))
        )
    _write_atomic(side_by_side, compare_path)
    emit("PREVIEW")


def upscale_frames(
    model_path: str, original_dir: Path, upscaled_dir: Path, compare_path: Path,
    slice_tiles: int | None,
) -> None:
    """Upscale every frame, skipping ones already done; refresh preview periodically."""
    upscaled_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(original_dir.glob(FRAME_GLOB))
    total = len(frames)
    emit(f"META frames={total}")
    emit("STAGE upscaling")

    for done, src in enumerate(frames, start=1):
        dst = upscaled_dir / src.name
        if dst.exists():
            emit(f"PROGRESS {done}/{total}")  # already upscaled (resume)
            continue

        if slice_tiles:
            result = upscale_slice(model_path, str(src), slice_tiles)
        else:
            result = upscale(model_path, str(src))
        _write_atomic(result, dst)
        emit(f"PROGRESS {done}/{total}")

        if done == 1 or done % PREVIEW_EVERY_N_FRAMES == 0 or done == total:
            _write_compare(src, result, compare_path)

        if _should_pause:
            emit("PAUSED")
            sys.exit(0)


def _subtitle_args(out_ext: str) -> list[str]:
    """Subtitle codec for the output container (mkv copies any; mp4 needs mov_text)."""
    if out_ext == ".mkv":
        return ["-c:s", "copy"]
    return ["-c:s", "mov_text"]  # mp4: text subs only


def encode_video(
    upscaled_dir: Path, source_video: str, fps: float, output_path: str, job_dir: Path,
) -> None:
    """Reassemble upscaled frames + original audio/subtitle tracks via ffmpeg (atomic)."""
    emit("STAGE encoding")
    out_ext = Path(output_path).suffix.lower()
    tmp_out = job_dir / f"_encode{out_ext}"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-framerate", f"{fps}",
        "-start_number", "0",
        "-i", str(upscaled_dir / FRAME_NAME),
        "-i", source_video,
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(H264_CRF),
        "-c:a", "copy",
        *_subtitle_args(out_ext),
        "-shortest",
        str(tmp_out),
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_out, output_path)


def run(model_path: str, input_path: str, output_path: str, job_dir: Path,
        slice_tiles: int | None) -> None:
    """Full pipeline: probe -> extract -> upscale -> encode."""
    original_dir = job_dir / "original"
    upscaled_dir = job_dir / "upscaled"
    compare_path = job_dir / "compare.png"
    job_dir.mkdir(parents=True, exist_ok=True)

    fps = probe_fps(input_path)
    emit(f"META fps={fps}")
    extract_frames(input_path, original_dir)
    upscale_frames(model_path, original_dir, upscaled_dir, compare_path, slice_tiles)
    encode_video(upscaled_dir, input_path, fps, output_path, job_dir)
    emit("DONE")


def main() -> None:
    """CLI entrypoint (spawned as a subprocess by the Flask app)."""
    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(description="Upscale a video frame-by-frame with Real-ESRGAN.")
    parser.add_argument("-m", "--model_path", required=True, help="path to the ESRGAN model")
    parser.add_argument("-i", "--input", required=True, help="path to the input video")
    parser.add_argument("-o", "--output", required=True, help="path to write the upscaled video")
    parser.add_argument("--job-dir", required=True, help="scratch dir for frames/compare image")
    parser.add_argument(
        "-s", "--slice", nargs="?", type=int, const=4, default=None,
        help="OPTIONAL: slice each frame into ~this many tiles (lowers VRAM use)",
    )
    args = parser.parse_args()

    run(args.model_path, args.input, args.output, Path(args.job_dir), args.slice)


if __name__ == "__main__":
    main()
