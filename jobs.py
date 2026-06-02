"""Background job manager for image/video upscaling subprocesses.

A single global JobManager runs at most one upscale job at a time (single GPU).
Each job is an OS subprocess; the manager pipes its stdout, echoes every line to
the server terminal, and parses the small protocol emitted by upscaler.py /
video_upscaler.py into an in-memory JobState the web layer polls.

Controls:
  * terminate(): SIGKILL the child and delete its temp dir. Works for image+video.
  * pause():     SIGTERM the child (video) so it finishes the current frame and
                 exits; finished frames stay on disk.
  * resume():    re-spawn the same command, which skips frames already on disk.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Statuses that mean a job currently owns the GPU and blocks new jobs.
ACTIVE_STATUSES = {"running", "paused"}

_PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)/(\d+)")
_FRAMES_RE = re.compile(r"^META frames=(\d+)")
_STAGE_RE = re.compile(r"^STAGE\s+(\w+)")


@dataclass
class JobState:
    """Mutable state for one upscale job, updated by the reader thread."""
    job_id: str
    kind: str                       # "image" or "video"
    cmd: list[str]
    output_name: str                # filename served from the outputs dir on success
    job_dir: Optional[Path] = None  # scratch dir to remove on terminate (video only)
    status: str = "running"         # running | paused | done | error | terminated
    stage: str = ""                 # extracting | upscaling | encoding
    done: int = 0
    total: int = 0
    control: Optional[str] = None    # "pause" | "terminate" intent, set before signaling
    preview_seq: int = 0             # bumps on each PREVIEW so the GUI can cache-bust
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)


class JobManager:
    """Owns the single active job and its reader thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[JobState] = None

    # --- lifecycle ----------------------------------------------------------
    def start(self, job_id: str, kind: str, cmd: list[str], output_name: str,
              job_dir: Optional[Path] = None) -> JobState:
        """Spawn a new job. Raises RuntimeError if one is already active."""
        with self._lock:
            if self._job is not None and self._job.status in ACTIVE_STATUSES:
                raise RuntimeError("A job is already running. Stop it before starting another.")
            job = JobState(job_id=job_id, kind=kind, cmd=cmd,
                           output_name=output_name, job_dir=job_dir)
            self._job = job
            self._spawn(job)
            return job

    def _spawn(self, job: JobState) -> None:
        """Launch the subprocess and a thread that drains its stdout."""
        job.proc = subprocess.Popen(  # pylint: disable=consider-using-with
            job.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, args=(job, job.proc), daemon=True).start()

    def _read_output(self, job: JobState, proc: subprocess.Popen) -> None:
        """Echo + parse the child's output, then settle the terminal status."""
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(f"[{job.kind}:{job.job_id[:8]}] {line}", flush=True)
            self._parse_line(job, line)

        proc.wait()
        with self._lock:
            # A newer process may have replaced this one (resume); ignore stale readers.
            if job.proc is not proc:
                return
            if job.control == "terminate":
                job.status = "terminated"
            elif job.control == "pause":
                job.status = "paused"
            elif job.status not in ("done",):
                job.status = "done" if proc.returncode == 0 else "error"

    @staticmethod
    def _parse_line(job: JobState, line: str) -> None:
        """Update job fields from one protocol line."""
        match = _PROGRESS_RE.match(line)
        if match:
            job.done, job.total = int(match.group(1)), int(match.group(2))
            return
        match = _FRAMES_RE.match(line)
        if match:
            job.total = int(match.group(1))
            return
        match = _STAGE_RE.match(line)
        if match:
            job.stage = match.group(1)
            return
        if line == "PREVIEW":
            job.preview_seq += 1
        elif line == "DONE":
            job.status = "done"

    # --- controls -----------------------------------------------------------
    def pause(self, job_id: str) -> None:
        """Gracefully pause a running video job (finishes the current frame)."""
        with self._lock:
            job = self._require(job_id)
            if job.status != "running":
                raise RuntimeError(f"Cannot pause a job that is '{job.status}'.")
            job.control = "pause"
            job.status = "paused"
            if job.proc and job.proc.poll() is None:
                job.proc.terminate()  # SIGTERM -> graceful exit after current frame

    def resume(self, job_id: str) -> JobState:
        """Resume a paused video job by re-running its command (skips done frames)."""
        with self._lock:
            job = self._require(job_id)
            if job.status != "paused":
                raise RuntimeError(f"Cannot resume a job that is '{job.status}'.")
            job.control = None
            job.status = "running"
            self._spawn(job)
            return job

    def terminate(self, job_id: str) -> None:
        """Hard-stop the job (SIGKILL) and delete its temp dir."""
        with self._lock:
            job = self._require(job_id)
            job.control = "terminate"
            job.status = "terminated"
            if job.proc and job.proc.poll() is None:
                job.proc.kill()  # SIGKILL -> frees VRAM immediately
        if job.job_dir:
            shutil.rmtree(job.job_dir, ignore_errors=True)

    # --- queries ------------------------------------------------------------
    def snapshot(self, job_id: str) -> Optional[dict]:
        """Return a JSON-serializable view of the job, or None if unknown."""
        with self._lock:
            job = self._job
            if job is None or job.job_id != job_id:
                return None
            percent = int(job.done / job.total * 100) if job.total else 0
            return {
                "job_id": job.job_id,
                "kind": job.kind,
                "status": job.status,
                "stage": job.stage,
                "done": job.done,
                "total": job.total,
                "percent": percent,
                "preview_seq": job.preview_seq,
                "result_ready": job.status == "done",
            }

    def active_job_id(self) -> Optional[str]:
        """Return the id of the running/paused job, or None (used to protect its temp dir)."""
        with self._lock:
            if self._job is not None and self._job.status in ACTIVE_STATUSES:
                return self._job.job_id
            return None

    def _require(self, job_id: str) -> JobState:
        """Return the active job matching job_id or raise (caller holds the lock)."""
        if self._job is None or self._job.job_id != job_id:
            raise RuntimeError("Unknown or no-longer-active job.")
        return self._job


# Module-level singleton imported by app.py
manager = JobManager()
