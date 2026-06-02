// Drives the upload -> preview -> start -> progress -> result flow on the home page.
(function () {
  const POLL_INTERVAL_MS = 750;

  const el = (id) => document.getElementById(id);
  const dropZone = el('drop-zone');
  const fileInput = el('file');
  const dropZoneContent = dropZone.querySelector('.drop-zone-content');
  const filePreview = el('file-preview');
  const previewImage = el('preview-image');
  const fileName = el('file-name');
  const uploadError = el('upload-error');
  const upscaleBtn = el('upscale-btn');

  const uploadSection = el('upload-section');
  const progressSection = el('progress-section');
  const resultSection = el('result-section');

  const stageLabel = el('stage-label');
  const frameCount = el('frame-count');
  const progressFill = el('progress-fill');
  const compareView = el('compare-view');
  const compareImage = el('compare-image');

  const pauseBtn = el('pause-btn');
  const resumeBtn = el('resume-btn');
  const terminateBtn = el('terminate-btn');

  const resultHeading = el('result-heading');
  const resultImage = el('result-image');
  const resultVideo = el('result-video');
  const downloadLink = el('download-link');

  let jobId = null;
  let jobKind = null;
  let pollTimer = null;

  const show = (node) => { node.style.display = ''; };
  const hide = (node) => { node.style.display = 'none'; };

  function setError(msg) {
    if (msg) {
      uploadError.textContent = msg;
      show(uploadError);
    } else {
      hide(uploadError);
    }
  }

  // Drag and drop wiring (mirrors the original image-only behaviour).
  dropZone.addEventListener('click', (e) => {
    if (e.target !== fileInput) fileInput.click();
  });
  const chooseFileBtn = dropZone.querySelector('.choose-file-btn');
  if (chooseFileBtn) {
    chooseFileBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
  }
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((name) => {
    dropZone.addEventListener(name, preventDefaults, false);
    document.body.addEventListener(name, preventDefaults, false);
  });
  function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
  ['dragenter', 'dragover'].forEach((name) =>
    dropZone.addEventListener(name, () => dropZone.classList.add('drop-zone-active')));
  ['dragleave', 'drop'].forEach((name) =>
    dropZone.addEventListener(name, () => dropZone.classList.remove('drop-zone-active')));

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) { fileInput.files = files; uploadFile(files[0]); }
  });
  fileInput.addEventListener('change', function () {
    if (this.files.length > 0) uploadFile(this.files[0]);
  });

  async function uploadFile(file) {
    setError(null);
    upscaleBtn.disabled = true;
    fileName.textContent = 'Uploading ' + file.name + '…';
    dropZoneContent.style.display = 'none';
    show(filePreview);

    const data = new FormData();
    data.append('file', file);
    try {
      const res = await fetch('/upload', { method: 'POST', body: data });
      const json = await res.json();
      if (!res.ok) { setError(json.error || 'Upload failed.'); return; }
      jobId = json.job_id;
      jobKind = json.kind;
      previewImage.src = json.preview_url;
      fileName.textContent = json.filename + (jobKind === 'video' ? ' (preview frame)' : '');
      upscaleBtn.disabled = false;
    } catch (err) {
      setError('Upload failed: ' + err.message);
    }
  }

  upscaleBtn.addEventListener('click', startJob);

  async function startJob() {
    if (!jobId) return;
    setError(null);
    upscaleBtn.disabled = true;

    const data = new FormData();
    data.append('model', el('model').value);
    data.append('slice_tiles', el('slice_tiles').value);
    try {
      const res = await fetch('/start/' + jobId, { method: 'POST', body: data });
      const json = await res.json();
      if (!res.ok) { setError(json.error || 'Could not start the job.'); upscaleBtn.disabled = false; return; }
      hide(uploadSection);
      show(progressSection);
      // Pause/Resume only apply to video jobs.
      pauseBtn.style.display = jobKind === 'video' ? '' : 'none';
      resumeBtn.style.display = 'none';
      poll();
    } catch (err) {
      setError('Could not start the job: ' + err.message);
      upscaleBtn.disabled = false;
    }
  }

  const STAGE_TEXT = {
    extracting: 'Extracting frames…',
    upscaling: 'Upscaling…',
    encoding: 'Encoding video…',
  };

  async function poll() {
    try {
      const res = await fetch('/status/' + jobId);
      if (res.ok) updateProgress(await res.json());
    } catch (err) {
      /* transient network hiccup; keep polling */
    }
  }

  function updateProgress(snap) {
    if (snap.status === 'done') { showResult(snap); return; }
    if (snap.status === 'terminated') { finish('Job terminated.', false); return; }
    if (snap.status === 'error') { finish('The job failed. Check the server log.', true); return; }

    const indeterminate = !snap.total;
    progressFill.classList.toggle('indeterminate', indeterminate && snap.status === 'running');
    progressFill.style.width = indeterminate ? '100%' : snap.percent + '%';

    if (snap.status === 'paused') {
      stageLabel.textContent = 'Paused';
      progressFill.classList.remove('indeterminate');
      pauseBtn.style.display = 'none';
      if (jobKind === 'video') resumeBtn.style.display = '';
    } else {
      stageLabel.textContent = STAGE_TEXT[snap.stage] || 'Upscaling…';
      resumeBtn.style.display = 'none';
      if (jobKind === 'video') pauseBtn.style.display = '';
    }

    frameCount.textContent = snap.total ? `${snap.done} / ${snap.total} frames` : '';

    if (snap.compare_url) {
      show(compareView);
      compareImage.src = snap.compare_url + '?seq=' + snap.preview_seq;
    }

    pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
  }

  function showResult(snap) {
    if (pollTimer) clearTimeout(pollTimer);
    hide(progressSection);
    show(resultSection);
    downloadLink.href = snap.download_url;
    if (jobKind === 'video') {
      resultHeading.textContent = 'Your upscaled video';
      resultVideo.src = snap.result_url;
      show(resultVideo);
    } else {
      resultHeading.textContent = 'Your upscaled image';
      resultImage.src = snap.result_url;
      show(resultImage);
    }
  }

  function finish(message, isError) {
    if (pollTimer) clearTimeout(pollTimer);
    hide(progressSection);
    show(uploadSection);
    setError(message);
    // Reset so the user can run another file.
    dropZoneContent.style.display = '';
    hide(filePreview);
    upscaleBtn.disabled = true;
    jobId = null;
  }

  async function control(action) {
    if (!jobId) return;
    try {
      const res = await fetch('/' + action + '/' + jobId, { method: 'POST' });
      if (!res.ok) {
        const json = await res.json().catch(() => ({}));
        setError(json.error || (action + ' failed.'));
      }
    } catch (err) {
      setError(action + ' failed: ' + err.message);
    }
  }

  pauseBtn.addEventListener('click', () => control('pause'));
  resumeBtn.addEventListener('click', () => { control('resume').then(poll); });
  terminateBtn.addEventListener('click', () => {
    if (confirm('Terminate this job and discard its progress?')) control('terminate');
  });

  const cleanupBtn = el('cleanup-btn');
  const tmpUsage = el('tmp-usage');
  cleanupBtn.addEventListener('click', async () => {
    cleanupBtn.disabled = true;
    try {
      const res = await fetch('/cleanup', { method: 'POST' });
      const json = await res.json();
      if (res.ok) tmpUsage.textContent = json.remaining_human + ' (freed ' + json.freed_human + ')';
    } catch (err) {
      setError('Cleanup failed: ' + err.message);
    } finally {
      cleanupBtn.disabled = false;
    }
  });
}());
