async function loadVerifiedRun() {
  const status = document.getElementById("statusText");
  try {
    const response = await fetch("/api/runs/latest");
    if (!response.ok) {
      status.textContent = "No verified inference run exists yet.";
      return;
    }
    const run = await response.json();
    const summary = run.summary || {};
    document.getElementById("runId").textContent = run.run_id;
    document.getElementById("device").textContent = run.runtime.actual_device;
    document.getElementById("videos").textContent = summary.videos_processed || 0;
    document.getElementById("detections").textContent =
      (summary.person_detections || 0) + (summary.object_detections || 0);
    status.textContent = `Completed ${run.completed_at_utc}. Malpractice classification and ground-truth metrics were not evaluated.`;
    document.getElementById("assets").innerHTML = run.assets.map(asset => {
      const frames = asset.sampled_frames || [];
      const people = frames.reduce((total, frame) => total + frame.people.length, 0);
      const objects = frames.reduce((total, frame) => total + frame.objects.length, 0);
      return `<article class="bench-card"><div class="bench-title">${asset.source.asset_id}</div>
        <div class="bench-stats"><span class="b-k">Source file</span><span class="b-v">${asset.source.filename}</span>
        <span class="b-k">Frames sampled</span><span class="b-v">${frames.length}</span>
        <span class="b-k">People</span><span class="b-v">${people}</span>
        <span class="b-k">Objects</span><span class="b-v">${objects}</span>
        <span class="b-k">SHA-256</span><span class="b-v">${asset.source.sha256}</span></div></article>`;
    }).join("");
  } catch (error) {
    status.textContent = `Unable to load the API: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", loadVerifiedRun);
