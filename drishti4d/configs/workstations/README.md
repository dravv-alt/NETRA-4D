# CBT workstation maps

Automatic mode is the default. It learns workstation regions from stable real
person and monitor observations and never copies fixed boxes between cameras.
Maps generated this way carry `calibration_status: AUTOMATIC`.

Existing evidence can be used to precompute an automatic map:

```powershell
python scripts/auto_calibrate_from_evidence.py `
  --detections outputs/<run>/VIDEO_03/detections.jsonl `
  --summary outputs/<run>/VIDEO_03/summary.json `
  --video-id VIDEO_03
```

Human selection is optional. Use it when a reviewer wants to override and
validate the automatic geometry:

```powershell
python scripts/calibrate_cbt_workstations.py `
  --source path/to/original-video.mp4 `
  --camera-id CAM04 `
  --video-id VIDEO_03 `
  --frame-index 250 `
  --calibrated-by <reviewer>
```

The full renderer automatically looks for `VIDEO_XX.yaml` here. Without one it
warms up automatic calibration; if stable workstations cannot be recovered,
the baseline temporal engine remains active. Use `--workstation-mode human
--require-workstation-map` only when a run must require human geometry. Never
copy a map between camera perspectives.
The tool falls back to a Tkinter/Pillow selector when OpenCV HighGUI is not
available, which is the case in the current local Python environment.
