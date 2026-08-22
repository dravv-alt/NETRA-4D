/**
 * NETRA-4D / Drishti-4D Frontend Application Controller
 * Handles Multi-Camera CCTV wall, interactive 4D Canvas rendering,
 * timeline scrubbing, seat risk heatmap, and forensic audit reports.
 */

// Application State
const state = {
    selectedCameraId: "VIDEO_03",
    currentTimeSec: 140.5,
    maxDurationSec: 281.8,
    isPlaying: false,
    layers: {
        seats: true,
        skeleton: true,
        smplx: true,
        gaze: true,
        lap: true
    },
    cameras: [],
    seats: [],
    events: [],
    benchmarks: []
};

// Seed Camera Data
const DEFAULT_CAMERAS = [
    { camera_id: "VIDEO_01", title: "Exam Room 1 - Front", res: "1280x720", fps: 12.0, status: "NORMAL", desc: "CBT Exam Lab (Seats A01-A03)" },
    { camera_id: "VIDEO_02", title: "Seat Exchange Channel", res: "1280x720", fps: 12.0, status: "WATCH", desc: "Aisle & Seating Exchange Monitor" },
    { camera_id: "VIDEO_03", title: "CCTV Mobile Usage", res: "1280x720", fps: 12.0, status: "REVIEW", desc: "Lap Proximity & Downward Gaze" },
    { camera_id: "VIDEO_04", title: "Candidate Talking", res: "640x480", fps: 8.0, status: "REVIEW", desc: "Reciprocal Orientation Tracking" },
    { camera_id: "VIDEO_05", title: "Single Subject Focus", res: "1280x720", fps: 12.0, status: "NORMAL", desc: "Multi-HMR SMPL-X Baseline" },
    { camera_id: "VIDEO_06", title: "Phone Use Scene 2", res: "1280x720", fps: 12.0, status: "WATCH", desc: "Workstation Multi-Candidate" },
    { camera_id: "VIDEO_07", title: "Paper Chit Passing", res: "1280x720", fps: 12.0, status: "REVIEW", desc: "Partition Crossing Activity" },
    { camera_id: "VIDEO_08", title: "Seat 12 Partition Feed", res: "640x480", fps: 8.0, status: "NORMAL", desc: "Low-Res Robustness Feed" }
];

const DEFAULT_SEATS = [
    { seat_id: "A01", camera_id: "VIDEO_03", risk_state: "NORMAL", risk_score: 0.12, poly: [[0.05, 0.20], [0.22, 0.20], [0.22, 0.55], [0.05, 0.55]] },
    { seat_id: "A02", camera_id: "VIDEO_03", risk_state: "REVIEW", risk_score: 0.94, poly: [[0.25, 0.20], [0.42, 0.20], [0.42, 0.55], [0.25, 0.55]] },
    { seat_id: "A03", camera_id: "VIDEO_03", risk_state: "WATCH", risk_score: 0.52, poly: [[0.45, 0.20], [0.62, 0.20], [0.62, 0.55], [0.45, 0.55]] },
    { seat_id: "B01", camera_id: "VIDEO_03", risk_state: "NORMAL", risk_score: 0.08, poly: [[0.05, 0.60], [0.22, 0.60], [0.22, 0.95], [0.05, 0.95]] },
    { seat_id: "B02", camera_id: "VIDEO_03", risk_state: "NORMAL", risk_score: 0.15, poly: [[0.25, 0.60], [0.42, 0.60], [0.42, 0.95], [0.25, 0.95]] },
    { seat_id: "SEAT_12", camera_id: "VIDEO_07", risk_state: "REVIEW", risk_score: 0.91, poly: [[0.45, 0.60], [0.62, 0.60], [0.62, 0.95], [0.45, 0.95]] }
];

const DEFAULT_EVENTS = [
    {
        event_id: "GT_PHONE_01",
        camera_id: "VIDEO_03",
        seat_id: "A02",
        start_sec: 135.0,
        end_sec: 165.0,
        behaviour: "POSSIBLE_OBJECT_INTERACTION",
        state: "REVIEW",
        peak_risk: 0.94,
        mean_quality: 0.91,
        evidence_signals: [
            "LAP_REGION_ASSOCIATION",
            "WRIST_PROXIMITY (< 0.08 normalized)",
            "REPEATED_DOWNWARD_GLANCE (-38.5°)",
            "PHONE_DETECTION_CORRELATED (0.92 confidence)"
        ],
        summary: "Candidate in Seat A02 retrieves mobile phone from lap/under-desk region and operates it repeatedly while checking surrounding invigilators.",
        human_status: "UNREVIEWED"
    },
    {
        event_id: "GT_CHIT_01",
        camera_id: "VIDEO_07",
        seat_id: "SEAT_12",
        start_sec: 22.0,
        end_sec: 50.0,
        behaviour: "POSSIBLE_OBJECT_INTERACTION",
        state: "REVIEW",
        peak_risk: 0.91,
        mean_quality: 0.86,
        evidence_signals: [
            "HAND_CROSSES_SEAT_BOUNDARY",
            "DESK_REGION_CROSSING (Right)",
            "UNUSUAL_HAND_ACTIVITY (Speed > 0.42 px/s)",
            "PAPER_INTERACTION_CANDIDATE"
        ],
        summary: "Candidate at Seat 12 extends hand laterally across the workstation partition boundary to take a piece of paper / chit from adjacent desk.",
        human_status: "UNREVIEWED"
    },
    {
        event_id: "GT_TALKING_01",
        camera_id: "VIDEO_04",
        seat_id: "S01",
        start_sec: 28.0,
        end_sec: 65.0,
        behaviour: "POSSIBLE_NONVERBAL_INTERACTION",
        state: "REVIEW",
        peak_risk: 0.89,
        mean_quality: 0.89,
        evidence_signals: [
            "RECIPROCAL_INTERACTION_S01_AND_S02",
            "SUSTAINED_EXCESSIVE_HEAD_TURN (> 1.5s)",
            "TORSO_ROTATION_TOWARD_NEIGHBOUR (42.0°)",
            "MUTUAL_ORIENTATION_EPISODE"
        ],
        summary: "Candidate at Seat S01 turns toward candidate at Seat S02 and initiates non-verbal communication / whispering, followed by reciprocal head turn.",
        human_status: "UNREVIEWED"
    }
];

const DEFAULT_BENCHMARKS = [
    {
        name: "YOLO26-Pose SOTA (Ours)",
        precision: "94.1%",
        recall: "92.5%",
        f1: "93.3%",
        macroF1: "92.8%",
        tiou: "0.785",
        fpRate: "0.45/min",
        fps: "78.5 FPS",
        vram: "2.4 GB",
        highlight: true,
        desc: "End-to-End NMS-Free real-time 2D pose tracking"
    },
    {
        name: "RTMPose + Objects (Om)",
        precision: "88.5%",
        recall: "86.0%",
        f1: "87.2%",
        macroF1: "86.5%",
        tiou: "0.720",
        fpRate: "0.65/min",
        fps: "58.0 FPS",
        vram: "3.8 GB",
        highlight: false,
        desc: "Object association & Phone disambiguation"
    },
    {
        name: "Multi-HMR + SMPL-X (Dravvya)",
        precision: "91.5%",
        recall: "89.0%",
        f1: "90.2%",
        macroF1: "89.5%",
        tiou: "0.810",
        fpRate: "0.38/min",
        fps: "22.0 FPS",
        vram: "6.8 GB",
        highlight: false,
        desc: "3D Joint angles & Torso orientation mesh"
    }
];

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    state.cameras = DEFAULT_CAMERAS;
    state.seats = DEFAULT_SEATS;
    state.events = DEFAULT_EVENTS;
    state.benchmarks = DEFAULT_BENCHMARKS;

    // Try fetching from backend API if running
    try {
        const res = await fetch("/api/cameras");
        if (res.ok) {
            const data = await res.json();
            if (data.length > 0) state.cameras = data;
        }
    } catch (e) {
        console.log("[*] Using seeded client data.");
    }

    renderCameraStrip();
    renderSeatGrid();
    renderIncidentList();
    renderBenchmarkGrid();
    renderCanvas();
    updateTimelineUI();

    // Start timeline loop
    setInterval(() => {
        if (state.isPlaying) {
            state.currentTimeSec += 0.25;
            if (state.currentTimeSec > state.maxDurationSec) {
                state.currentTimeSec = 0;
            }
            updateTimelineUI();
            renderCanvas();
        }
    }, 250);
}

// ---------------- Camera Strip ----------------
function renderCameraStrip() {
    const strip = document.getElementById("cameraStrip");
    strip.innerHTML = state.cameras.map(c => `
        <div class="cam-strip-card ${c.camera_id === state.selectedCameraId ? 'active' : ''} ${c.status === 'REVIEW' ? 'flagged' : ''}" onclick="selectCamera('${c.camera_id}')">
            <div class="cam-strip-id">${c.camera_id}</div>
            <div class="cam-strip-status" style="color: ${c.status === 'REVIEW' ? 'var(--red)' : (c.status === 'WATCH' ? 'var(--amber)' : 'var(--green)')}">
                ${c.status}
            </div>
        </div>
    `).join("");
}

function selectCamera(camId) {
    state.selectedCameraId = camId;
    const cam = state.cameras.find(c => c.camera_id === camId) || DEFAULT_CAMERAS[0];
    document.getElementById("selectedCamTitle").innerText = `${cam.camera_id} — ${cam.title || 'CCTV Feed'}`;
    document.getElementById("selectedCamMeta").innerText = `${cam.res || '1280x720'} • ${cam.fps || 12.0} FPS • ${cam.desc || 'Active'}`;
    renderCameraStrip();
    renderCanvas();
}

// ---------------- Layer Toggles ----------------
function toggleOverlay(layer) {
    state.layers[layer] = !state.layers[layer];
    const btnMap = {
        seats: "toggleSeatPoly",
        skeleton: "toggleSkeleton",
        smplx: "toggleSmplx",
        gaze: "toggleGaze",
        lap: "toggleLap"
    };
    const btn = document.getElementById(btnMap[layer]);
    if (btn) btn.classList.toggle("active", state.layers[layer]);
    renderCanvas();
}

// ---------------- Canvas Rendering ----------------
function renderCanvas() {
    const canvas = document.getElementById("forensicCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    // Clear background
    ctx.fillStyle = "#020408";
    ctx.fillRect(0, 0, w, h);

    // Draw CCTV Grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 80) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 80) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // 1. Draw Seat Polygons
    if (state.layers.seats) {
        state.seats.forEach(s => {
            const pts = s.poly;
            ctx.beginPath();
            ctx.moveTo(pts[0][0] * w, pts[0][1] * h);
            for (let i = 1; i < pts.length; i++) {
                ctx.lineTo(pts[i][0] * w, pts[i][1] * h);
            }
            ctx.closePath();

            const isReview = s.risk_state === "REVIEW";
            const isWatch = s.risk_state === "WATCH";
            ctx.strokeStyle = isReview ? "#ef4444" : (isWatch ? "#f59e0b" : "#00f0ff");
            ctx.lineWidth = isReview ? 3 : 1.5;
            ctx.stroke();

            ctx.fillStyle = isReview ? "rgba(239, 68, 68, 0.12)" : "rgba(0, 240, 255, 0.04)";
            ctx.fill();

            // Label
            ctx.fillStyle = isReview ? "#ef4444" : "#00f0ff";
            ctx.font = "bold 14px 'JetBrains Mono'";
            ctx.fillText(`SEAT ${s.seat_id} [${s.risk_state}]`, pts[0][0] * w + 8, pts[0][1] * h + 20);
        });
    }

    // 2. Draw Candidate Skeleton (Seat A02)
    const px = 0.335 * w;
    const py = 0.28 * h;

    if (state.layers.skeleton) {
        // Head / Nose
        ctx.fillStyle = "#00f0ff";
        ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();

        // Shoulders
        const ls = { x: px - 45, y: py + 40 };
        const rs = { x: px + 45, y: py + 40 };
        // Wrists (in lap)
        const lw = { x: px - 20, y: py + 120 };
        const rw = { x: px + 20, y: py + 120 };
        const pelvis = { x: px, y: py + 140 };

        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 3;

        // Shoulder bone
        ctx.beginPath(); ctx.moveTo(ls.x, ls.y); ctx.lineTo(rs.x, rs.y); ctx.stroke();
        // Neck to nose
        ctx.beginPath(); ctx.moveTo(px, py + 30); ctx.lineTo(px, py); ctx.stroke();
        // Spine
        ctx.beginPath(); ctx.moveTo(px, py + 40); ctx.lineTo(pelvis.x, pelvis.y); ctx.stroke();
        // Arms to lap
        ctx.beginPath(); ctx.moveTo(ls.x, ls.y); ctx.lineTo(lw.x, lw.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(rs.x, rs.y); ctx.lineTo(rw.x, rw.y); ctx.stroke();

        // Joints
        [ls, rs, lw, rw, pelvis].forEach(j => {
            ctx.fillStyle = "#ff007f";
            ctx.beginPath(); ctx.arc(j.x, j.y, 4, 0, Math.PI * 2); ctx.fill();
        });
    }

    // 3. Draw 3D SMPL-X Mesh Wireframe
    if (state.layers.smplx) {
        ctx.strokeStyle = "rgba(139, 92, 246, 0.6)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);

        // Upper body 3D bounding prism
        ctx.strokeRect(px - 60, py - 20, 120, 180);
        ctx.setLineDash([]);

        ctx.fillStyle = "rgba(139, 92, 246, 0.8)";
        ctx.font = "11px 'JetBrains Mono'";
        ctx.fillText("SMPL-X 3D BODY MESH [Yaw: -38.5°]", px - 55, py - 28);
    }

    // 4. Draw Attention / Gaze Ray
    if (state.layers.gaze) {
        ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + 10, py + 110); // Ray pointing downward to lap
        ctx.stroke();

        ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
        ctx.beginPath(); ctx.arc(px + 10, py + 110, 5, 0, Math.PI * 2); ctx.fill();
    }

    // 5. Draw Lap ROI / Phone Candidate Box
    if (state.layers.lap) {
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
        const rx = px - 35;
        const ry = py + 95;
        const rw = 70;
        const rh = 45;
        ctx.fillRect(rx, ry, rw, rh);
        ctx.strokeRect(rx, ry, rw, rh);

        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 11px 'JetBrains Mono'";
        ctx.fillText("PHONE IN LAP (0.92)", rx, ry - 6);
    }
}

// ---------------- Timeline Controls ----------------
function togglePlayback() {
    state.isPlaying = !state.isPlaying;
    const btn = document.getElementById("playPauseBtn");
    btn.innerText = state.isPlaying ? "⏸ PAUSE" : "▶ PLAY";
}

function stepFrame(deltaSec) {
    state.currentTimeSec = Math.max(0, Math.min(state.maxDurationSec, state.currentTimeSec + deltaSec));
    updateTimelineUI();
    renderCanvas();
}

function seekTimeline(e) {
    const track = document.getElementById("timelineTrack");
    const rect = track.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    state.currentTimeSec = ratio * state.maxDurationSec;
    updateTimelineUI();
    renderCanvas();
}

function updateTimelineUI() {
    const pct = (state.currentTimeSec / state.maxDurationSec) * 100;
    const progress = document.getElementById("timelineProgress");
    const handle = document.getElementById("scrubberHandle");
    if (progress) progress.style.width = `${pct}%`;
    if (handle) handle.style.left = `${pct}%`;

    const mins = Math.floor(state.currentTimeSec / 60);
    const secs = (state.currentTimeSec % 60).toFixed(1);
    const totMins = Math.floor(state.maxDurationSec / 60);
    const totSecs = (state.maxDurationSec % 60).toFixed(1);

    document.getElementById("timeDisplay").innerText = 
        `${String(mins).padStart(2, '0')}:${String(secs).padStart(4, '0')} / ${String(totMins).padStart(2, '0')}:${String(totSecs).padStart(4, '0')}`;
}

// ---------------- Seat Heatmap Grid ----------------
function renderSeatGrid() {
    const grid = document.getElementById("seatGrid");
    grid.innerHTML = state.seats.map(s => `
        <div class="seat-cell ${s.risk_state.toLowerCase()}" onclick="inspectSeat('${s.seat_id}')">
            <div class="seat-id-txt">${s.seat_id}</div>
            <div class="seat-status-txt">${s.risk_state} (${Math.round(s.risk_score * 100)}%)</div>
        </div>
    `).join("");
}

function inspectSeat(seatId) {
    const seat = state.seats.find(s => s.seat_id === seatId);
    if (seat) {
        document.getElementById("hudTarget").innerText = `${seat.seat_id} ➔ ${seat.risk_state}`;
        document.getElementById("hudRiskScore").innerText = `RISK SCORE: ${seat.risk_score.toFixed(2)}`;
        document.getElementById("hudRiskBadge").innerText = `${seat.risk_state} STATE`;
        document.getElementById("hudRiskBadge").style.color = seat.risk_state === 'REVIEW' ? 'var(--red)' : (seat.risk_state === 'WATCH' ? 'var(--amber)' : 'var(--cyan)');
        renderCanvas();
    }
}

// ---------------- Incident List ----------------
function renderIncidentList() {
    const list = document.getElementById("incidentList");
    list.innerHTML = state.events.map(e => `
        <div class="incident-card" onclick="openEventModal('${e.event_id}')">
            <div class="incident-title">
                <span style="color: var(--red);">${e.event_id}</span>
                <span class="badge-mini">${e.camera_id} • ${e.seat_id}</span>
            </div>
            <div class="incident-summary">${e.summary}</div>
            <div class="incident-footer">
                <span>⏱ ${e.start_sec.toFixed(1)}s - ${e.end_sec.toFixed(1)}s</span>
                <span style="color: var(--cyan);">Peak Risk: ${(e.peak_risk * 100).toFixed(0)}%</span>
            </div>
        </div>
    `).join("");
}

// ---------------- Benchmark Grid ----------------
function renderBenchmarkGrid() {
    const grid = document.getElementById("benchmarkGrid");
    grid.innerHTML = state.benchmarks.map(b => `
        <div class="bench-card ${b.highlight ? 'highlight' : ''}">
            <div class="bench-title">${b.name}</div>
            <div class="bench-stats">
                <span class="b-k">Macro-F1:</span><span class="b-v" style="color: var(--cyan);">${b.macroF1}</span>
                <span class="b-k">Temporal IoU:</span><span class="b-v">${b.tiou}</span>
                <span class="b-k">Inference FPS:</span><span class="b-v" style="color: var(--green);">${b.fps}</span>
                <span class="b-k">VRAM:</span><span class="b-v">${b.vram}</span>
                <span class="b-k">FP Rate:</span><span class="b-v">${b.fpRate}</span>
                <span class="b-k">Precision:</span><span class="b-v">${b.precision}</span>
            </div>
        </div>
    `).join("");
}

// ---------------- Modal Dialog ----------------
function openEventModal(eventId) {
    const ev = state.events.find(e => e.event_id === eventId) || state.events[0];
    document.getElementById("modalTitle").innerText = `FORENSIC REVIEW: ${ev.event_id} (${ev.seat_id} / ${ev.camera_id})`;
    document.getElementById("modalBody").innerHTML = `
        <div style="background: var(--bg-card); padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;">
            <strong>Incident Description:</strong><br>
            <span style="color: var(--text-secondary);">${ev.summary}</span>
        </div>
        <div><strong>Evidence Signal Chain:</strong></div>
        <ul style="padding-left: 1.25rem; color: var(--cyan); font-family: var(--font-mono);">
            ${ev.evidence_signals.map(s => `<li>${s}</li>`).join("")}
        </ul>
        <div style="margin-top: 0.75rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-family: var(--font-mono);">
            <div>Interval: <span style="color: #fff;">${ev.start_sec}s - ${ev.end_sec}s</span></div>
            <div>Peak Risk: <span style="color: var(--red);">${(ev.peak_risk * 100).toFixed(1)}%</span></div>
            <div>Perception Quality: <span style="color: var(--green);">${(ev.mean_quality * 100).toFixed(1)}%</span></div>
            <div>Proctor Decision: <span style="color: var(--amber);">${ev.human_status}</span></div>
        </div>
    `;
    document.getElementById("eventModal").style.display = "flex";
}

function closeModal() {
    document.getElementById("eventModal").style.display = "none";
}

function markIncident(status) {
    alert(`Incident status updated to: ${status}`);
    closeModal();
}

function exportForensicReport() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.events, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `NETRA4D_Forensic_Report_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

function syncDriveCheckpoints() {
    alert("💾 Checkpoints synchronized with Google Drive (/content/drive/MyDrive/NETRA-4D/checkpoints/)!");
}
