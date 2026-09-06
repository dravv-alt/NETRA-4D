"""
Risk Score Engine and Alert Manager for Watcher.

Implements decaying risk accumulator, alert generation, and SQLite logging.
"""

import sqlite3
import yaml
import logging
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "LOW"
    MILD = "MILD"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


@dataclass
class AlertCard:
    """Explainable alert card for invigilators."""
    alert_id: str
    seat_id: str
    camera_id: str
    timestamp: str
    risk_score: float
    severity: str
    behaviors: List[Dict[str, Any]]
    explanation: str
    recommended_action: str
    confidence: float
    evidence_frame: Optional[str] = None


class RiskEngine:
    """
    Decaying risk accumulator for behavioral risk scoring.
    
    Core formula:
    WatcherRisk(t) = decay × WatcherRisk(t−1) + Σ event_weight × event_confidence × context_multiplier
    
    Normalized to 0-100 scale.
    """
    
    def __init__(self, config_path: str = "configs/rules.yaml", fps: float = 25.0):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.fps = fps
        
        # Risk parameters
        risk_config = self.config.get("alerts", {})
        self.decay = 0.998  # Decay factor per frame (~0.95 per second at 25fps)
        self.low_threshold = risk_config.get("low_threshold", 25)
        self.mild_threshold = risk_config.get("mild_threshold", 45)
        self.moderate_threshold = risk_config.get("moderate_threshold", 70)
        self.high_threshold = risk_config.get("high_threshold", 85)
        self.sustained_duration_sec = risk_config.get("sustained_duration_sec", 3)
        self.cooldown_sec = risk_config.get("cooldown_sec", 15)
        
        # Event weights from rules config
        self.event_weights = {}
        rules = self.config.get("rules", {})
        for rule_name, rule_config in rules.items():
            if isinstance(rule_config, dict) and "weight" in rule_config:
                self.event_weights[rule_name] = rule_config["weight"]
        
        # Per-seat risk state
        self.risk_scores: Dict[str, float] = {}  # seat_id -> current risk score
        self.risk_history: Dict[str, List[float]] = {}  # seat_id -> history of scores
        self.max_history = int(300 * fps)  # 5 minutes
        self.alert_state: Dict[str, Dict] = {}  # seat_id -> alert state
        self.last_alert_time: Dict[str, float] = {}  # seat_id -> timestamp
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}")
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def update_risk(self, seat_id: str, events: List, 
                    pose_confidence: float = 1.0,
                    visibility_factor: float = 1.0,
                    camera_quality_factor: float = 1.0) -> float:
        """
        Update risk score for a seat based on detected events.
        
        Args:
            seat_id: Seat identifier
            events: List of BehaviorEvent objects
            pose_confidence: Average pose confidence (0-1)
            visibility_factor: Visibility factor (0-1)
            camera_quality_factor: Camera quality factor (0-1)
            
        Returns:
            Updated risk score (0-100)
        """
        # Initialize if new seat
        if seat_id not in self.risk_scores:
            self.risk_scores[seat_id] = 0.0
            self.risk_history[seat_id] = []
            self.alert_state[seat_id] = {"active": False, "level": RiskLevel.LOW.value}
            self.last_alert_time[seat_id] = 0
        
        # Apply smooth decay
        current_risk = self.risk_scores[seat_id] * self.decay
        
        # Add event contributions
        for event in events:
            weight = event.weight
            confidence = event.confidence
            context_mult = max(0.5, pose_confidence) * visibility_factor * camera_quality_factor
            
            # Add weighted contribution scaled for noticeable risk spikes
            contribution = weight * confidence * context_mult * 7.5
            current_risk += contribution
        
        # Cap at 100
        current_risk = min(current_risk, 100.0)
        
        # Update state
        self.risk_scores[seat_id] = current_risk
        self.risk_history[seat_id].append(current_risk)
        
        # Trim history
        if len(self.risk_history[seat_id]) > self.max_history:
            self.risk_history[seat_id] = self.risk_history[seat_id][-self.max_history:]
        
        return current_risk
    
    def get_risk_level(self, score: float) -> RiskLevel:
        """Get risk level from score."""
        if score >= self.high_threshold:
            return RiskLevel.HIGH
        elif score >= self.moderate_threshold:
            return RiskLevel.MODERATE
        elif score >= self.mild_threshold:
            return RiskLevel.MILD
        else:
            return RiskLevel.LOW
    
    def check_alert_condition(self, seat_id: str, current_time: float, events: List = None) -> Optional[Dict]:
        """
        Check if alert should be generated for seat.
        
        Returns:
            Alert info dict or None
        """
        if seat_id not in self.risk_scores:
            return None
        
        score = self.risk_scores[seat_id]
        level = self.get_risk_level(score)
        
        # Check cooldown
        last_alert = self.last_alert_time.get(seat_id, 0)
        if current_time - last_alert < self.cooldown_sec:
            return None
        
        # Immediate alert for phone or suspicious object
        has_priority_event = False
        if events:
            for e in events:
                etype = getattr(e, "event_type", None)
                if etype is not None and (etype.value if hasattr(etype, 'value') else str(etype)) in ["PHONE_DETECTED", "OBJECT_NEAR_HAND"]:
                    has_priority_event = True
                    break
        
        if has_priority_event and score >= 35:
            return {
                "seat_id": seat_id,
                "risk_score": score,
                "level": level.value if level != RiskLevel.LOW else RiskLevel.MODERATE.value,
                "threshold": 35,
                "sustained_frames": 1
            }
        
        # Check sustained duration or threshold
        history = self.risk_history.get(seat_id, [])
        if len(history) >= int(self.fps):  # at least 1s
            sustained_frames = min(int(self.sustained_duration_sec * self.fps), len(history))
            recent_scores = history[-sustained_frames:]
            avg_recent = float(np.mean(recent_scores))
            
            if avg_recent >= self.mild_threshold or score >= self.moderate_threshold:
                return {
                    "seat_id": seat_id,
                    "risk_score": score,
                    "level": level.value if level != RiskLevel.LOW else RiskLevel.MILD.value,
                    "threshold": self.mild_threshold,
                    "sustained_frames": sustained_frames
                }
        
        return None
    
    def create_alert_card(self, seat_id: str, track_id: int, camera_id: str,
                          events: List, risk_score: float, 
                          evidence_frame_path: Optional[str] = None) -> AlertCard:
        """Create explainable alert card."""
        alert_id = f"W-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{seat_id}"
        
        # Group events by type
        behavior_summary = {}
        for event in events:
            etype = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            if etype not in behavior_summary:
                behavior_summary[etype] = {
                    "count": 0,
                    "total_duration": 0,
                    "max_confidence": 0,
                    "details": []
                }
            behavior_summary[etype]["count"] += 1
            behavior_summary[etype]["total_duration"] += event.duration_sec
            behavior_summary[etype]["max_confidence"] = max(
                behavior_summary[etype]["max_confidence"], event.confidence)
            if event.details:
                behavior_summary[etype]["details"].append(event.details)
        
        behaviors = []
        for btype, info in behavior_summary.items():
            behaviors.append({
                "type": btype,
                "count": info["count"],
                "duration_seconds": round(info["total_duration"], 1),
                "confidence": round(info["max_confidence"], 2),
                "details": info["details"]
            })
        
        # Generate explanation
        explanation_parts = []
        for b in behaviors:
            if b["type"] == "REPEATED_SIDEWARD_GLANCE":
                explanation_parts.append(f"repeated sideward glances ({b['count']} in window)")
            elif b["type"] == "SUSTAINED_HEAD_TURN":
                explanation_parts.append(f"sustained head turn ({b['duration_seconds']:.1f}s)")
            elif b["type"] == "TORSO_ROTATION_TOWARD_NEIGHBOR":
                explanation_parts.append(f"torso oriented toward neighbor ({b['duration_seconds']:.1f}s)")
            elif b["type"] == "HAND_BELOW_DESK":
                explanation_parts.append(f"hand below desk / in lap ({b['duration_seconds']:.1f}s)")
            elif b["type"] == "HAND_NEAR_EAR":
                explanation_parts.append(f"hand near ear ({b['duration_seconds']:.1f}s)")
            elif b["type"] == "HAND_NEAR_FACE":
                explanation_parts.append(f"hand near face/mouth ({b['duration_seconds']:.1f}s)")
            elif b["type"] == "OBJECT_NEAR_HAND":
                obj_cls = b['details'][0].get('object_class', 'unknown') if b['details'] else 'unknown'
                explanation_parts.append(f"suspicious object near hand ({obj_cls})")
            elif b["type"] == "PHONE_DETECTED":
                explanation_parts.append(f"phone detected (conf: {b['confidence']:.2f})")
        
        if not explanation_parts:
            explanation_parts.append(f"elevated behavioral suspicion index")
            
        explanation = f"Seat {seat_id}: " + "; ".join(explanation_parts) + f". Risk score: {risk_score:.0f}/100."
        
        severity = self.get_risk_level(risk_score).value
        if severity == "LOW" and risk_score >= 35:
            severity = "MILD"
        
        return AlertCard(
            alert_id=alert_id,
            seat_id=seat_id,
            camera_id=camera_id,
            timestamp=datetime.now().isoformat(),
            risk_score=round(risk_score, 1),
            severity=severity,
            behaviors=behaviors,
            explanation=explanation,
            recommended_action="Manual verification by invigilator",
            confidence=round(np.mean([b["confidence"] for b in behaviors]) if behaviors else 0.7, 2),
            evidence_frame=evidence_frame_path
        )
    
    def get_risk_history(self, seat_id: str, seconds: int = 60) -> List[float]:
        """Get recent risk history for a seat."""
        if seat_id not in self.risk_history:
            return []
        frames = int(seconds * self.fps)
        return self.risk_history[seat_id][-frames:] if len(self.risk_history[seat_id]) > frames else self.risk_history[seat_id]


class AlertManager:
    """Manages alert generation, deduplication, and storage."""
    
    def __init__(self, risk_engine: RiskEngine, db_manager: 'DatabaseManager'):
        self.risk_engine = risk_engine
        self.db = db_manager
        self.camera_id = "CAM-01"
        self.alerts_generated = 0
    
    def process_frame(self, seat_id: str, track_id: int, 
                      timestamp: float, frame_idx: int,
                      events: List, fusion_result = None,
                      pose_confidence: float = 1.0,
                      evidence_frame: Optional[np.ndarray] = None) -> List[AlertCard]:
        """Process frame and generate alerts if conditions met."""
        alerts = []
        
        # Update risk score
        risk_score = self.risk_engine.update_risk(
            seat_id, events, pose_confidence
        )
        
        # Check alert condition
        alert_info = self.risk_engine.check_alert_condition(seat_id, timestamp, events)
        
        if alert_info:
            # Save evidence frame if available
            evidence_path = None
            if evidence_frame is not None:
                evidence_path = f"outputs/debug/snapshots/alert_{seat_id}_{frame_idx:06d}_{alert_info['level']}.jpg"
                Path(evidence_path).parent.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(evidence_path, evidence_frame)
            
            # Create alert card
            alert_card = self.risk_engine.create_alert_card(
                seat_id, track_id, self.camera_id, events, risk_score, evidence_path
            )
            
            # Store in database
            self.db.insert_alert(alert_card)
            
            self.alerts_generated += 1
            self.risk_engine.last_alert_time[seat_id] = timestamp
            
            alerts.append(alert_card)
        
        # Store events
        for event in events:
            self.db.insert_event(event, seat_id, track_id, timestamp)
        
        # Always store risk history
        self.db.insert_risk_score(seat_id, track_id, timestamp, risk_score, events, pose_confidence)
        
        return alerts
    
    def get_alerts(self, seat_id: Optional[str] = None, 
                   limit: int = 100) -> List[Dict]:
        """Get recent alerts from database."""
        return self.db.get_alerts(seat_id, limit)


class DatabaseManager:
    """SQLite database manager for Watcher logs."""
    
    def __init__(self, db_path: str = "watcher.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema with WAL mode and long busy timeout."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=60000")
        except Exception:
            pass
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                seat_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL,
                severity TEXT NOT NULL,
                behaviors TEXT NOT NULL,  -- JSON
                explanation TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_frame TEXT,
                acknowledged INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seat_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                count INTEGER DEFAULT 1,
                confidence REAL NOT NULL,
                details TEXT,  -- JSON
                weight REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seat_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL,
                events TEXT,  -- JSON of event types
                pose_confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                config TEXT,  -- JSON
                started_at TEXT NOT NULL,
                ended_at TEXT,
                total_frames INTEGER,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Feedback table (for invigilator actions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                action TEXT NOT NULL,  -- 'acknowledge', 'dismiss', 'escalate'
                invigilator_id TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_seat ON alerts(seat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_seat ON events(seat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_seat ON risk_history(seat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_time ON risk_history(timestamp)")
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def insert_alert(self, alert: AlertCard) -> int:
        """Insert alert into database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO alerts 
            (alert_id, seat_id, camera_id, timestamp, risk_score, severity,
             behaviors, explanation, recommended_action, confidence, evidence_frame)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(alert.alert_id), str(alert.seat_id), str(alert.camera_id), str(alert.timestamp),
            float(alert.risk_score), str(alert.severity),
            json.dumps(alert.behaviors), str(alert.explanation),
            str(alert.recommended_action), float(alert.confidence), str(alert.evidence_frame) if alert.evidence_frame else None
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_event(self, event, seat_id: str, track_id: int, timestamp: float) -> int:
        """Insert behavior event into database."""
        cursor = self.conn.cursor()
        event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
        ts_str = f"{float(timestamp):.3f}s"
        cursor.execute("""
            INSERT INTO events 
            (seat_id, track_id, timestamp, event_type, duration_sec, count,
             confidence, details, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(seat_id), int(track_id), ts_str,
            event_type_str, float(event.duration_sec), int(event.count),
            float(event.confidence), json.dumps(event.details) if event.details else None,
            float(event.weight)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_risk_score(self, seat_id: str, track_id: int, timestamp: float,
                          risk_score: float, events: List,
                          pose_confidence: float = 1.0) -> int:
        """Insert risk score into database."""
        cursor = self.conn.cursor()
        event_types = [(e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)) for e in events]
        ts_str = f"{float(timestamp):.3f}s"
        cursor.execute("""
            INSERT INTO risk_history 
            (seat_id, track_id, timestamp, risk_score, events, pose_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(seat_id), int(track_id), ts_str,
            float(risk_score), json.dumps(event_types), float(pose_confidence)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_source(self, source_id: str, source_type: str, config: Dict) -> int:
        """Log source session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sources (source_id, source_type, config, started_at)
            VALUES (?, ?, ?, ?)
        """, (source_id, source_type, json.dumps(config), datetime.now().isoformat()))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_source_end(self, source_id: str, total_frames: int):
        """Update source session end."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sources SET ended_at = ?, total_frames = ?, status = 'completed'
            WHERE source_id = ?
        """, (datetime.now().isoformat(), total_frames, source_id))
        self.conn.commit()
    
    def get_alerts(self, seat_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent alerts."""
        cursor = self.conn.cursor()
        if seat_id:
            cursor.execute("""
                SELECT * FROM alerts WHERE seat_id = ? ORDER BY timestamp DESC LIMIT ?
            """, (seat_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_events(self, seat_id: Optional[str] = None, 
                   event_type: Optional[str] = None,
                   limit: int = 100) -> List[Dict]:
        """Get recent events."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if seat_id:
            query += " AND seat_id = ?"
            params.append(seat_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_risk_history(self, seat_id: str, limit: int = 1000) -> List[Dict]:
        """Get risk history for a seat."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM risk_history WHERE seat_id = ? ORDER BY timestamp DESC LIMIT ?
        """, (seat_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def export_to_json(self, output_path: str):
        """Export all data to JSON."""
        cursor = self.conn.cursor()
        
        data = {}
        for table in ["alerts", "events", "risk_history", "sources", "feedback"]:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            table_rows = []
            for row in rows:
                r_dict = {}
                for k, v in dict(row).items():
                    if isinstance(v, bytes):
                        r_dict[k] = v.decode('utf-8', errors='ignore')
                    else:
                        r_dict[k] = v
                table_rows.append(r_dict)
            data[table] = table_rows
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Database exported to {output_path}")
    
    def export_to_csv(self, output_dir: str):
        """Export tables to CSV files."""
        import csv
        cursor = self.conn.cursor()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for table in ["alerts", "events", "risk_history", "sources", "feedback"]:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            if rows:
                output_file = Path(output_dir) / f"{table}.csv"
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(rows[0].keys())
                    for row in rows:
                        row_vals = []
                        for key in row.keys():
                            val = row[key]
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', errors='ignore')
                            row_vals.append(val)
                        writer.writerow(row_vals)
                logger.info(f"Exported {table} to {output_file}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_risk_engine(config_path: str = "configs/rules.yaml", fps: float = 25.0) -> RiskEngine:
    """Factory function to create RiskEngine from config."""
    return RiskEngine(config_path, fps)


def create_database_manager(db_path: str = "watcher.db") -> DatabaseManager:
    """Factory function to create DatabaseManager."""
    return DatabaseManager(db_path)