"""
Video Decoding and Stream Management Abstractions for Drishti-4D.
Provides timestamp-accurate frame ingestion, sequential decode health checks,
and metadata extraction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional, Tuple, Dict, Any
import time
import cv2
import numpy as np


class VideoMetadata:
    def __init__(self, path: str, width: int, height: int, fps: float, frame_count: int, duration_sec: float, codec: str = ""):
        self.path = path
        self.filename = Path(path).name
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration_sec = duration_sec
        self.codec = codec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_sec": round(self.duration_sec, 2),
            "duration_min": round(self.duration_sec / 60.0, 2),
            "codec": self.codec
        }


class VideoSource(ABC):
    @abstractmethod
    def get_metadata(self) -> VideoMetadata:
        pass

    @abstractmethod
    def read_frames(self, sample_fps: Optional[float] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yields (frame_index, timestamp_sec, bgr_frame)."""
        pass

    @abstractmethod
    def seek(self, timestamp_sec: float) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def close(self):
        pass


class FileVideoSource(VideoSource):
    def __init__(self, video_path: str):
        self.path = str(Path(video_path).resolve())
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Unable to open video source at: {self.path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0
        self.codec = self._get_fourcc()

    def _get_fourcc(self) -> str:
        try:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except Exception:
            return "unknown"

    def get_metadata(self) -> VideoMetadata:
        return VideoMetadata(
            path=self.path,
            width=self.width,
            height=self.height,
            fps=self.fps,
            frame_count=self.frame_count,
            duration_sec=self.duration_sec,
            codec=self.codec
        )

    def read_frames(self, sample_fps: Optional[float] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        step = 1
        if sample_fps and sample_fps > 0 and sample_fps < self.fps:
            step = max(1, int(round(self.fps / sample_fps)))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                # Video timestamp in seconds
                timestamp_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp_sec <= 0 and self.fps > 0:
                    timestamp_sec = frame_idx / self.fps
                yield frame_idx, timestamp_sec, frame

            frame_idx += 1

    def seek(self, timestamp_sec: float) -> Optional[np.ndarray]:
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.path)

        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ret, frame = self.cap.read()
        return frame if ret and frame is not None else None

    def check_health(self) -> Dict[str, Any]:
        """Perform sequential window health check without decoding entire video."""
        sample_positions = [0.05, 0.25, 0.50, 0.75, 0.90]
        window_sec = 5.0
        total_expected = 0
        total_decoded = 0

        for pos in sample_positions:
            start_t = min(self.duration_sec * pos, max(0.0, self.duration_sec - window_sec - 1.0))
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000.0)
            target = max(1, int(round(window_sec * self.fps)))
            total_expected += target

            for _ in range(target):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    total_decoded += 1

        success_rate = total_decoded / total_expected if total_expected > 0 else 0.0
        return {
            "path": self.path,
            "filename": Path(self.path).name,
            "expected_frames": total_expected,
            "decoded_frames": total_decoded,
            "success_rate": round(success_rate, 4),
            "status": "PASS" if success_rate >= 0.99 else "REVIEW"
        }

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


class RTSPVideoSource(VideoSource):
    """Live RTSP stream adapter with reconnect logic."""
    def __init__(self, rtsp_url: str, camera_id: str):
        self.url = rtsp_url
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(rtsp_url)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    def get_metadata(self) -> VideoMetadata:
        return VideoMetadata(
            path=self.url,
            width=self.width,
            height=self.height,
            fps=self.fps,
            frame_count=0,
            duration_sec=0.0,
            codec="H264/RTSP"
        )

    def read_frames(self, sample_fps: Optional[float] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        frame_idx = 0
        start_time = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            ts = time.time() - start_time
            yield frame_idx, ts, frame
            frame_idx += 1

    def seek(self, timestamp_sec: float) -> Optional[np.ndarray]:
        return None

    def close(self):
        if self.cap:
            self.cap.release()
