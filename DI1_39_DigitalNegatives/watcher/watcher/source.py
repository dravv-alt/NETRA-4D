"""
Source abstraction for Watcher video ingestion.

Supports multiple source types:
- file://path/to/video.mp4
- webcam://0
- rtsp://camera-url
- http://stream-url
"""

import cv2
import yaml
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import enhancer
from watcher.enhancement import VideoEnhancer, QualityInfo, QualityFlag

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Supported video source types."""
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    HTTP = "http"
    UNKNOWN = "unknown"


@dataclass
class SourceInfo:
    """Metadata about an opened video source."""
    source_type: SourceType
    source_string: str
    is_opened: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    error_message: str = ""


class VideoSource:
    """
    Unified video source handler for Watcher.
    
    Abstracts away the differences between file, webcam, RTSP, and HTTP sources
    while providing a consistent interface for frame reading.
    """
    
    def __init__(self, config_path: str = "configs/source.yaml"):
        """
        Initialize the video source from configuration.
        
        Args:
            config_path: Path to the source configuration YAML file.
        """
        # Resolve config path to absolute
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            # If relative, assume it's relative to current working directory
            config_path_obj = Path.cwd() / config_path_obj
        self.config_path = config_path_obj.resolve()
        self.config = self._load_config()
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_info: Optional[SourceInfo] = None
        self._frame_count = 0
        # Base directory for relative paths in config (project root = parent of configs/)
        self.base_dir = self.config_path.parent.parent
        
        # Initialize video enhancer
        enhancer_config_path = self.base_dir / "configs" / "enhancement.yaml"
        self.enhancer = VideoEnhancer(str(enhancer_config_path))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_source_string(self, source_string: str) -> Tuple[SourceType, str]:
        """
        Parse a source string into type and path/url.
        
        Args:
            source_string: Source string like 'file://path.mp4' or 'webcam://0'
            
        Returns:
            Tuple of (SourceType, path_or_url)
        """
        if source_string.startswith("file://"):
            return SourceType.FILE, source_string[7:]
        elif source_string.startswith("webcam://"):
            return SourceType.WEBCAM, source_string[9:]
        elif source_string.startswith("rtsp://"):
            return SourceType.RTSP, source_string[7:]
        elif source_string.startswith("http://") or source_string.startswith("https://"):
            return SourceType.HTTP, source_string
        else:
            return SourceType.UNKNOWN, source_string
    
    def _open_file(self, path: str) -> bool:
        """Open a video file."""
        # Resolve path relative to base_dir (config file directory)
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.base_dir / path
        
        if not path_obj.exists():
            logger.error(f"Video file not found: {path_obj}")
            return False
            
        self.cap = cv2.VideoCapture(str(path_obj))
        return self.cap.isOpened()
    
    def _open_webcam(self, device_str: str) -> bool:
        """Open a webcam device."""
        try:
            device_index = int(device_str)
        except ValueError:
            logger.error(f"Invalid webcam device index: {device_str}")
            return False
        
        self.cap = cv2.VideoCapture(device_index)
        
        if not self.cap.isOpened():
            return False
        
        # Apply requested resolution from config
        webcam_config = self.config.get("source", {}).get("webcam", {})
        width = webcam_config.get("width", 1280)
        height = webcam_config.get("height", 720)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        return True
    
    def _open_rtsp(self, url: str) -> bool:
        """Open an RTSP stream."""
        rtsp_config = self.config.get("source", {}).get("rtsp", {})
        transport = rtsp_config.get("transport", "tcp")
        timeout = rtsp_config.get("timeout", 10)
        
        # Set FFmpeg options for RTSP
        # Note: OpenCV uses FFmpeg backend for RTSP
        options = f"rtsp_transport={transport}|stimeout={timeout * 1000000}"
        
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Note: OpenCV doesn't directly support setting FFmpeg options easily
        # For production, consider using python-ffmpeg or GStreamer
        
        return self.cap.isOpened()
    
    def _open_http(self, url: str) -> bool:
        """Open an HTTP/MJPEG stream."""
        http_config = self.config.get("source", {}).get("http", {})
        timeout = http_config.get("timeout", 10)
        
        self.cap = cv2.VideoCapture(url)
        return self.cap.isOpened()
    
    def open(self) -> SourceInfo:
        """
        Open the configured video source.
        
        Returns:
            SourceInfo with metadata about the opened source.
        """
        source_config = self.config.get("source", {})
        active_source = source_config.get("active", "file://data/videos/sample.mp4")
        
        source_type, path_or_url = self._parse_source_string(active_source)
        
        success = False
        error_msg = ""
        
        try:
            if source_type == SourceType.FILE:
                success = self._open_file(path_or_url)
                if not success:
                    error_msg = f"Failed to open video file: {path_or_url}"
            elif source_type == SourceType.WEBCAM:
                success = self._open_webcam(path_or_url)
                if not success:
                    error_msg = f"Failed to open webcam device: {path_or_url}"
            elif source_type == SourceType.RTSP:
                success = self._open_rtsp(path_or_url)
                if not success:
                    error_msg = f"Failed to open RTSP stream: {path_or_url}"
            elif source_type == SourceType.HTTP:
                success = self._open_http(path_or_url)
                if not success:
                    error_msg = f"Failed to open HTTP stream: {path_or_url}"
            else:
                error_msg = f"Unknown source type: {active_source}"
        except Exception as e:
            error_msg = f"Exception opening source: {str(e)}"
            success = False
        
        # Gather metadata if opened successfully
        width = 0
        height = 0
        fps = 0.0
        frame_count = 0
        
        if success and self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # For live sources, frame_count may be 0 or unreliable
            if source_type in (SourceType.WEBCAM, SourceType.RTSP, SourceType.HTTP):
                frame_count = 0
        
        self.source_info = SourceInfo(
            source_type=source_type,
            source_string=active_source,
            is_opened=success,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            error_message=error_msg
        )
        
        return self.source_info
    
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """
        Read the next frame from the source and apply enhancement.
        
        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
            # Apply enhancement
            frame, quality = self.enhancer.enhance(frame)
            # Store quality info for potential use
            self._last_quality = quality
        return ret, frame
    
    def read_frames(self, num_frames: int) -> list:
        """
        Read a specified number of frames.
        
        Args:
            num_frames: Number of frames to read.
            
        Returns:
            List of successfully read frames.
        """
        frames = []
        for _ in range(num_frames):
            ret, frame = self.read_frame()
            if not ret:
                logger.warning(f"Failed to read frame {self._frame_count + 1}")
                break
            frames.append(frame)
        return frames
    
    def save_frame(self, frame: Any, output_path: str) -> bool:
        """
        Save a frame to disk.
        
        Args:
            frame: Frame to save (numpy array).
            output_path: Output file path.
            
        Returns:
            True if saved successfully.
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cv2.imwrite(str(output_path_obj), frame)
            logger.info(f"Saved frame to {output_path_obj}")
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False
    
    def close(self):
        """Close the video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Video source closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    @property
    def frame_count(self) -> int:
        """Number of frames read so far."""
        return self._frame_count


def create_source(config_path: str = "configs/source.yaml") -> VideoSource:
    """
    Factory function to create a VideoSource from config.
    
    Args:
        config_path: Path to source configuration.
        
    Returns:
        Configured VideoSource instance.
    """
    return VideoSource(config_path)