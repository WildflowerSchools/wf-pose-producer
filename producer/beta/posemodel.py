from dataclasses import dataclass, field
from typing import List


@dataclass
class Keypoint:
    x: float
    y: float
    quality: float


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

@dataclass
class Pose2D:
    track_label: str
    keypoints: List[Keypoint] = field(default_factory=list)
    quality: float = None
    bbox: List[float] = None
    box_id: str = None
    pose_id: str = None


@dataclass
class PoseFrame:
    image_id: str
    timestamp: str
    assignment_id: str
    environment_id: str
    image_name: str
    video_path: str
    poses: List[Pose2D] = None
