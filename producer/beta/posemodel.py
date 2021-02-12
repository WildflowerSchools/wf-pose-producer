from dataclasses import dataclass, field
# from datetime import datetime
import json
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
    box_id: float = str


@dataclass
class PoseFrame:
    image_id: str
    timestamp: str
    assignment_id: str
    environment_id: str
    image_name: str
    video_path: str
    poses: List[Pose2D] = None




# from marshmallow import Schema, fields

# from producer import settings as s


# NaN = float('nan')
#
#
# class Keypoint(Schema):
#     coordinates = fields.List(fields.Float(allow_nan=True, default=NaN))
#     quality = fields.Float(allow_nan=True, default=NaN)

#
# class PoseHoneycomb(Schema):
#     timestamp = fields.String()
#     camera = fields.UUID()
#     pose_model = fields.UUID()
#     source = fields.UUID()
#     source_type = fields.String()
#     keypoints = fields.List(fields.Nested(Keypoint()))
#     track_label = fields.String()
#     tags = fields.List(fields.String())
#     quality = fields.Float(allow_nan=True, default=NaN)
#
#
# class PoseLocal(PoseHoneycomb):
#     bbox = fields.List(fields.Float())
#

# class PoseFrame(Schema):
#     image_id = fields.UUID()
#     image_name = fields.String()
#     video_path = fields.String()
#     assignment_id = fields.UUID()
#     environment_id = fields.UUID()
#     poses = fields.List(fields.Nested(PoseLocal()))
#
