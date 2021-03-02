"""

video ->  detection -> box-tracker -> estimator -> pose-tracker ---> pose-upload
                                                                 |-> pose-local

"""
from producer.pubsub import BunnyTrail



exchanges = [
    BunnyTrail("videos", "video", "extract-frames"),
    BunnyTrail("images", "detection", "detector"),
    BunnyTrail("boxes", "estimator", "estimation"),
    BunnyTrail("boxes", "box-tracker", "catalog"),
    # BunnyTrail("boxes", "box-local", "catalog"),
    BunnyTrail("poses", "pose-tracker", "2dpose"),
    BunnyTrail("poses", "pose-deduplicate", "imageid"),
    # BunnyTrail("poses", "pose-upload", "2dposeset"),
    BunnyTrail("poses", "pose-local", "2dposeset"),
    BunnyTrail("errors", "errors", "error"),
]
