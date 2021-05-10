from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class FrameExtractRequest:
    environment_name: str
    timestamp: str
    duration: str
