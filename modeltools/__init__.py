from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
import yaml


@dataclass_json
@dataclass
class FileRef:
    identifier: str
    name: str


@dataclass_json
@dataclass
class Argument:
    name: str
    value: str


@dataclass_json
@dataclass
class ModelManifest:
    version: str
    name: str
    files: List[FileRef] = field(default_factory=list)
    args: List[Argument] = field(default_factory=list)

    def to_yaml(self):
        return yaml.dump(self.to_dict())
