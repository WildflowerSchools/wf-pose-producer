import importlib
import os

from modeltools import ModelManifest


class ModelHandler:
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.manifest = None
        self.model_class = None
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        print(f"model is in {model_dir}")
        print(f"gpu_id is {gpu_id}")
        with open(os.path.join(model_dir, "manifest.yml"), 'r') as manifest_fp:
            _raw = manifest_fp.read()
            self.manifest = ModelManifest.from_yaml(_raw)
            mod_name, cls_name = self. manifest.model_class.split(":")
            module = importlib.import_module(mod_name)
            self.model_class = getattr(module, cls_name)
            self.model = self.model_class(gpu_id, self.manifest.args_dict())
        self.initialized = True

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print('=' * 80)
        print(context)
        print(dir(context))
        print(context.manifest)
        print(dir(context.manifest))
        print('=' * 80)
        return self.model.process_batch(data)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
