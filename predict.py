from cog import BasePredictor, Input
import torch
import random
import torch
import os
import sys
import folder_paths
import numpy as np
from tool import ToolUtils
import base64
from io import BytesIO
import boto3
import PIL
from custom_nodes.ComfyUI_InstantID.InstantID import (InstantIDModelLoader,InstantIDFaceAnalysis,ApplyInstantIDAdvanced)
from comfy_extras.nodes_model_advanced import (RescaleCFG)
from custom_nodes.ComfyUI_Tool_Yolo_Rectangle_Cropper.nodes import (ToolYoloDetectionNode)

class Predictor(BasePredictor):
    def setup(self):
        self.prefix = "hello"

    def predict(self, text: str = Input(description="Text to prefix with 'hello '")) -> str:
        return self.prefix + " " + text