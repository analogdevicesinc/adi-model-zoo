# Copyright © 2025 Analog Devices, Inc. All Rights Reserved. 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AI8X Inference Module

This module provides functionality to run AI8X model inference using
the ai8x-training framework to generate expected outputs for testing.
"""

import numpy as np
from pathlib import Path

import os
import sys
import importlib
import torch
from collections import OrderedDict

sys.path.append(os.path.join(os.getcwd(), "data", "model"))

from . import ai8x

def ai8x_softmax_2(x):
    """
    Softmax implementation similar to izer: y_i = 2^(x_i/16384) / sum(2^(x_j/16384))
    """
    bx = np.power(2, x/16384)
    sm = bx.sum(axis=0)
    return bx / sm

class AI8XInference:
    """Wrapper for ai8x-training framework to generate expected outputs."""

    def __init__(self,
                 model_class: str,
                 model_module: str,
                 model_checkpoint_path: Path,
                 hardware: str = "MAX78002",
                 module: bool = False):
    
        self.model = model_class
        self.model_module = model_module
        self.model_checkpoint_path = model_checkpoint_path
        self.hardware = hardware

        if not self.model_checkpoint_path.exists():
            raise RuntimeError(
                f"Model checkpoint not found at {self.model_checkpoint_path}. "
                "Please ensure the model checkpoint path is correct and the file exists."
            )
        
        self.device = 'cpu'
        print("Working with device:", self.device)

        ai8x.set_device(device=87 if self.hardware == "MAX78002" else 85, simulate=True, round_avg=False)
        
        try:
            mod = importlib.import_module(self.model_module)
            self.model = getattr(mod, self.model)(bias=True, quantize_activation=True, weight_bits=8, bias_bits=8)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Failed to import model {self.model} from {self.model_module}: {e}"
            )
        
        self.model.to(self.device)
        self.model.eval()

        trained_checkpoint_path = self.model_checkpoint_path
        checkpoint = torch.load(trained_checkpoint_path, map_location=self.device, weights_only=False)

        if module:
            sd = checkpoint["state_dict"]
            new_sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
            self.model.load_state_dict(new_sd, strict=False)
        else:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            
        ai8x.update_model(self.model)

        print(f"Loaded model weights from {trained_checkpoint_path}")

    def generate_expected_output(self, 
                                 sample_input: np.ndarray,
                                 ) -> np.ndarray:

        x = torch.from_numpy(sample_input).unsqueeze(0).to(self.device)
        y = self.model(x)

        return y
