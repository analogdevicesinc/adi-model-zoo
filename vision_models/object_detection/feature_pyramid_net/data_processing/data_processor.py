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

from PIL import Image
from torchvision import transforms
import numpy as np
import struct

INPUT_TENSOR_SHAPE = (256, 320)

class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
    
def _pack_bitstream(input):
    """
    Pack bitstream from numpy array of shape (3, H, W)
    """
    merged_array = (
        (input[2].astype(np.uint32) & 0xFF) << 16 |   # B
        (input[1].astype(np.uint32) & 0xFF) << 8 |    # G
        (input[0].astype(np.uint32) & 0xFF)           # R
    )

    merged_array = merged_array[np.newaxis, ...]  # shape (1, 256, 320)
    flat = merged_array.flatten()

    return flat

def preprocess(fname, pack=True, **kwargs):
    """
    Preprocess input of model
    """

    img = Image.open(fname).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(INPUT_TENSOR_SHAPE),
        transforms.ToTensor(),
        normalize()
    ])

    img_tensor = transform(img)
    if pack:
        return _pack_bitstream(img_tensor.numpy())
    else:
        return img_tensor.numpy()

def postprocess(fname, **kwargs):
    """
    Load and postprocess detection results from a numpy file.
    
    Args:
        fname: Path to numpy file with shape (1, 120) or (120,) and dtype uint32
        
    Returns:
        tuple: (List of bboxes, List of class_idx, List of scores)
               - Each bbox has shape (1, 4) containing [x1, y1, x2, y2] as floats
               - Each class_idx has shape (1, 1) containing the class index
               - Each score has shape (1, 1) containing the confidence score as float
    """
    # Load the numpy array
    data = np.load(fname)
    
    # Flatten if needed to ensure shape is (25,)
    if data.ndim == 2:
        data = data.flatten()
    
    # Ensure dtype is uint32
    data = data.astype(np.uint32)
    
    # Initialize output lists
    bboxes = []
    class_indices = []
    scores = []
    
    # Process every 6 elements
    for i in range(0, len(data), 6):
        if i + 5 >= len(data):
            break
            
        # Extract class_idx and bbox values
        class_idx = data[i]
        bbox_hex = data[i+1:i+5]
        score_hex = data[i+5]
        
        # Convert bbox values from uint32 (IEEE754) to float
        bbox_float = np.array([
            struct.unpack('!f', struct.pack('!I', val))[0] 
            for val in bbox_hex
        ])

        # Convert score value from uint32 (IEEE754) to float
        score_float = np.array([
            struct.unpack('!f', struct.pack('!I', score_hex))[0]
        ])
        
        # Skip if class_idx is 0 or all bbox values are zero
        if class_idx == 0 or np.all(bbox_float == 0):
            continue
        
        # Add to output lists with proper shapes
        bboxes.append(bbox_float.reshape(1, 4))
        class_indices.append(np.array([[class_idx]]))
        scores.append(score_float.reshape(1, 1))
    
    return (bboxes, class_indices, scores)

if __name__ == "__main__":
    print("Example usage in playground.ipynb")
